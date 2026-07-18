"""
BEV Raw Labeled — dense mesh top-down render + ins_pcds bbox overlay.

Background rendering:
  - Loads ScanNet _vh_clean_2.ply mesh (dense textured triangles).
  - Renders top-down via numpy Z-buffer rasterizer: each triangle rasterized
    per-pixel with per-vertex color interpolation (Gouraud shading).
  - Ceiling clipped at floor + 2m absolute height so floor stays visible.
  - Areas outside mesh coverage supplemented with ins_pcds point projection.

Bounding boxes:
  - Computed from ins_pcds (same coordinate space, guaranteed alignment).
  - Dashed orange "★" = anchor, solid colored A/B/C = candidates.

Prompt:
  - Box-to-candidate mapping text.
  - Coordinate frame: X=East(+), Y=North(+).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

from .base import VLMInputFormat, MASK3D_DIR, _pil_b64

SCANNET_ORIGIN = Path("/data/knuvi/bosung/scannet_origin")

try:
    from seqvlm.detector_free_rc import _load_axis_alignment
    HAS_AXIS = True
except ImportError:
    HAS_AXIS = False

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

CAND_COLORS = [
    (59, 130, 246), (220, 50,  50), (34, 160,  80), (160,  70, 230),
    (230, 110,  20), (20, 170, 160), (220,  60, 140), (120, 180,  30),
]
ANC_FILL   = (251, 146,  60)
ANC_BORDER = (180,  55,   0)
BG_COLOR   = (235, 235, 235)   # canvas background


def _font(size, bold=True):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)
    except Exception:
        return ImageFont.load_default()


def _apply_axis(pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
    R = mat[:3, :3].astype(np.float32)
    t = mat[:3, 3].astype(np.float32)
    return (R @ pts.astype(np.float32).T).T + t


def _hull(pts2d: np.ndarray):
    if len(pts2d) < 3:
        return None
    try:
        h = ConvexHull(pts2d)
        return pts2d[h.vertices]
    except Exception:
        return None


def _load_mesh(scene_id: str):
    path = SCANNET_ORIGIN / scene_id / f"{scene_id}_vh_clean_2.ply"
    ply   = PlyData.read(str(path))
    v     = ply['vertex']
    verts = np.column_stack([v['x'], v['y'], v['z'],
                             v['red'], v['green'], v['blue']]).astype(np.float32)
    faces = np.vstack(ply['face']['vertex_indices']).astype(np.int32)
    return verts, faces


def render_bev_mesh(
    scene_id: str,
    ins_pcds: np.ndarray,
    axis_mat: np.ndarray,
    candidates: list[int],
    anchors: list[int],
    img_size: int = 900,
    ceil_clip_m: float = 2.0,
) -> tuple[Image.Image | None, dict]:
    """
    Dense mesh top-down render using PIL painter's algorithm:
    1. Load _vh_clean_2.ply, apply axis alignment.
    2. Clip triangles above floor + ceil_clip_m (removes ceiling).
    3. Sort triangles by mean Z DESCENDING: furniture tops drawn first,
       floor drawn last → floor fills every gap between objects.
    4. Each triangle filled with average vertex RGB (flat shading, fast via PIL).
    5. Viewport = union of mesh + all ins_pcds extents.
    6. Supplement gray areas outside mesh with ins_pcds point projection.
    """
    verts, faces = _load_mesh(scene_id)
    xyz_al = _apply_axis(verts[:, :3], axis_mat)
    rgb    = verts[:, 3:6].astype(np.uint8)

    # ── Ceiling clip ─────────────────────────────────────────────────────────
    z_floor = xyz_al[:, 2].min()
    z_clip  = z_floor + ceil_clip_m
    tri_z   = xyz_al[faces][:, :, 2].mean(axis=1)
    faces   = faces[tri_z < z_clip]

    # ── Viewport: union of mesh + ins_pcds ────────────────────────────────
    xmin_v, ymin_v = xyz_al[:, 0].min(), xyz_al[:, 1].min()
    xmax_v, ymax_v = xyz_al[:, 0].max(), xyz_al[:, 1].max()
    # ins_pcds are already in axis-aligned space — no axis_mat needed
    for i in range(len(ins_pcds)):
        if ins_pcds[i].shape[0] == 0: continue
        xy = ins_pcds[i][:, :2]   # already aligned
        xmin_v = min(xmin_v, xy[:, 0].min()); ymin_v = min(ymin_v, xy[:, 1].min())
        xmax_v = max(xmax_v, xy[:, 0].max()); ymax_v = max(ymax_v, xy[:, 1].max())

    span = max(xmax_v - xmin_v, ymax_v - ymin_v, 1.0)
    pad  = span * 0.05
    M = 8; pw = img_size - 2*M; ph = img_size - 2*M

    def to_px(x_world, y_world):
        px = (np.asarray(x_world) - xmin_v + pad) / (span + 2*pad) * pw + M
        py = (1.0 - (np.asarray(y_world) - ymin_v + pad) / (span + 2*pad)) * ph + M
        return px, py

    coord_info = {"to_px": to_px, "px_per_m": pw / (span + 2*pad)}

    # ── Project all vertices ─────────────────────────────────────────────────
    vx, vy = to_px(xyz_al[:, 0], xyz_al[:, 1])

    # ── Sort DESCENDING by mean Z: furniture top first, floor last ───────────
    mean_z = xyz_al[faces][:, :, 2].mean(axis=1)
    order  = np.argsort(-mean_z)

    # ── PIL painter render ───────────────────────────────────────────────────
    img  = Image.new("RGB", (img_size, img_size), BG_COLOR)
    draw = ImageDraw.Draw(img)

    for fi in order:
        i0, i1, i2 = faces[fi]
        poly = [(int(vx[i0]), int(vy[i0])),
                (int(vx[i1]), int(vy[i1])),
                (int(vx[i2]), int(vy[i2]))]
        r = int((int(rgb[i0,0]) + int(rgb[i1,0]) + int(rgb[i2,0])) // 3)
        g = int((int(rgb[i0,1]) + int(rgb[i1,1]) + int(rgb[i2,1])) // 3)
        b = int((int(rgb[i0,2]) + int(rgb[i1,2]) + int(rgb[i2,2])) // 3)
        draw.polygon(poly, fill=(r, g, b))

    # ── Supplement: ins_pcds points for areas outside mesh ───────────────────
    # ins_pcds are already in axis-aligned space — use directly, no axis_mat
    canvas = np.array(img)
    bg     = np.array(BG_COLOR, dtype=np.uint8)
    for i in range(len(ins_pcds)):
        if ins_pcds[i].shape[0] == 0: continue
        pts = ins_pcds[i][:, :3]   # already aligned
        mask_z = pts[:, 2] < z_clip
        pts = pts[mask_z]
        if len(pts) == 0: continue
        rgb_p = ins_pcds[i][mask_z, 3:6].astype(np.uint8)
        pvx, pvy = to_px(pts[:, 0], pts[:, 1])
        pvx = np.round(pvx).astype(np.int32)
        pvy = np.round(pvy).astype(np.int32)
        mask = (pvx >= 0) & (pvx < img_size) & (pvy >= 0) & (pvy < img_size)
        pvx = pvx[mask]; pvy = pvy[mask]; rgb_p = rgb_p[mask]
        # Paint with 2px radius for visibility
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(0,0)]:
            px2 = pvx+dx; py2 = pvy+dy
            m2 = (px2 >= 0) & (px2 < img_size) & (py2 >= 0) & (py2 < img_size)
            is_bg = np.all(canvas[py2[m2], px2[m2]] == bg, axis=1)
            canvas[py2[m2][is_bg], px2[m2][is_bg]] = rgb_p[m2][is_bg]

    return Image.fromarray(canvas, "RGB"), coord_info


def draw_bbox_overlays(
    img: Image.Image,
    ins_pcds: np.ndarray,
    axis_mat: np.ndarray,
    candidates: list[int],
    anchors: list[int],
    coord_info: dict,
    lbl_size: int = 16,
) -> None:
    """Draw AABB bboxes: dashed orange for anchor, solid color for candidates."""
    to_px = coord_info["to_px"]
    draw  = ImageDraw.Draw(img)
    fnt   = _font(lbl_size)

    def inst_aabb(idx):
        if idx >= len(ins_pcds) or ins_pcds[idx].shape[0] == 0:
            return None
        # ins_pcds are already axis-aligned — use directly
        pts = ins_pcds[idx][:, :2]
        vx, vy = to_px(pts[:, 0], pts[:, 1])
        return float(vx.min()), float(vy.min()), float(vx.max()), float(vy.max())

    def dashed_rect(x0, y0, x1, y1, color, w=2, dash=8):
        for x in range(int(x0), int(x1), dash * 2):
            draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=w)
            draw.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=w)
        for y in range(int(y0), int(y1), dash * 2):
            draw.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=w)
            draw.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=w)

    def badge(x, y, text, bg):
        bb = draw.textbbox((0, 0), text, font=fnt)
        tw, th = bb[2] - bb[0] + 6, bb[3] - bb[1] + 4
        # keep badge inside image
        x = max(2, min(int(x), img.width - tw - 4))
        y = max(2, min(int(y), img.height - th - 4))
        draw.rectangle([x, y, x + tw, y + th], fill=bg, outline=(255, 255, 255), width=1)
        draw.text((x + 3, y + 2), text, fill=(255, 255, 255), font=fnt)

    # Anchor
    for ai in anchors:
        bb = inst_aabb(ai)
        if bb is None: continue
        x0, y0, x1, y1 = bb
        dashed_rect(x0, y0, x1, y1, ANC_BORDER, w=3)
        badge(x0, max(y0 - lbl_size - 8, 2), "★", ANC_FILL)

    # Candidates
    for rank, c in enumerate(candidates):
        bb = inst_aabb(c)
        if bb is None: continue
        col = CAND_COLORS[rank % len(CAND_COLORS)]
        ltr = chr(65 + rank) if rank < 26 else str(rank)
        x0, y0, x1, y1 = bb
        draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=col, width=3)
        badge(x0, y0, ltr, col)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are analyzing a top-down Bird's Eye View (BEV) of a 3D indoor scene.

The image shows detected objects projected top-down with their actual 3D colors.
Bounding boxes mark specific objects:
  - Dashed orange box labeled "★" = ANCHOR reference object
  - Solid colored boxes labeled A, B, C, ... = CANDIDATE objects to choose from

COORDINATE FRAME (consistent between image and 3D coordinates):
  UP    = North (+Y)     RIGHT = East  (+X)
  DOWN  = South (-Y)     LEFT  = West  (-X)

Each candidate has 3D spatial coordinates relative to the anchor:
  anchor_delta=(dx, dy, dz): offset from anchor center in meters
    - dx>0: East of anchor (right in BEV),   dx<0: West (left)
    - dy>0: North of anchor (up in BEV),     dy<0: South (down)
    - dz>0: above anchor,                    dz<0: below
  dist: horizontal distance from anchor (meters)
  z: absolute height (meters)
  size: (width × depth × height) in meters

Use BOTH the BEV image (visual layout) and the 3D coordinates (precise positions) to reason.

Respond in JSON:
{
  "process": "step-by-step spatial + visual reasoning",
  "selected_key": "A",
  "confidence": 0.0
}"""

USER_PROMPT = """Query: {query}

3D Spatial Information:
{box_mapping}

Coordinate frame: X=East(+)/West(-), Y=North(+)/South(-), Z=Up(+)/Down(-).
The BEV image above shows the same scene from directly above — use it together with the coordinates.

Which candidate (A, B, C, ...) best matches the query?
Respond with JSON: {{"process": "...", "selected_key": "A", "confidence": 0.0}}"""


class BevRawLabeledFormat(VLMInputFormat):
    name = "bev_raw_labeled"
    source_paper = "Ours — instance point cloud BEV with bbox overlay"
    paper_faithful = "ours"

    def build(self, query, scene_id, candidates, anchors,
              relation_info, frame_data, scene_data, config):

        labels   = scene_data.get("labels", [])
        img_size = config.get("bev_img_size", 900)
        images   = []
        diffs    = []

        try:
            axis_mat = _load_axis_alignment(scene_id) if HAS_AXIS else np.eye(4)
            ins_pcds = scene_data.get("ins_pcds")
            if ins_pcds is None:
                ins_pcds = np.load(f"{MASK3D_DIR}/{scene_id}.npz",
                                   allow_pickle=True)['ins_pcds']

            img, coord_info = render_bev_mesh(
                scene_id, ins_pcds, axis_mat, candidates, anchors, img_size=img_size
            )
            if img is not None:
                draw_bbox_overlays(img, ins_pcds, axis_mat, candidates, anchors, coord_info)
                images.append(_pil_b64(img, fmt="JPEG", quality=92))
            else:
                diffs.append("BEV render returned None")
        except Exception as e:
            import traceback
            diffs.append(f"Render error: {e}\n{traceback.format_exc()[-300:]}")

        # Prompt: box mapping with 3D coordinates
        locs = scene_data.get("locs", [])
        lines = []

        # Anchor position
        anchor_pos = None
        if anchors:
            ai = anchors[0]
            albl = labels[ai][:24] if ai < len(labels) else "?"
            if ai < len(locs):
                ax, ay, az = locs[ai][:3]
                anchor_pos = np.array([ax, ay, az], dtype=np.float32)
                lines.append(f"  ★ (dashed orange) = anchor: {albl}"
                             f"  center=({ax:.2f},{ay:.2f},{az:.2f})")
            else:
                lines.append(f"  ★ (dashed orange) = anchor: {albl}")

        # Candidates with anchor-relative coords
        for rank, c in enumerate(candidates):
            ltr = chr(65 + rank) if rank < 26 else str(rank)
            clbl = labels[c][:24] if c < len(labels) else "?"
            if c < len(locs):
                cx, cy, cz, wx, wy, wz = locs[c][:6]
                if anchor_pos is not None:
                    dx = round(float(cx - anchor_pos[0]), 2)
                    dy = round(float(cy - anchor_pos[1]), 2)
                    dz = round(float(cz - anchor_pos[2]), 2)
                    dist = round(float(np.sqrt(dx**2 + dy**2)), 2)
                    lines.append(
                        f"  {ltr} (solid box) = candidate: {clbl}"
                        f"  anchor_delta=({dx:+.2f},{dy:+.2f},{dz:+.2f})"
                        f"  dist={dist}m  z={cz:.2f}m"
                        f"  size=({wx:.2f}×{wy:.2f}×{wz:.2f})m"
                    )
                else:
                    lines.append(
                        f"  {ltr} (solid box) = candidate: {clbl}"
                        f"  center=({cx:.2f},{cy:.2f},{cz:.2f})"
                        f"  size=({wx:.2f}×{wy:.2f}×{wz:.2f})m"
                    )
            else:
                lines.append(f"  {ltr} (solid box) = candidate: {clbl}")

        box_mapping = "\n".join(lines) or "  (unavailable)"
        prompt_text = USER_PROMPT.format(query=query, box_mapping=box_mapping)

        meta = self._base_metadata(
            uses_real_rgb=True,
            uses_rendered_image=True,
            uses_spatial_text=True,
            uses_visual_prompt=True,
            uses_multiview=False,
            uses_query_decomposition=False,
            uses_multiscale=False,
            bev_type="instance_pcd_topdown_convex_hull",
            coordinate_frame="X=East Y=North Z=Up (ScanNet axis-aligned)",
            implementation_difference_from_paper=diffs,
        )
        return {
            "images":   images,
            "system":   SYSTEM_PROMPT,
            "prompt":   prompt_text,
            "metadata": meta,
        }

    @staticmethod
    def parse_response(raw: str, n_cands: int) -> int:
        from .base import parse_selected_key
        letters = [chr(65 + i) if i < 26 else str(i) for i in range(n_cands)]
        key = parse_selected_key(raw, set(letters))
        if key and key in letters:
            return letters.index(key)
        return -1
