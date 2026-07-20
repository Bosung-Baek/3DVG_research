#!/usr/bin/env python3
"""Create slide-ready qualitative assets.

The assets are designed for presentation slides rather than dense paper
figures. They use real locked query/result records and visualize predictions on
BEV/3D overlays so that success and failure cases can be compared at a glance.
"""

from __future__ import annotations

import json
import math
import re
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from input_formats.bev_raw_labeled import render_bev_mesh

OUT_DIR = ROOT / "experiments" / "presentation_assets"

GREEN = "#16A34A"
RED = "#DC2626"
BLUE = "#2563EB"
ORANGE = "#F97316"
PURPLE = "#7C3AED"
GRAY = "#64748B"
BLACK = "#111827"
BG = "#F8FAFC"
MASK3D_DIR = Path("/data/knuvi/bosung/Mask3d/scannet200")
SCANNET_ORIGIN = Path("/data/knuvi/bosung/scannet_origin")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_case(path: str, query_id: bool = False) -> dict[int, dict]:
    rows = read_jsonl(ROOT / path)
    out = {}
    for row in rows:
        key = int(row["query_id"]) if query_id else int(row["case"])
        out[key] = row
    return out


def parse_tuple(text: str) -> tuple[float, float, float]:
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    if len(nums) < 3:
        raise ValueError(f"Could not parse tuple: {text}")
    return (nums[0], nums[1], nums[2])


def parse_size(text: str) -> tuple[float, float, float]:
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    if len(nums) < 3:
        raise ValueError(f"Could not parse size: {text}")
    return (nums[0], nums[1], nums[2])


def candidate_letter(i: int) -> str:
    return chr(ord("A") + i)


def font(size: int, bold: bool = True):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.load_default()


def load_axis_alignment(scene_id: str) -> np.ndarray:
    txt = SCANNET_ORIGIN / scene_id / f"{scene_id}.txt"
    mat = np.eye(4, dtype=np.float32)
    if not txt.exists():
        return mat
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "axisAlignment" not in line:
            continue
        vals = [float(x) for x in line.split("=")[-1].strip().split()]
        if len(vals) == 16:
            return np.array(vals, dtype=np.float32).reshape(4, 4)
    return mat


def instance_aabb_px(ins_pcds: np.ndarray, idx: int, to_px):
    if idx >= len(ins_pcds) or ins_pcds[idx].shape[0] == 0:
        return None
    pts = ins_pcds[idx][:, :2]
    px, py = to_px(pts[:, 0], pts[:, 1])
    return float(px.min()), float(py.min()), float(px.max()), float(py.max())


def draw_dashed_rect(draw: ImageDraw.ImageDraw, box, color, width=5, dash=16):
    x0, y0, x1, y1 = box
    for x in range(int(x0), int(x1), dash * 2):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=width)
    for y in range(int(y0), int(y1), dash * 2):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=width)


def draw_badge(draw: ImageDraw.ImageDraw, xy, text: str, fill, text_fill=(255, 255, 255), size=30):
    fnt = font(size, True)
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0] + 18, bbox[3] - bbox[1] + 12
    x = max(8, min(int(x), 1100 - tw - 8))
    y = max(8, min(int(y), 1100 - th - 8))
    draw.rounded_rectangle([x, y, x + tw, y + th], radius=8, fill=fill, outline=(255, 255, 255), width=2)
    draw.text((x + 9, y + 6), text, fill=text_fill, font=fnt)


def instance_center(ins_pcds: np.ndarray, idx: int) -> np.ndarray | None:
    if idx >= len(ins_pcds) or ins_pcds[idx].shape[0] == 0:
        return None
    pts = ins_pcds[idx][:, :3]
    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0


def find_nearest_label_instance(scene_id: str, center: tuple[float, float, float], keywords: list[str]) -> int | None:
    data = np.load(MASK3D_DIR / f"{scene_id}.npz", allow_pickle=True)
    ins_pcds = data["ins_pcds"]
    labels = data["ins_labels"] if "ins_labels" in data.files else []
    q = np.array(center, dtype=np.float32)
    best = None
    best_dist = float("inf")
    for idx, label in enumerate(labels):
        label_text = str(label).lower()
        if keywords and not any(k.lower() in label_text for k in keywords):
            continue
        c = instance_center(ins_pcds, idx)
        if c is None:
            continue
        dist = float(np.linalg.norm(c - q))
        if dist < best_dist:
            best = idx
            best_dist = dist
    return best


def make_real_bev_overlay(
    scene_id: str,
    candidate_ids: list[int],
    anchor_ids: list[int],
    gt_id: int | None,
    pred_id: int | None,
    result_badge: str,
    img_size: int = 1100,
    crop_local: bool = False,
    pred_success: bool = False,
):
    data = np.load(MASK3D_DIR / f"{scene_id}.npz", allow_pickle=True)
    ins_pcds = data["ins_pcds"]
    axis_mat = load_axis_alignment(scene_id)
    render_candidate_ids = list(candidate_ids)
    for extra_id in (pred_id, gt_id):
        if extra_id is not None and extra_id not in render_candidate_ids:
            render_candidate_ids.append(extra_id)
    base, coord_info = render_bev_mesh(
        scene_id,
        ins_pcds,
        axis_mat,
        render_candidate_ids,
        anchor_ids,
        img_size=img_size,
    )
    img = base.convert("RGBA")
    overlay = ImageDraw.Draw(img, "RGBA")
    to_px = coord_info["to_px"]

    visible_boxes = []
    for aid in anchor_ids:
        box = instance_aabb_px(ins_pcds, aid, to_px)
        if box is None:
            continue
        visible_boxes.append(box)
        draw_dashed_rect(overlay, box, (249, 115, 22, 255), width=5)
        draw_badge(overlay, (box[0], box[1] - 48), "anchor", (249, 115, 22, 235), size=24)

    for rank, cid in enumerate(render_candidate_ids):
        box = instance_aabb_px(ins_pcds, cid, to_px)
        if box is None:
            continue
        visible_boxes.append(box)
        if cid == gt_id and cid == pred_id:
            color = (22, 163, 74, 255)
            fill = None
            suffix = " GT/PRED"
        elif cid == gt_id:
            color = (22, 163, 74, 255)
            fill = None
            suffix = " GT"
        elif cid == pred_id:
            if pred_success:
                color = (22, 163, 74, 255)
                fill = None
                suffix = " SEL"
            else:
                color = (220, 38, 38, 255)
                fill = None
                suffix = " PRED"
        else:
            color = (37, 99, 235, 255)
            fill = None
            suffix = ""
        x0, y0, x1, y1 = box
        overlay.rounded_rectangle([x0, y0, x1, y1], radius=4, fill=fill, outline=color, width=6)
        draw_badge(overlay, (x0, y0 - 52), f"{candidate_letter(rank)} id{cid}{suffix}", color, size=25)

    if crop_local and visible_boxes:
        xs0, ys0, xs1, ys1 = zip(*visible_boxes)
        pad = 180
        crop = (
            max(0, int(min(xs0) - pad)),
            max(0, int(min(ys0) - pad)),
            min(img.width, int(max(xs1) + pad)),
            min(img.height, int(max(ys1) + pad)),
        )
        img = img.crop(crop).resize((img_size, img_size), resample=1)
        overlay = ImageDraw.Draw(img, "RGBA")

    draw_badge(overlay, (20, 20), result_badge, (17, 24, 39, 238), size=30)
    return img.convert("RGB")


def show_image(ax, img, title: str):
    ax.imshow(img)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.axis("off")


def parse_nr3d_prompt(prompt: str) -> tuple[dict | None, list[dict]]:
    anchor = None
    m = re.search(
        r"id(?P<id>-?\d+)\s+(?P<name>[^:]+): center=\((?P<center>[^)]+)\) size=\((?P<size>[^)]+)\)",
        prompt,
    )
    if m:
        anchor = {
            "letter": "Anchor",
            "id": int(m.group("id")),
            "name": m.group("name").strip(),
            "center": parse_tuple(m.group("center")),
            "size": parse_size(m.group("size")),
        }

    candidates = []
    pat = re.compile(
        r"(?P<letter>[A-Z])\.\s+id(?P<id>-?\d+)\s+(?P<name>[^:]+): "
        r"center=\((?P<center>[^)]+)\) size=\((?P<size>[^)]+)\)"
    )
    for line in prompt.splitlines():
        m = pat.search(line)
        if not m:
            continue
        dm = re.search(r"xy_dist=(?P<dist>[0-9.]+)m", line)
        candidates.append(
            {
                "letter": m.group("letter"),
                "id": int(m.group("id")),
                "name": m.group("name").strip(),
                "center": parse_tuple(m.group("center")),
                "size": parse_size(m.group("size")),
                "dist": float(dm.group("dist")) if dm else None,
            }
        )
    return anchor, candidates


def parse_scanrefer_prompt(prompt: str, instance_ids: list[int]) -> tuple[dict | None, list[dict]]:
    anchor = None
    m = re.search(r"Anchor object:\s+(?P<name>.*?)\s+center=\((?P<center>[^)]+)\)", prompt)
    if m:
        anchor = {
            "letter": "Anchor",
            "id": None,
            "name": m.group("name").strip(),
            "center": parse_tuple(m.group("center")),
            "size": (0.35, 0.35, 0.35),
        }

    candidates = []
    pat = re.compile(
        r"\[(?P<letter>[A-Z])\]\s+(?P<name>[^:]+): "
        r"anchor_delta=\((?P<delta>[^)]+)\)\s+dist=(?P<dist>[0-9.]+)m\s+"
        r"size=\((?P<size>[^)]+)\)"
    )
    for i, m in enumerate(pat.finditer(prompt)):
        delta = parse_tuple(m.group("delta"))
        center = tuple((anchor["center"][j] if anchor else 0.0) + delta[j] for j in range(3))
        candidates.append(
            {
                "letter": m.group("letter"),
                "id": instance_ids[i] if i < len(instance_ids) else i,
                "name": m.group("name").strip(),
                "center": center,
                "size": parse_size(m.group("size")),
                "dist": float(m.group("dist")),
            }
        )
    return anchor, candidates


def cuboid_faces(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    x = [cx - sx / 2, cx + sx / 2]
    y = [cy - sy / 2, cy + sy / 2]
    z = [cz - sz / 2, cz + sz / 2]
    v = np.array(
        [
            [x[0], y[0], z[0]],
            [x[1], y[0], z[0]],
            [x[1], y[1], z[0]],
            [x[0], y[1], z[0]],
            [x[0], y[0], z[1]],
            [x[1], y[0], z[1]],
            [x[1], y[1], z[1]],
            [x[0], y[1], z[1]],
        ]
    )
    return [
        [v[j] for j in [0, 1, 2, 3]],
        [v[j] for j in [4, 5, 6, 7]],
        [v[j] for j in [0, 1, 5, 4]],
        [v[j] for j in [2, 3, 7, 6]],
        [v[j] for j in [1, 2, 6, 5]],
        [v[j] for j in [0, 3, 7, 4]],
    ]


def object_color(obj_id: int | None, gt_id: int | None, pred_id: int | None, selected_id: int | None = None) -> str:
    if obj_id == gt_id and obj_id == pred_id:
        return GREEN
    if obj_id == gt_id:
        return GREEN
    if obj_id == pred_id:
        return RED
    if obj_id == selected_id:
        return GREEN
    return BLUE


def draw_3d_overlay(ax, candidates, anchor=None, gt_id=None, pred_id=None, selected_id=None, title="3D overlay"):
    for obj in candidates:
        color = object_color(obj["id"], gt_id, pred_id, selected_id)
        poly = Poly3DCollection(
            cuboid_faces(obj["center"], obj["size"]),
            alpha=0.18,
            facecolor=color,
            edgecolor=color,
            linewidths=2.0,
        )
        ax.add_collection3d(poly)
        cx, cy, cz = obj["center"]
        suffix = []
        if obj["id"] == gt_id:
            suffix.append("GT")
        if obj["id"] == pred_id:
            suffix.append("PRED")
        if obj["id"] == selected_id and obj["id"] != pred_id:
            suffix.append("SEL")
        ax.text(
            cx,
            cy,
            cz + obj["size"][2] / 2 + 0.08,
            f"{obj['letter']} id{obj['id']}" + (f" ({'/'.join(suffix)})" if suffix else ""),
            color=color,
            fontsize=10,
            weight="bold",
        )

    if anchor:
        poly = Poly3DCollection(
            cuboid_faces(anchor["center"], anchor["size"]),
            alpha=0.10,
            facecolor=GRAY,
            edgecolor=BLACK,
            linewidths=1.4,
        )
        ax.add_collection3d(poly)
        cx, cy, cz = anchor["center"]
        ax.text(cx, cy, cz + 0.35, f"Anchor: {anchor['name']}", color=BLACK, fontsize=9)

    all_objs = candidates + ([anchor] if anchor else [])
    xs = [o["center"][0] for o in all_objs]
    ys = [o["center"][1] for o in all_objs]
    zs = [o["center"][2] for o in all_objs]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.5)
    ax.set_xlim(np.mean(xs) - span * 0.68, np.mean(xs) + span * 0.68)
    ax.set_ylim(np.mean(ys) - span * 0.68, np.mean(ys) + span * 0.68)
    ax.set_zlim(max(0.0, min(zs) - 0.3), max(zs) + 0.95)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=28, azim=-58)
    ax.set_title(title, fontsize=15, weight="bold")


def draw_bev_overlay(
    ax,
    candidates,
    anchor=None,
    gt_id=None,
    pred_id=None,
    selected_id=None,
    title="BEV overlay",
    result_label="",
):
    ax.set_facecolor("#F1F5F9")
    for obj in candidates:
        cx, cy, _ = obj["center"]
        sx, sy, _ = obj["size"]
        color = object_color(obj["id"], gt_id, pred_id, selected_id)
        rect = Rectangle(
            (cx - sx / 2, cy - sy / 2),
            sx,
            sy,
            linewidth=2.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.16,
        )
        ax.add_patch(rect)
        ax.scatter([cx], [cy], s=75, c=color, edgecolors="white", linewidths=1.5, zorder=4)
        suffix = []
        if obj["id"] == gt_id:
            suffix.append("GT")
        if obj["id"] == pred_id:
            suffix.append("PRED")
        if obj["id"] == selected_id and obj["id"] != pred_id:
            suffix.append("SEL")
        label = f"{obj['letter']} id{obj['id']}" + (f"\n{'/'.join(suffix)}" if suffix else "")
        ax.text(cx, cy + sy / 2 + 0.08, label, ha="center", va="bottom", fontsize=12, weight="bold", color=color)

    if anchor:
        cx, cy, _ = anchor["center"]
        sx, sy, _ = anchor["size"]
        ax.add_patch(
            Rectangle(
                (cx - sx / 2, cy - sy / 2),
                sx,
                sy,
                linewidth=2.0,
                edgecolor=BLACK,
                facecolor=GRAY,
                alpha=0.10,
                linestyle="--",
            )
        )
        ax.scatter([cx], [cy], s=55, c=BLACK, marker="x", zorder=5)
        ax.text(cx, cy - sy / 2 - 0.10, f"anchor\n{anchor['name']}", ha="center", va="top", fontsize=10, color=BLACK)

    all_objs = candidates + ([anchor] if anchor else [])
    xs = [o["center"][0] for o in all_objs]
    ys = [o["center"][1] for o in all_objs]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.5)
    ax.set_xlim(np.mean(xs) - span * 0.75, np.mean(xs) + span * 0.75)
    ax.set_ylim(np.mean(ys) - span * 0.75, np.mean(ys) + span * 0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#CBD5E1", linewidth=0.8, alpha=0.75)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, fontsize=16, weight="bold")
    if result_label:
        ax.text(
            0.02,
            0.98,
            result_label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
        )


def text_panel(ax, title: str, body: str, title_color=BLACK, body_size=13, mono=False):
    ax.axis("off")
    ax.set_facecolor("white")
    ax.text(0.02, 0.96, title, ha="left", va="top", fontsize=17, weight="bold", color=title_color)
    ax.text(
        0.02,
        0.82,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        family="monospace" if mono else "DejaVu Sans",
        linespacing=1.28,
        color=BLACK,
    )


def prompt_excerpt(prompt: str, max_lines: int = 12, width: int = 76) -> str:
    lines = [line for line in prompt.splitlines() if line.strip()]
    keep = []
    for line in lines:
        if (
            line.startswith("Query:")
            or line.startswith("Anchor")
            or line.startswith("  id")
            or line.startswith("Candidates:")
            or re.match(r"\s*(\[?[A-Z]\]?\.?|[A-Z]\s+\()", line)
        ):
            keep.extend(textwrap.wrap(line, width=width, subsequent_indent="    "))
        if len(keep) >= max_lines:
            break
    return "\n".join(keep[:max_lines])


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_nr3d_failure_success_assets(nr3d_e0: dict[int, dict], nr3d_spatial: dict[int, dict]) -> None:
    e0 = nr3d_e0[0]
    sp = nr3d_spatial[0]
    anchor, candidates = parse_nr3d_prompt(sp["prompt_text"])
    gt_id = sp["obj_id"]
    e0_pred_id = e0["trace"]["pred_instance"]
    spatial_pred_id = sp["selected_instance_id"]
    candidate_ids = [c["id"] for c in candidates]
    anchor_ids = [anchor["id"]] if anchor else []

    # E0 failure: separate BEV overlay asset.
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.55])
    ax_text = fig.add_subplot(gs[0, 0])
    body = (
        f"Query:\n\"{textwrap.fill(sp['caption'], 44)}\"\n\n"
        "Route: E0 RGB canvas\n"
        f"Prediction: id{e0_pred_id}\n"
        f"Ground truth: id{gt_id}\n"
        f"IoU: {e0['iou']:.3f}\n"
        "Result: failure"
    )
    text_panel(ax_text, "Failure case", body, title_color=RED, body_size=15)
    ax_bev = fig.add_subplot(gs[0, 1])
    fail_bev = make_real_bev_overlay(
        e0["scene_id"],
        candidate_ids,
        anchor_ids,
        gt_id=gt_id,
        pred_id=e0_pred_id,
        result_badge=f"E0 predicted id{e0_pred_id}; GT id{gt_id}",
    )
    show_image(ax_bev, fail_bev, "E0 failure on actual mesh BEV render")
    fig.suptitle("Presentation Asset: E0 Failure Overlay", fontsize=21, weight="bold")
    save(fig, "asset_01_e0_failure_bev_overlay")

    # Spatial-only success: separate BEV overlay asset.
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.55])
    ax_text = fig.add_subplot(gs[0, 0])
    body = (
        f"Query:\n\"{textwrap.fill(sp['caption'], 44)}\"\n\n"
        "Route: spatial-only text\n"
        f"Prediction: {sp['selected_letter']} / id{spatial_pred_id}\n"
        f"Ground truth: id{gt_id}\n"
        f"IoU: {sp['iou']:.3f}\n"
        "Result: success\n\n"
        "Why this input helps:\n"
        "The prompt exposes candidate-anchor\n"
        "distances and relative offsets."
    )
    text_panel(ax_text, "Success case", body, title_color=GREEN, body_size=15)
    ax_bev = fig.add_subplot(gs[0, 1])
    success_bev = make_real_bev_overlay(
        e0["scene_id"],
        candidate_ids,
        anchor_ids,
        gt_id=gt_id,
        pred_id=spatial_pred_id,
        result_badge=f"Spatial-only selected id{spatial_pred_id}; GT id{gt_id}",
    )
    show_image(ax_bev, success_bev, "Spatial-only success on actual mesh BEV render")
    fig.suptitle("Presentation Asset: Spatial-Only Success Overlay", fontsize=21, weight="bold")
    save(fig, "asset_02_spatial_success_bev_overlay")

    # Optional 3D counterpart for slides that prefer perspective rendering.
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax_fail = fig.add_subplot(gs[0, 0], projection="3d")
    draw_3d_overlay(ax_fail, candidates, anchor=anchor, gt_id=gt_id, pred_id=e0_pred_id, title="E0 failure: red prediction")
    ax_success = fig.add_subplot(gs[0, 1], projection="3d")
    draw_3d_overlay(ax_success, candidates, anchor=anchor, gt_id=gt_id, pred_id=spatial_pred_id, title="Spatial-only success: green prediction")
    fig.suptitle(f"Same query, different evidence\n\"{sp['caption']}\"", fontsize=18, weight="bold")
    save(fig, "asset_03_failure_vs_success_3d_overlay")


def make_geometric_2x2(scan_e0: dict[int, dict], scan_3d: dict[int, dict]) -> None:
    e0 = scan_e0[116]
    pos = scan_3d[116]
    anchor, candidates = parse_scanrefer_prompt(pos["prompt_text"], pos["candidate_instance_ids"])
    gt_id = pos["gt_instance_id"]
    e0_pred_id = e0["trace"]["pred_instance"]
    pos_pred_id = pos["selected_instance_id"]
    candidate_ids = pos["candidate_instance_ids"]
    anchor_ids = []
    if anchor:
        anchor_id = find_nearest_label_instance(pos["scene_id"], anchor["center"], ["bookshelf", "book"])
        if anchor_id is not None:
            anchor_ids.append(anchor_id)
    known_ids = {c["id"] for c in candidates}
    if gt_id not in known_ids:
        # Some locked ScanRefer prompts are truncated in the source file. Recover
        # the GT candidate box from the paired E0 record so the slide overlay
        # still shows the target object.
        cx, cy, cz, sx, sy, sz = e0["target_box"]
        candidates.append(
            {
                "letter": candidate_letter(pos["candidate_instance_ids"].index(gt_id)),
                "id": gt_id,
                "name": pos["target_category"],
                "center": (cx, cy, cz),
                "size": (sx, sy, sz),
                "dist": None,
            }
        )

    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.38], hspace=0.34, wspace=0.16)

    ax_q = fig.add_subplot(gs[0, 0])
    e0_body = (
        "Input to E0 route\n\n"
        f"Query:\n\"{textwrap.fill(pos['query'], 54)}\"\n\n"
        "VLM evidence:\n"
        "- candidate RGB canvases\n"
        "- visual appearance and local context"
    )
    text_panel(ax_q, "(a) E0 input query", e0_body, title_color=RED, body_size=13)

    ax_pos_q = fig.add_subplot(gs[0, 1])
    pos_body = (
        "Input to 3D-position route\n\n"
        f"Query:\n\"{textwrap.fill(pos['query'], 54)}\"\n\n"
        "Structured evidence:\n"
        "- candidate-anchor offsets\n"
        "- distances to the bookshelf anchor\n"
        "- candidate sizes / heights"
    )
    text_panel(ax_pos_q, "(b) 3D-position input query", pos_body, title_color=GREEN, body_size=12.5, mono=False)

    ax_e0_bev = fig.add_subplot(gs[1, 0])
    e0_bev = make_real_bev_overlay(
        pos["scene_id"],
        candidate_ids,
        anchor_ids,
        gt_id=gt_id,
        pred_id=e0_pred_id,
        result_badge=f"E0 predicted id{e0_pred_id}; GT id{gt_id}",
        crop_local=True,
    )
    show_image(ax_e0_bev, e0_bev, "E0 prediction on actual mesh BEV render")

    ax_pos_bev = fig.add_subplot(gs[1, 1])
    pos_bev = make_real_bev_overlay(
        pos["scene_id"],
        candidate_ids,
        anchor_ids,
        gt_id=gt_id,
        pred_id=pos_pred_id,
        result_badge=f"3D-position selected id{pos_pred_id}; GT id{gt_id}; IoU={pos['iou']:.3f}",
        crop_local=True,
        pred_success=True,
    )
    show_image(ax_pos_bev, pos_bev, "3D-position prediction on actual mesh BEV render")

    fig.suptitle("Geometric Query: E0 vs 3D-Position Prediction", fontsize=22, weight="bold")
    save(fig, "asset_04_geometric_e0_vs_3d_position_2x2")


def make_recovery_grid(scan_e0: dict[int, dict], scan_sources: dict[str, dict[int, dict]]) -> None:
    cases = [
        ("Spatial-only", scan_sources["spatial"][1], "desk beside chair"),
        ("Spatial-only", scan_sources["spatial"][3], "bench next to bench"),
        ("3D-position", scan_sources["3dpos"][116], "bench between bookshelves"),
        ("BEV", scan_sources["bev"][100], "couch behind wall object"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for col, (route_name, routed, short_label) in enumerate(cases):
        case_id = int(routed["query_id"])
        e0 = scan_e0[case_id]
        anchor, _ = parse_scanrefer_prompt(routed["prompt_text"], routed["candidate_instance_ids"])
        anchor_ids = []
        if anchor:
            keywords = routed.get("anchor_nouns") or []
            anchor_id = find_nearest_label_instance(routed["scene_id"], anchor["center"], keywords)
            if anchor_id is not None:
                anchor_ids.append(anchor_id)

        candidate_ids = routed["candidate_instance_ids"]
        gt_id = routed["gt_instance_id"]
        e0_pred_id = e0.get("trace", {}).get("pred_instance")
        route_pred_id = routed["selected_instance_id"]
        fail_img = make_real_bev_overlay(
            routed["scene_id"],
            candidate_ids,
            anchor_ids,
            gt_id=gt_id,
            pred_id=e0_pred_id,
            result_badge=f"E0 fail: id{e0_pred_id}; GT id{gt_id}",
            crop_local=True,
        )
        success_img = make_real_bev_overlay(
            routed["scene_id"],
            candidate_ids,
            anchor_ids,
            gt_id=gt_id,
            pred_id=route_pred_id,
            result_badge=f"{route_name} success: id{route_pred_id}; GT id{gt_id}",
            crop_local=True,
            pred_success=True,
        )
        show_image(axes[0, col], fail_img, f"E0 failure\n{short_label}")
        show_image(axes[1, col], success_img, f"{route_name} recovery")

    fig.suptitle("Failure-to-Recovery Examples on Actual Mesh BEV Renders", fontsize=23, weight="bold")
    save(fig, "asset_05_mesh_bev_recovery_grid")


def write_readme() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = """# Presentation Assets

Slide-ready qualitative assets generated from real locked query/result records.
Each asset is saved as both PNG and PDF.

| Asset | File stem | Use |
|---|---|---|
| 1 | `asset_01_e0_failure_bev_overlay` | E0 failure case with prediction/GT overlaid on an actual mesh BEV render. |
| 2 | `asset_02_spatial_success_bev_overlay` | Spatial-only success case for the same query, overlaid on an actual mesh BEV render. |
| 3 | `asset_03_failure_vs_success_3d_overlay` | Same failure/success pair as a 3D box rendering. |
| 4 | `asset_04_geometric_e0_vs_3d_position_2x2` | 2x2 comparison for a geometric bench/bookshelf query, with actual mesh BEV overlays. |
| 5 | `asset_05_mesh_bev_recovery_grid` | Four slide-ready E0 failure vs routed recovery examples on actual mesh BEV renders. |

Regenerate:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /home/knuvi/anaconda3/envs/sam3/bin/python tools/create_presentation_assets.py
```
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    nr3d_e0 = by_case("inputs/nr3d/official_e0_nr3d_openrouter_qwen_250.jsonl")
    nr3d_spatial = by_case("experiments/ablation/nr3d_missing_branches/openrouter_qwen/spatial_only_all/nr3d_query_type_routed_vlm_results.jsonl")
    scan_e0 = by_case("inputs/scanrefer/full_E0_baseline_qwen72b.jsonl")
    scan_3d = by_case("inputs/scanrefer/seeground_ablation_3dpos_only/results.jsonl", query_id=True)
    scan_sources = {
        "spatial": by_case("inputs/scanrefer/seeground_ablation_spatial_only/results.jsonl", query_id=True),
        "3dpos": scan_3d,
        "bev": by_case("inputs/scanrefer/bev_raw_labeled/results.jsonl", query_id=True),
    }

    make_nr3d_failure_success_assets(nr3d_e0, nr3d_spatial)
    make_geometric_2x2(scan_e0, scan_3d)
    make_recovery_grid(scan_e0, scan_sources)
    write_readme()
    print(f"Wrote presentation assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
