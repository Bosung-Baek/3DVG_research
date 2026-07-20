#!/usr/bin/env python3
"""Create qualitative figures with actual queries, canvases, and box layouts.

These figures are intentionally different from aggregate result plots. They use
real case outputs and pre-rendered SeqVLM canvases from the local workspace.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SEQVLM_ROOT = Path("/home/knuvi/bosung/SeqVLM")
NR3D_CANVAS_ROOT = SEQVLM_ROOT / "data" / "preprocessed_nr3d"
SCANREFER_CANVAS_ROOT = SEQVLM_ROOT / "data" / "scanrefer_preprocessed"
OUT_DIR = ROOT / "experiments" / "qualitative_figures"

GREEN = "#16A34A"
RED = "#DC2626"
BLUE = "#2563EB"
ORANGE = "#F97316"
PURPLE = "#7C3AED"
GRAY = "#64748B"
BLACK = "#111827"


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


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_pil(img: Image.Image, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / f"{stem}.png")
    img.convert("RGB").save(OUT_DIR / f"{stem}.pdf")


def candidate_letter(i: int) -> str:
    return chr(ord("A") + i)


def wrap(text: str, width: int = 64) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


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
            "is_anchor": True,
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
                "is_anchor": False,
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
            "is_anchor": True,
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
                "is_anchor": False,
            }
        )
    return anchor, candidates


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


def draw_box_scene(ax, candidates, anchor=None, selected_id=None, e0_pred_id=None, gt_id=None, title=None):
    colors = [BLUE, ORANGE, PURPLE, "#0891B2", "#DB2777", "#65A30D", "#9333EA"]
    all_objs = list(candidates)
    if anchor:
        all_objs.append(anchor)
    for i, obj in enumerate(candidates):
        obj_id = obj["id"]
        if obj_id == gt_id:
            color = GREEN
        elif obj_id == e0_pred_id:
            color = RED
        elif obj_id == selected_id:
            color = GREEN
        else:
            color = colors[i % len(colors)]
        poly = Poly3DCollection(cuboid_faces(obj["center"], obj["size"]), alpha=0.16, facecolor=color, edgecolor=color, linewidths=1.8)
        ax.add_collection3d(poly)
        cx, cy, cz = obj["center"]
        label = f"{obj['letter']} / id{obj_id}"
        if obj_id == gt_id:
            label += " GT"
        if obj_id == e0_pred_id:
            label += " E0"
        if obj_id == selected_id:
            label += " sel"
        ax.text(cx, cy, cz + obj["size"][2] / 2 + 0.08, label, color=color, weight="bold", fontsize=9)
    if anchor:
        poly = Poly3DCollection(cuboid_faces(anchor["center"], anchor["size"]), alpha=0.08, facecolor=GRAY, edgecolor=BLACK, linewidths=1.2, linestyle="--")
        ax.add_collection3d(poly)
        cx, cy, cz = anchor["center"]
        ax.text(cx, cy, cz + anchor["size"][2] / 2 + 0.08, f"Anchor: {anchor['name']}", color=BLACK, fontsize=8)

    xs = [o["center"][0] for o in all_objs]
    ys = [o["center"][1] for o in all_objs]
    zs = [o["center"][2] for o in all_objs]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.5)
    ax.set_xlim(np.mean(xs) - span * 0.65, np.mean(xs) + span * 0.65)
    ax.set_ylim(np.mean(ys) - span * 0.65, np.mean(ys) + span * 0.65)
    ax.set_zlim(max(0, min(zs) - 0.3), max(zs) + 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=26, azim=-58)
    ax.set_title(title or "3D candidate boxes")


def canvas_path(scene_id: str, instance_id: int, dataset: str) -> Path:
    root = SCANREFER_CANVAS_ROOT if dataset == "scanrefer" else NR3D_CANVAS_ROOT
    return root / scene_id / str(instance_id) / "canvas.jpg"


def load_canvas(scene_id: str, instance_id: int, dataset: str, width: int = 520) -> Image.Image:
    p = canvas_path(scene_id, instance_id, dataset)
    if not p.exists():
        # NR3D and ScanRefer folders share many scenes but have different coverage.
        alt_root = NR3D_CANVAS_ROOT if dataset == "scanrefer" else SCANREFER_CANVAS_ROOT
        p = alt_root / scene_id / str(instance_id) / "canvas.jpg"
    img = Image.open(p).convert("RGB")
    ratio = width / img.width
    return img.resize((width, int(img.height * ratio)), Image.Resampling.LANCZOS)


def label_canvas(img: Image.Image, label: str, color: str, subtitle: str = "") -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font_big = get_font(34, True)
    font_small = get_font(20, False)
    pad = 12
    text = label
    bbox = draw.textbbox((0, 0), text, font=font_big)
    draw.rounded_rectangle((pad, pad, pad + bbox[2] + 22, pad + bbox[3] + 18), radius=8, fill=color)
    draw.text((pad + 11, pad + 8), text, fill="white", font=font_big)
    if subtitle:
        sb = draw.textbbox((0, 0), subtitle, font=font_small)
        y = out.height - sb[3] - 20
        draw.rounded_rectangle((pad, y - 8, pad + sb[2] + 18, y + sb[3] + 8), radius=6, fill=(17, 24, 39))
        draw.text((pad + 9, y), subtitle, fill="white", font=font_small)
    return out


def make_montage(images: list[Image.Image], cols: int = 2, bg=(248, 250, 252), gap: int = 18) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), bg)
    for i, im in enumerate(images):
        x = (i % cols) * (w + gap)
        y = (i // cols) * (h + gap)
        canvas.paste(im, (x, y))
    return canvas


def show_pil(ax, img: Image.Image, title: str = "") -> None:
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


def prompt_excerpt(prompt: str, max_lines: int = 11, width: int = 82) -> str:
    lines = [line for line in prompt.splitlines() if line.strip()]
    keep = []
    for line in lines:
        if line.startswith("Query:") or line.startswith("Anchor") or line.startswith("  id") or re.match(r"\s*(Candidates:|\[?[A-D]\]?\.?|\s+[A-D]\.)", line):
            if len(line) > width:
                keep.extend(textwrap.wrap(line, width=width, subsequent_indent="    "))
            else:
                keep.append(line)
        if len(keep) >= max_lines:
            break
    return "\n".join(keep)


def figure_1_scene_query_boxes(nr3d_spatial):
    case = nr3d_spatial[0]
    anchor, cands = parse_nr3d_prompt(case["prompt_text"])
    fig = plt.figure(figsize=(13.2, 6.2))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    draw_box_scene(ax3d, cands, anchor=anchor, selected_id=case["selected_instance_id"], gt_id=case["obj_id"], title="3D scene candidates from box coordinates")
    ax_text = fig.add_subplot(1, 2, 2)
    ax_text.axis("off")
    text = (
        "Natural-language query\n"
        f"\"{case['caption']}\"\n\n"
        "Candidate boxes\n"
        + "\n".join(
            f"{c['letter']}. id{c['id']} {c['name']}: center=({c['center'][0]:+.2f},{c['center'][1]:+.2f},{c['center'][2]:+.2f}), "
            f"xy_dist={c.get('dist'):.2f}m"
            for c in cands
        )
        + f"\n\nAnchor: id{anchor['id']} {anchor['name']}\nGround truth: id{case['obj_id']} / selected by spatial-only: id{case['selected_instance_id']}"
    )
    ax_text.text(0.02, 0.98, text, ha="left", va="top", fontsize=11, linespacing=1.35)
    fig.suptitle("Qualitative 1: 3D Scene + Query + Candidate Boxes", fontsize=15, weight="bold")
    save(fig, "qual1_3d_scene_query_candidate_boxes")


def figure_2_e0_canvas(nr3d_e0):
    case = nr3d_e0[0]
    ids = case["trace"]["prop_indices"]
    imgs = []
    for i, inst_id in enumerate(ids):
        letter = candidate_letter(i)
        color = GREEN if inst_id == case["obj_id"] else RED if inst_id == case["trace"]["pred_instance"] else BLUE
        subtitle = []
        if inst_id == case["obj_id"]:
            subtitle.append("GT")
        if inst_id == case["trace"]["pred_instance"]:
            subtitle.append("E0 selected")
        imgs.append(label_canvas(load_canvas(case["scene_id"], inst_id, "nr3d", 300), f"{letter}: id{inst_id}", color, " / ".join(subtitle)))
    montage = make_montage(imgs, cols=4, gap=14)

    font_title = get_font(40, True)
    font_body = get_font(24, False)
    pad = 36
    header_h = 170
    footer_h = 90
    canvas = Image.new("RGB", (montage.width + pad * 2, montage.height + header_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    title = "E0 RGB canvas example with candidate letter/id overlay"
    draw.text((pad, 28), title, fill=BLACK, font=font_title)
    draw.text((pad, 88), f"Query: \"{case['caption']}\"", fill=BLACK, font=font_body)
    canvas.paste(montage, (pad, header_h))
    footer = f"E0 selected id{case['trace']['pred_instance']} (wrong, IoU={case['iou']:.3f}); green marks ground-truth candidate."
    draw.text((pad, header_h + montage.height + 28), footer, fill=BLACK, font=font_body)
    save_pil(canvas, "qual2_e0_rgb_canvas_candidate_overlay")


def figure_3_failure_vs_spatial(nr3d_e0, nr3d_spatial):
    e0 = nr3d_e0[0]
    sp = nr3d_spatial[0]
    anchor, cands = parse_nr3d_prompt(sp["prompt_text"])
    pred_id = e0["trace"]["pred_instance"]
    gt_id = sp["obj_id"]
    e0_img = label_canvas(load_canvas(e0["scene_id"], pred_id, "nr3d", 620), f"E0 selected id{pred_id}", RED, f"Wrong / IoU={e0['iou']:.3f}")
    gt_img = label_canvas(load_canvas(e0["scene_id"], gt_id, "nr3d", 620), f"Spatial selected id{gt_id}", GREEN, f"Correct / IoU={sp['iou']:.3f}")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1.1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    show_pil(ax1, e0_img, "E0 RGB canvas prediction")
    ax2 = fig.add_subplot(gs[1, 0])
    show_pil(ax2, gt_img, "Ground-truth / spatial-only prediction")
    ax3d = fig.add_subplot(gs[:, 1], projection="3d")
    draw_box_scene(ax3d, cands, anchor=anchor, selected_id=sp["selected_instance_id"], e0_pred_id=pred_id, gt_id=gt_id, title="Spatial-only evidence")
    ax_text = fig.add_subplot(gs[:, 2])
    ax_text.axis("off")
    text = (
        f"Query\n\"{sp['caption']}\"\n\n"
        "E0 route\n"
        f"- input: RGB canvases\n- selected: id{pred_id}\n- IoU: {e0['iou']:.3f} / Acc@0.25=False\n\n"
        "Spatial-only route\n"
        "- input: category + 3D position + anchor distance\n"
        f"- selected: {sp['selected_letter']} / id{sp['selected_instance_id']}\n"
        f"- IoU: {sp['iou']:.3f} / Acc@0.25=True\n\n"
        "Prompt excerpt\n"
        + prompt_excerpt(sp["prompt_text"], 9, width=74)
    )
    ax_text.text(0.02, 0.98, text, va="top", ha="left", fontsize=10, family="monospace", linespacing=1.25)
    fig.suptitle("Qualitative 3: E0 Failure vs Spatial-Only Success", fontsize=15, weight="bold")
    save(fig, "qual3_e0_failure_vs_spatial_success")


def figure_4_geometric_e0_vs_3dpos(scan_e0, scan_3d):
    e0 = scan_e0[197]
    pos = scan_3d[197]
    anchor, cands = parse_scanrefer_prompt(pos["prompt_text"], pos["candidate_instance_ids"])
    e0_pred_id = e0["trace"]["pred_instance"]
    gt_id = pos["gt_instance_id"]
    imgs = []
    for i, inst_id in enumerate(pos["candidate_instance_ids"]):
        letter = candidate_letter(i)
        color = GREEN if inst_id == gt_id else RED if inst_id == e0_pred_id else BLUE
        subtitle = []
        if inst_id == e0_pred_id:
            subtitle.append("E0 selected")
        if inst_id == gt_id:
            subtitle.append("GT / 3D-pos selected")
        imgs.append(label_canvas(load_canvas(pos["scene_id"], inst_id, "scanrefer", 360), f"{letter}: id{inst_id}", color, " / ".join(subtitle)))
    montage = make_montage(imgs, cols=3, gap=12)

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.7, 1.2, 1.05], height_ratios=[1, 1])
    ax_img = fig.add_subplot(gs[:, 0])
    show_pil(ax_img, montage, "E0 RGB candidate canvases")
    ax3d = fig.add_subplot(gs[:, 1], projection="3d")
    draw_box_scene(
        ax3d,
        cands,
        anchor=anchor,
        selected_id=pos["selected_instance_id"],
        e0_pred_id=e0_pred_id,
        gt_id=gt_id,
        title="3D-position prompt geometry",
    )
    ax_text = fig.add_subplot(gs[:, 2])
    ax_text.axis("off")
    text = (
        f"Geometric query\n\"{wrap(pos['query'], 58)}\"\n\n"
        "E0 RGB route\n"
        f"- selected: id{e0_pred_id}\n- IoU: {e0['iou']:.3f}\n- Acc@0.25=False\n\n"
        "3D-position route\n"
        f"- selected: {candidate_letter(pos['selected_candidate_id'])} / id{pos['selected_instance_id']}\n"
        f"- IoU: {pos['iou']:.3f}\n- Acc@0.25=True\n\n"
        "3D-position evidence\n"
        + prompt_excerpt(pos["prompt_text"], 11, width=72)
    )
    ax_text.text(0.02, 0.98, text, va="top", ha="left", fontsize=9.5, family="monospace", linespacing=1.22)
    fig.suptitle("Qualitative 4: Geometric Query, E0 vs 3D-Position Prediction", fontsize=15, weight="bold")
    save(fig, "qual4_geometric_e0_vs_3d_position")


def write_readme():
    text = """# Qualitative Figures

These figures use real query/result records and pre-rendered SeqVLM canvas
images from the local workspace.

| Figure | File stem | Description |
|---|---|---|
| 1 | `qual1_3d_scene_query_candidate_boxes` | 3D scene coordinate view + natural-language query + candidate boxes. |
| 2 | `qual2_e0_rgb_canvas_candidate_overlay` | E0 RGB canvas candidates with candidate letter/id overlay. |
| 3 | `qual3_e0_failure_vs_spatial_success` | E0 failure vs spatial-only success on the same NR3D query. |
| 4 | `qual4_geometric_e0_vs_3d_position` | Geometric query comparison: E0 prediction vs 3D-position prediction. |

Regenerate:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /home/knuvi/anaconda3/envs/sam3/bin/python tools/create_qualitative_figures.py
```
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main():
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

    figure_1_scene_query_boxes(nr3d_spatial)
    figure_2_e0_canvas(nr3d_e0)
    figure_3_failure_vs_spatial(nr3d_e0, nr3d_spatial)
    figure_4_geometric_e0_vs_3dpos(scan_e0, scan_3d)
    write_readme()
    print(f"Wrote qualitative figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
