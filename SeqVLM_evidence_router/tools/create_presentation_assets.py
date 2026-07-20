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
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "presentation_assets"

GREEN = "#16A34A"
RED = "#DC2626"
BLUE = "#2563EB"
ORANGE = "#F97316"
PURPLE = "#7C3AED"
GRAY = "#64748B"
BLACK = "#111827"
BG = "#F8FAFC"


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
    draw_bev_overlay(
        ax_bev,
        candidates,
        anchor=anchor,
        gt_id=gt_id,
        pred_id=e0_pred_id,
        title="E0 failure shown on BEV overlay",
        result_label=f"E0 predicted id{e0_pred_id}; GT id{gt_id}",
    )
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
    draw_bev_overlay(
        ax_bev,
        candidates,
        anchor=anchor,
        gt_id=gt_id,
        pred_id=spatial_pred_id,
        title="Spatial-only success shown on BEV overlay",
        result_label=f"Spatial-only selected id{spatial_pred_id}; GT id{gt_id}",
    )
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
    e0 = scan_e0[197]
    pos = scan_3d[197]
    anchor, candidates = parse_scanrefer_prompt(pos["prompt_text"], pos["candidate_instance_ids"])
    gt_id = pos["gt_instance_id"]
    e0_pred_id = e0["trace"]["pred_instance"]
    pos_pred_id = pos["selected_instance_id"]
    known_ids = {c["id"] for c in candidates}
    if gt_id not in known_ids:
        # The locked ScanRefer 3D-position output stores a truncated prompt for
        # this case. Recover the GT candidate box from the paired E0 record so
        # the slide overlay still shows the successful 3D-position prediction.
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
        "- distances to the door anchor\n"
        "- candidate sizes / heights"
    )
    text_panel(ax_pos_q, "(b) 3D-position input query", pos_body, title_color=GREEN, body_size=12.5, mono=False)

    ax_e0_bev = fig.add_subplot(gs[1, 0])
    draw_bev_overlay(
        ax_e0_bev,
        candidates,
        anchor=anchor,
        gt_id=gt_id,
        pred_id=e0_pred_id,
        title="E0 prediction on shared BEV overlay",
        result_label=f"E0 predicted id{e0_pred_id}; GT id{gt_id}",
    )

    ax_pos_bev = fig.add_subplot(gs[1, 1])
    draw_bev_overlay(
        ax_pos_bev,
        candidates,
        anchor=anchor,
        gt_id=gt_id,
        pred_id=pos_pred_id,
        title="3D-position prediction on shared BEV overlay",
        result_label=f"3D-position selected id{pos_pred_id}; GT id{gt_id}",
    )

    fig.suptitle("Geometric Query: E0 vs 3D-Position Prediction", fontsize=22, weight="bold")
    save(fig, "asset_04_geometric_e0_vs_3d_position_2x2")


def write_readme() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = """# Presentation Assets

Slide-ready qualitative assets generated from real locked query/result records.
Each asset is saved as both PNG and PDF.

| Asset | File stem | Use |
|---|---|---|
| 1 | `asset_01_e0_failure_bev_overlay` | E0 failure case with prediction/GT overlaid on BEV. |
| 2 | `asset_02_spatial_success_bev_overlay` | Spatial-only success case for the same query, overlaid on BEV. |
| 3 | `asset_03_failure_vs_success_3d_overlay` | Same failure/success pair as a 3D box rendering. |
| 4 | `asset_04_geometric_e0_vs_3d_position_2x2` | 2x2 comparison: E0 query/input vs 3D-position query/input, with BEV overlays. |

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

    make_nr3d_failure_success_assets(nr3d_e0, nr3d_spatial)
    make_geometric_2x2(scan_e0, scan_3d)
    write_readme()
    print(f"Wrote presentation assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
