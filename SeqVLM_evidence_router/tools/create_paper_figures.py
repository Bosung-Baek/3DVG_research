#!/usr/bin/env python3
"""Create paper-facing figures from locked experiment artifacts.

The script intentionally reads only repository-local JSON/JSONL summaries. It
does not call VLM APIs and does not depend on private data assets.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "figures"

ROUTE_LABELS = {
    "E0": "E0 RGB\ncanvas",
    "seeground_ablation_spatial_only": "Spatial-only\ntext",
    "bev_raw_labeled": "BEV labeled\nlayout",
    "seeground_ablation_3dpos_only": "3D position\ntext",
}

FORMAT_LABELS = {
    "E0_baseline": "E0 RGB\ncanvas",
    "BEV_labeled_layout": "BEV labeled\nlayout",
    "spatial_only_text": "Spatial-only\ntext",
    "3d_position_text": "3D position\ntext",
}

COLORS = {
    "e0": "#4C78A8",
    "router": "#59A14F",
    "spatial": "#F28E2B",
    "bev": "#E15759",
    "pos3d": "#B07AA1",
    "gray": "#6B7280",
    "light": "#F8FAFC",
    "line": "#334155",
}


def read_json(rel_path: str):
    with (ROOT / rel_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(rel_path: str):
    rows = []
    with (ROOT / rel_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "savefig.facecolor": "white",
        }
    )


def fig_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#F8FAFC", ec="#334155"):
        patch = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x + 0.18, y + h - 0.32, title, va="top", ha="left", weight="bold", fontsize=11)
        ax.text(x + 0.18, y + h - 0.78, body, va="top", ha="left", fontsize=9, linespacing=1.2)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=COLORS["line"]),
        )

    box(0.4, 2.2, 2.0, 1.5, "Query", "Natural-language\nreferring expression", "#EFF6FF")
    box(3.0, 2.05, 2.35, 1.8, "Evidence router", "Deterministic rules\nwith E0 fallback", "#ECFDF5")
    route_boxes = [
        (6.0, 4.45, "E0 RGB canvas", "visual / mixed", "#EEF2FF"),
        (6.0, 3.30, "Spatial-only text", "proximity", "#FFF7ED"),
        (6.0, 2.15, "BEV layout", "pure ordinal", "#FEF2F2"),
        (6.0, 1.00, "3D position text", "pure geometric", "#F5F3FF"),
    ]
    for x, y, title, tag, fc in route_boxes:
        box(x, y, 2.25, 0.85, title, "", fc)
        ax.text(
            x + 0.06,
            y + 0.98,
            tag,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#475569",
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="#CBD5E1"),
        )
    box(9.1, 2.05, 2.25, 1.8, "VLM selection", "Select target from\nsame candidate pool", "#F8FAFC")

    arrow(2.4, 2.95, 3.0, 2.95)
    for y in [4.88, 3.73, 2.58, 1.43]:
        arrow(5.35, 2.95, 6.0, y)
        arrow(8.25, y, 9.1, 2.95)

    rules = [
        "1. proximity_derived -> spatial-only text",
        "2. visual attribute -> E0 fallback",
        "3. pure ordinal -> BEV",
        "4. pure geometric -> 3D position text",
        "5. viewpoint/mixed/ambiguous -> E0 fallback",
    ]
    ax.text(
        0.55,
        0.55,
        "Routing policy is dataset-agnostic: the dataset name is not used as a decision variable.\n"
        + "\n".join(rules),
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", fc="#FFFFFF", ec="#CBD5E1"),
    )
    ax.set_title("Evidence-Aware Routing Pipeline")
    save(fig, "fig1_pipeline_overview")


def fig_main_results() -> None:
    table = read_json("experiments/main_table/table.json")
    datasets = ["ScanRefer", "NR3D"]
    metrics = ["acc_iou25", "acc_iou50", "mean_iou"]
    metric_labels = ["Acc@0.25", "Acc@0.50", "mIoU"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=False)
    x = np.arange(len(datasets))
    width = 0.34

    for ax, metric, label in zip(axes, metrics, metric_labels):
        e0 = [next(r for r in table if r["dataset"] == d and r["method"] == "E0 baseline")[metric] for d in datasets]
        rt = [next(r for r in table if r["dataset"] == d and r["method"] == "evidence router")[metric] for d in datasets]
        ax.bar(x - width / 2, e0, width, label="E0 baseline", color=COLORS["e0"])
        ax.bar(x + width / 2, rt, width, label="Evidence router", color=COLORS["router"])
        for i, (a, b) in enumerate(zip(e0, rt)):
            ax.text(i + width / 2, b + 0.008, f"+{b-a:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, datasets)
        ax.set_ylim(0.35, 0.72)
        ax.set_title(label)
        ax.set_ylabel(label)
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Main Results on 250-Query Zero-Shot Evaluation")
    save(fig, "fig2_main_results")


def fig_input_format_ablation() -> None:
    data = read_json("experiments/ablation/input_format_overall_scanrefer.json")
    names = list(data.keys())
    vals = [data[n]["acc_iou25"] for n in names]
    labels = [FORMAT_LABELS[n] for n in names]
    colors = [COLORS["e0"], COLORS["bev"], COLORS["spatial"], COLORS["pos3d"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0.30, 0.55)
    ax.set_ylabel("Acc@0.25")
    ax.set_title("Input Format Ablation on ScanRefer: One Format for All Queries")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom")
    ax.text(
        0.02,
        0.96,
        "No non-E0 input is best when blindly applied to every query.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CBD5E1"),
    )
    save(fig, "fig3_input_format_ablation_scanrefer")


def fig_query_type_heatmap() -> None:
    data = read_json("experiments/ablation/input_format_by_query_type_scanrefer.json")
    query_types = sorted(data.keys(), key=lambda k: data[k]["E0_baseline"]["num_queries"], reverse=True)
    formats = ["E0_baseline", "BEV_labeled_layout", "spatial_only_text", "3d_position_text"]
    matrix = np.array([[data[q][f]["acc_iou25"] for f in formats] for q in query_types])
    counts = [data[q]["E0_baseline"]["num_queries"] for q in query_types]

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=0.8)
    ax.set_xticks(np.arange(len(formats)), [FORMAT_LABELS[f].replace("\n", " ") for f in formats], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(query_types)), [f"{q} (N={n})" for q, n in zip(query_types, counts)])
    ax.set_title("ScanRefer Query-Type / Input-Format Interaction (Acc@0.25)")
    for i in range(matrix.shape[0]):
        best = int(np.argmax(matrix[i]))
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 0.5 else "#111827"
            weight = "bold" if j == best else "normal"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color=color, weight=weight, fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Acc@0.25")
    save(fig, "fig4_query_type_input_heatmap_scanrefer")


def fig_route_distribution_and_contribution() -> None:
    table = read_json("experiments/main_table/table.json")
    contrib = read_json("experiments/ablation/route_contribution/summary.json")
    datasets = ["scanrefer", "nr3d"]
    pretty = {"scanrefer": "ScanRefer", "nr3d": "NR3D"}
    routes = ["E0", "seeground_ablation_spatial_only", "bev_raw_labeled", "seeground_ablation_3dpos_only"]
    colors = [COLORS["e0"], COLORS["spatial"], COLORS["bev"], COLORS["pos3d"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    left = axes[0]
    bottom = np.zeros(len(datasets))
    for route, color in zip(routes, colors):
        vals = []
        for d in datasets:
            row = next(r for r in table if r["dataset"].lower() == d and r["method"] == "evidence router")
            vals.append(row["route_counts"].get(route, 0))
        left.bar([pretty[d] for d in datasets], vals, bottom=bottom, color=color, label=ROUTE_LABELS[route].replace("\n", " "))
        bottom += np.array(vals)
    left.set_ylabel("Number of routed queries")
    left.set_title("Final Route Distribution")
    left.legend(frameon=False, loc="upper right")

    right = axes[1]
    x_labels = []
    rec = []
    reg = []
    for d in datasets:
        for route in ["seeground_ablation_spatial_only", "bev_raw_labeled", "seeground_ablation_3dpos_only"]:
            if route in contrib[d]:
                x_labels.append(f"{pretty[d]}\n{ROUTE_LABELS[route]}")
                rec.append(contrib[d][route]["recovery"])
                reg.append(-contrib[d][route]["regression"])
    x = np.arange(len(x_labels))
    right.bar(x, rec, color=COLORS["router"], label="Recovery")
    right.bar(x, reg, color=COLORS["bev"], label="Regression")
    right.axhline(0, color="#111827", lw=1)
    right.set_xticks(x, x_labels)
    right.set_ylabel("Count vs E0")
    right.set_title("Route-Level Contribution")
    right.legend(frameon=False)
    save(fig, "fig5_route_distribution_contribution")


def fig_policy_and_component_ablation() -> None:
    cumulative = read_json("experiments/ablation/policy_ablation/summary.json")["cumulative"]
    components = read_json("experiments/ablation/router_components/summary.json")
    variants = [
        ("e0_only", "E0 only"),
        ("proximity_only", "Proximity\nonly"),
        ("proximity_plus_ordinal", "Prox. +\nordinal"),
        ("proximity_plus_ordinal_geometric", "Prox. + ord.\n+ geom."),
        ("full_router", "Full\nrouter"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.8))

    ax = axes[0]
    x = np.arange(len(variants))
    for dataset, color, marker in [("scanrefer", COLORS["e0"], "o"), ("nr3d", COLORS["router"], "s")]:
        vals = [cumulative[k][dataset]["acc_iou25"] for k, _ in variants]
        ax.plot(x, vals, marker=marker, color=color, lw=2, label=dataset.upper() if dataset == "nr3d" else "ScanRefer")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [label for _, label in variants])
    ax.set_ylabel("Acc@0.25")
    ax.set_ylim(0.48, 0.68)
    ax.set_title("Cumulative Policy Ablation")
    ax.legend(frameon=False)

    ax = axes[1]
    comp_order = ["full_router", "without_visual_fallback", "without_purity_constraint", "without_viewpoint_fallback", "without_priority_ordering"]
    comp_labels = ["Full", "No visual\nfallback", "No purity\nconstraint", "No viewpoint\nfallback", "No priority\nordering"]
    scan = [components[k]["scanrefer"]["acc_iou25"] for k in comp_order]
    nr = [components[k]["nr3d"]["acc_iou25"] for k in comp_order]
    x = np.arange(len(comp_order))
    width = 0.36
    ax.bar(x - width / 2, scan, width, color=COLORS["e0"], label="ScanRefer")
    ax.bar(x + width / 2, nr, width, color=COLORS["router"], label="NR3D")
    ax.set_xticks(x, comp_labels)
    ax.set_ylim(0.48, 0.68)
    ax.set_ylabel("Acc@0.25")
    ax.set_title("Router Component Ablation")
    ax.legend(frameon=False)
    save(fig, "fig6_policy_component_ablation")


def fig_llm_router_comparison() -> None:
    main = read_json("experiments/main_table/table.json")
    llm = read_json("experiments/ablation/llm_router_priority_openrouter_qwen/summary.json")
    datasets = ["ScanRefer", "NR3D"]
    methods = ["E0 baseline", "Dictionary router", "Priority LLM router"]
    vals = {}
    for d in datasets:
        vals[(d, "E0 baseline")] = next(r for r in main if r["dataset"] == d and r["method"] == "E0 baseline")["acc_iou25"]
        vals[(d, "Dictionary router")] = next(r for r in main if r["dataset"] == d and r["method"] == "evidence router")["acc_iou25"]
        vals[(d, "Priority LLM router")] = llm[d.lower()]["acc_iou25"] if d == "ScanRefer" else llm["nr3d"]["acc_iou25"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    x = np.arange(len(methods))
    for ax, d in zip(axes, datasets):
        y = [vals[(d, m)] for m in methods]
        ax.bar(x, y, color=[COLORS["e0"], COLORS["router"], COLORS["pos3d"]])
        ax.set_xticks(x, ["E0", "Dictionary\nrouter", "Priority LLM\nrouter"])
        ax.set_ylim(0.45 if d == "ScanRefer" else 0.58, 0.69)
        ax.set_ylabel("Acc@0.25")
        ax.set_title(d)
        for i, v in enumerate(y):
            ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom")
    fig.suptitle("Dictionary Router vs Priority-Prompted LLM Router")
    save(fig, "fig7_llm_router_comparison")


def fig_oracle_and_rerun() -> None:
    oracle = read_json("experiments/ablation/representation_oracle/summary.json")
    rerun = read_json("experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

    ax = axes[0]
    labels = ["ScanRefer", "NR3D\navailable-source"]
    e0 = [oracle["scanrefer"]["e0_acc25"], oracle["nr3d_available_source"]["e0_acc25"]]
    router = [oracle["scanrefer"]["router_acc25"], oracle["nr3d_available_source"]["router_acc25"]]
    orc = [oracle["scanrefer"]["oracle_acc25"], oracle["nr3d_available_source"]["oracle_acc25"]]
    x = np.arange(2)
    width = 0.25
    ax.bar(x - width, e0, width, color=COLORS["e0"], label="E0")
    ax.bar(x, router, width, color=COLORS["router"], label="Router")
    ax.bar(x + width, orc, width, color="#EDC948", label="Representation oracle")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.45, 0.78)
    ax.set_ylabel("Acc@0.25")
    ax.set_title("Representation Oracle Gap")
    ax.legend(frameon=False)

    ax = axes[1]
    names = ["E0", "Recomp.", "Rerun 1", "Rerun 2", "Rerun\nmean"]
    vals = [
        rerun["baseline_e0"]["acc_iou25"],
        rerun["final_recomposition"]["acc_iou25"],
        rerun["repeats"][0]["acc_iou25"],
        rerun["repeats"][1]["acc_iou25"],
        rerun["repeat_mean"]["acc_iou25"],
    ]
    colors = [COLORS["e0"], COLORS["router"], COLORS["spatial"], COLORS["spatial"], COLORS["gray"]]
    ax.bar(names, vals, color=colors)
    ax.errorbar(4, vals[4], yerr=rerun["repeat_population_std"]["acc_iou25"], color="#111827", capsize=5)
    ax.set_ylim(0.58, 0.68)
    ax.set_ylabel("Acc@0.25")
    ax.set_title("NR3D End-to-End Rerun Variance")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    save(fig, "fig8_oracle_and_rerun")


def fig_failure_case_summary() -> None:
    cases = read_jsonl("experiments/failure_visualization/cases.jsonl")
    buckets = {}
    for row in cases:
        buckets.setdefault(row["bucket"], []).append(row)

    selected = [
        ("recovery", "Spatial recovery", next(r for r in cases if r["bucket"] == "e0_fail_spatial_success")),
        ("regression", "Spatial regression", next(r for r in cases if r["bucket"] == "spatial_regression")),
        ("bev", "BEV recovery", next(r for r in cases if r["bucket"] == "bev_success" and r["transition"] == "recovery")),
        ("fallback", "Visual fallback kept correct", next(r for r in cases if r["bucket"] == "visual_fallback_kept_correct")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 7.6))
    for ax, (_, title, row) in zip(axes.ravel(), selected):
        ax.axis("off")
        color = {
            "recovery": "#ECFDF5",
            "regression": "#FEF2F2",
            "bev": "#EFF6FF",
            "fallback": "#F8FAFC",
        }[_]
        rect = plt.Rectangle((0.02, 0.05), 0.96, 0.90, facecolor=color, edgecolor="#CBD5E1", linewidth=1.2)
        ax.add_patch(rect)
        caption = "\n".join(textwrap.wrap(row["caption"], width=58))
        text = (
            f"{title}\n"
            f"Dataset: {row['dataset']}    Case: {row['case']}    Type: {row['query_type']}\n"
            f"Route: {row['route_name']} ({row['route_reason']})\n\n"
            f"Query: {caption}\n\n"
            f"E0 IoU: {row['e0_iou']:.3f}    Routed IoU: {row['routed_iou']:.3f}\n"
            f"Transition: {row['transition']}"
        )
        ax.text(0.07, 0.90, text, transform=ax.transAxes, ha="left", va="top", fontsize=10, linespacing=1.25)
    fig.suptitle("Representative Failure / Recovery Cases")
    save(fig, "fig9_failure_case_summary")


def write_readme() -> None:
    text = """# Paper Figures

This directory contains paper-facing visualizations generated from locked
experiment artifacts. The figures are derived from repository-local JSON/JSONL
files and do not require VLM API calls or private raw scene assets.

## Generated Figures

| File stem | Purpose |
|---|---|
| `fig1_pipeline_overview` | Overview of evidence-aware routing and input-format selection. |
| `fig2_main_results` | Main ScanRefer/NR3D comparison against E0 baseline. |
| `fig3_input_format_ablation_scanrefer` | Shows that blindly applying one non-E0 input to every ScanRefer query is worse than E0. |
| `fig4_query_type_input_heatmap_scanrefer` | Query-type / input-format interaction heatmap. |
| `fig5_route_distribution_contribution` | Final route distribution and recovery/regression contribution. |
| `fig6_policy_component_ablation` | Cumulative policy and router-component ablations. |
| `fig7_llm_router_comparison` | Dictionary router vs priority-prompted LLM router. |
| `fig8_oracle_and_rerun` | Representation oracle gap and NR3D end-to-end rerun variance. |
| `fig9_failure_case_summary` | Lightweight representative recovery/regression/fallback cases. |

Every figure is saved as both `.png` and `.pdf`.

Regenerate all figures:

```bash
python tools/create_paper_figures.py
```
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_pipeline_overview()
    fig_main_results()
    fig_input_format_ablation()
    fig_query_type_heatmap()
    fig_route_distribution_and_contribution()
    fig_policy_and_component_ablation()
    fig_llm_router_comparison()
    fig_oracle_and_rerun()
    fig_failure_case_summary()
    write_readme()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
