#!/usr/bin/env python3
"""Collect high-cost/additional experiment summaries into stable JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    recomposition = load_json(ROOT / "experiments/summary.json")
    main_rows = recomposition["main_table"]
    e0 = next(row for row in main_rows if row["dataset"] == "NR3D" and row["method"] == "E0 baseline")
    final_recomposition = next(
        row for row in main_rows if row["dataset"] == "NR3D" and row["method"] == "evidence router"
    )

    repeat_paths = [
        ROOT / "experiments/end_to_end_nr3d_final_router_openrouter_qwen/summary.json",
        ROOT / "experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat2/summary.json",
    ]
    repeat_consistency_paths = [
        ROOT / "experiments/end_to_end_nr3d_final_router_openrouter_qwen/consistency_summary.json",
        ROOT / "experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat2/consistency_summary.json",
    ]
    repeats = []
    for idx, (summary_path, consistency_path) in enumerate(zip(repeat_paths, repeat_consistency_paths), start=1):
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        consistency = load_json(consistency_path) if consistency_path.exists() else {}
        repeats.append(
            {
                "run": idx,
                "path": str(summary_path.relative_to(ROOT)),
                "acc_iou25": summary["acc_iou25"],
                "acc_iou50": summary["acc_iou50"],
                "mean_iou": summary["mean_iou"],
                "num_acc25_changed_vs_recomposition": consistency.get("num_acc25_changed_vs_final"),
                "route_summary": consistency.get("route_summary", {}),
            }
        )

    acc25_values = [r["acc_iou25"] for r in repeats]
    acc50_values = [r["acc_iou50"] for r in repeats]
    miou_values = [r["mean_iou"] for r in repeats]
    repeat_summary = {
        "dataset": "nr3d_250",
        "model": "openrouter-qwen",
        "baseline_e0": e0,
        "final_recomposition": final_recomposition,
        "num_repeats": len(repeats),
        "repeats": repeats,
        "repeat_mean": {
            "acc_iou25": round(mean(acc25_values), 4) if acc25_values else None,
            "acc_iou50": round(mean(acc50_values), 4) if acc50_values else None,
            "mean_iou": round(mean(miou_values), 4) if miou_values else None,
        },
        "repeat_population_std": {
            "acc_iou25": round(pstdev(acc25_values), 4) if len(acc25_values) > 1 else 0.0,
            "acc_iou50": round(pstdev(acc50_values), 4) if len(acc50_values) > 1 else 0.0,
            "mean_iou": round(pstdev(miou_values), 4) if len(miou_values) > 1 else 0.0,
        },
        "interpretation": (
            "E0 route is reused and deterministic in this setup; observed variance comes from "
            "new VLM calls on non-E0 spatial/BEV branches."
        ),
    }
    dump_json(ROOT / "experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat_summary.json", repeat_summary)

    model_change_path = ROOT / "experiments/ablation/vlm_model_change/openrouter_qwen3_vl_8b_final_router/summary.json"
    invalid_model_path = ROOT / "experiments/ablation/vlm_model_change/openrouter_qwen_vl_7b_final_router/summary.json"
    model_change = load_json(model_change_path) if model_change_path.exists() else {}
    model_change_summary = {
        "dataset": "nr3d_250",
        "experiment": "branch_vlm_model_change",
        "baseline_e0_openrouter_qwen": e0,
        "final_router_openrouter_qwen_recomposition": final_recomposition,
        "alternate_branch_model": model_change.get("route_policy", {}).get("model_override"),
        "alternate_alias": model_change.get("model"),
        "alternate_result": {
            "acc_iou25": model_change.get("acc_iou25"),
            "acc_iou50": model_change.get("acc_iou50"),
            "mean_iou": model_change.get("mean_iou"),
        },
        "route_counts": model_change.get("route_counts"),
        "caveat": (
            "This run changes only the non-E0 routed branch model. E0 routes still reuse the "
            "official openrouter-qwen E0 source output, so this is branch-model sensitivity, "
            "not a full end-to-end alternate-VLM baseline."
        ),
        "source": str(model_change_path.relative_to(ROOT)) if model_change_path.exists() else None,
        "invalid_model_diagnostic": (
            str(invalid_model_path.relative_to(ROOT)) if invalid_model_path.exists() else None
        ),
    }
    dump_json(ROOT / "experiments/ablation/vlm_model_change/summary.json", model_change_summary)

    additional = {
        "nr3d_full_branch_fill_in": str(
            (ROOT / "experiments/ablation/nr3d_missing_branches/openrouter_qwen/summary.json").relative_to(ROOT)
        ),
        "nr3d_end_to_end_repeat_summary": "experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat_summary.json",
        "gt_proposal_oracle": "experiments/gt_proposal/summary.json",
        "vlm_model_change": "experiments/ablation/vlm_model_change/summary.json",
    }
    dump_json(ROOT / "experiments/additional_experiments_summary.json", additional)


if __name__ == "__main__":
    main()
