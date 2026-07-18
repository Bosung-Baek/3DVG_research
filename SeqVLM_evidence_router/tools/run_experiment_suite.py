#!/usr/bin/env python3
"""Generate paper experiment tables from available source outputs.

This script is intentionally result-file based. It supports the experiments
needed for the paper without requiring the full raw ScanNet/Mask3D pipeline.
Completed source outputs are placed under inputs/ or optional experiment paths;
the script aggregates them into main-table and ablation summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = REPO / "inputs"
DEFAULT_OUT = REPO / "experiments"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def row_id(row: dict) -> int:
    for key in ("case", "case_id", "query_id"):
        if key in row:
            return int(str(row[key]).lstrip("0") or "0")
    raise KeyError(f"No case id in row: {sorted(row)}")


def acc25(row: dict) -> bool:
    if "acc25" in row:
        return bool(row["acc25"])
    if "success_iou25" in row:
        return bool(row["success_iou25"])
    return float(row.get("iou", 0) or 0) >= 0.25


def acc50(row: dict) -> bool:
    if "acc50" in row:
        return bool(row["acc50"])
    if "success_iou50" in row:
        return bool(row["success_iou50"])
    return float(row.get("iou", 0) or 0) >= 0.50


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"num_queries": 0, "acc_iou25": None, "acc_iou50": None, "mean_iou": None}
    return {
        "num_queries": n,
        "acc_iou25": round(sum(acc25(r) for r in rows) / n, 4),
        "acc_iou50": round(sum(acc50(r) for r in rows) / n, 4),
        "mean_iou": round(sum(float(r.get("iou", 0) or 0) for r in rows) / n, 4),
    }


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def run_final_router(input_dir: Path, out_dir: Path) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/evaluate_universal_evidence_router.py"),
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        cwd=REPO,
    )
    return json.loads((out_dir / "summary.json").read_text())


def scanrefer_source_tables(input_dir: Path) -> tuple[dict, dict]:
    scan = input_dir / "scanrefer"
    sources = {
        "E0_baseline": load_jsonl(scan / "full_E0_baseline_qwen72b.jsonl"),
        "BEV_labeled_layout": load_jsonl(scan / "bev_raw_labeled/results.jsonl"),
        "spatial_only_text": load_jsonl(scan / "seeground_ablation_spatial_only/results.jsonl"),
        "3d_position_text": load_jsonl(scan / "seeground_ablation_3dpos_only/results.jsonl"),
    }
    overall = {name: metrics(rows) for name, rows in sources.items()}

    labels = {row_id(r): r.get("query_type", "uncategorized") for r in sources["BEV_labeled_layout"]}
    by_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for source_name, rows in sources.items():
        for row in rows:
            by_type[labels.get(row_id(row), "uncategorized")][source_name].append(row)
    by_type_metrics = {
        qtype: {source_name: metrics(rows) for source_name, rows in source_rows.items()}
        for qtype, source_rows in sorted(by_type.items())
    }
    return overall, by_type_metrics


def final_main_table(final_summary: dict) -> list[dict]:
    return [
        {
            "dataset": "ScanRefer",
            "method": "E0 baseline",
            "acc_iou25": 0.504,
            "acc_iou50": 0.452,
            "mean_iou": 0.4306,
            "source": "inputs/scanrefer/full_E0_baseline_qwen72b.jsonl",
        },
        {
            "dataset": "ScanRefer",
            "method": "evidence router",
            **final_summary["scanrefer"],
            "source": "outputs/universal_evidence_router/scanrefer_universal_evidence_routed_results.jsonl",
        },
        {
            "dataset": "NR3D",
            "method": "E0 baseline",
            "acc_iou25": 0.612,
            "acc_iou50": 0.604,
            "mean_iou": 0.6107,
            "source": "inputs/nr3d/official_e0_nr3d_openrouter_qwen_250.jsonl",
        },
        {
            "dataset": "NR3D",
            "method": "evidence router",
            **final_summary["nr3d"],
            "source": "outputs/universal_evidence_router/nr3d_universal_evidence_routed_results.jsonl",
        },
    ]


def optional_result(path: Path | None, label: str) -> dict:
    if not path:
        return {"status": "not_configured", "label": label}
    if not path.exists():
        return {"status": "missing", "label": label, "path": str(path)}
    if path.suffix == ".jsonl":
        rows = load_jsonl(path)
        return {"status": "available", "label": label, "path": str(path), "metrics": metrics(rows)}
    if path.suffix == ".json":
        obj = json.loads(path.read_text())
        return {"status": "available", "label": label, "path": str(path), "summary": obj}
    return {"status": "available", "label": label, "path": str(path), "note": "Unsupported extension for metric aggregation."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--llm-router-result", type=Path, default=None)
    ap.add_argument("--vlm-model-result", type=Path, default=None)
    ap.add_argument("--gt-proposal-result", type=Path, default=None)
    ap.add_argument("--runtime-result", type=Path, default=None)
    ap.add_argument("--failure-case-result", type=Path, default=None)
    args = ap.parse_args()

    final_out = args.out_dir / "main_table/final_router_outputs"
    final_summary = run_final_router(args.input_dir, final_out)
    scan_overall, scan_by_type = scanrefer_source_tables(args.input_dir)

    outputs = {
        "main_table": final_main_table(final_summary),
        "ablation": {
            "input_format_overall_scanrefer": scan_overall,
            "input_format_by_query_type_scanrefer": scan_by_type,
            "dictionary_vs_llm_router": optional_result(args.llm_router_result, "LLM router"),
            "vlm_model_change": optional_result(args.vlm_model_result, "alternate VLM model"),
            "gt_proposal": optional_result(args.gt_proposal_result, "GT proposal"),
            "runtime": optional_result(args.runtime_result, "runtime comparison"),
            "failure_visualization": optional_result(args.failure_case_result, "failure case visualization"),
        },
        "notes": [
            "Main table and ScanRefer input-format ablations are computed from bundled source outputs.",
            "LLM-router, VLM-model, GT-proposal, runtime, and failure-case rows accept externally generated result files.",
        ],
    }
    write_json(args.out_dir / "summary.json", outputs)
    write_json(args.out_dir / "main_table/table.json", outputs["main_table"])
    write_json(args.out_dir / "ablation/input_format_overall_scanrefer.json", scan_overall)
    write_json(args.out_dir / "ablation/input_format_by_query_type_scanrefer.json", scan_by_type)
    write_json(args.out_dir / "ablation/optional_experiments.json", outputs["ablation"])
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
