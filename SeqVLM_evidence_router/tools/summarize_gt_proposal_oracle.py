#!/usr/bin/env python3
"""Summarize Nr3D Mask3D proposal oracle diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "experiments/gt_proposal/nr3d_mask3d_oracle_case_table.csv"
OUT = REPO / "experiments/gt_proposal/summary.json"


def as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    n = len(rows)
    oracle25 = sum(as_bool(r["oracle25"]) for r in rows)
    oracle50 = sum(as_bool(r["oracle50"]) for r in rows)
    zero_candidates = sum(int(r["n_candidates"]) == 0 for r in rows)
    zero_canvas = sum(int(r["n_candidates_with_canvas"]) == 0 for r in rows)
    avg_candidates = sum(int(r["n_candidates"]) for r in rows) / n
    avg_canvas_candidates = sum(int(r["n_candidates_with_canvas"]) for r in rows) / n
    best_iou_mean = sum(float(r["best_iou"]) for r in rows) / n
    result = {
        "dataset": "nr3d_250",
        "diagnostic": "Mask3D candidate proposal oracle, GT used only for IoU scoring",
        "num_queries": n,
        "oracle25": round(oracle25 / n, 4),
        "oracle50": round(oracle50 / n, 4),
        "oracle25_count": oracle25,
        "oracle50_count": oracle50,
        "best_iou_mean": round(best_iou_mean, 4),
        "avg_candidates": round(avg_candidates, 4),
        "avg_candidates_with_canvas": round(avg_canvas_candidates, 4),
        "zero_candidate_cases": zero_candidates,
        "zero_canvas_candidate_cases": zero_canvas,
        "source_csv": str(CSV_PATH.relative_to(REPO)),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
