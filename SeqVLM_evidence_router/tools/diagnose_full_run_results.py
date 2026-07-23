#!/usr/bin/env python3
"""Diagnose a materialized full-dataset VLM run.

This is intentionally lightweight: it reads a partial or complete results.jsonl
and reports status, route performance, and candidate-pool oracle performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BOSUNG = REPO.parents[0]
SEQVLM = BOSUNG / "SeqVLM"

sys.path.insert(0, str(SEQVLM))
sys.path.insert(0, str(SEQVLM / "seqvlm"))

from seqvlm.utils import calc_iou, load_pc, load_seg_inst  # noqa: E402


ROUTE_DISPLAY = {
    "E0": "E0 RGB canvas",
    "seeground_ablation_spatial_only": "spatial-only text",
    "seeground_ablation_3dpos_only": "3D position text",
    "bev_raw_labeled": "BEV labeled layout",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def mean(values: list[float | bool]) -> float:
    return float(sum(values) / max(len(values), 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.results_jsonl)
    gt_cache = {}
    seg_cache = {}

    def get_gt_box(scene_id: str, target_id: int):
        if scene_id not in gt_cache:
            gt_cache[scene_id] = load_pc(scene_id)
        obj_ids, _labels, locs = gt_cache[scene_id]
        target_id = int(target_id)
        if target_id not in obj_ids:
            return None
        return np.asarray(locs[obj_ids.index(target_id)], dtype=float)

    def get_seg(scene_id: str):
        if scene_id not in seg_cache:
            seg_cache[scene_id] = load_seg_inst(scene_id)
        return seg_cache[scene_id]

    oracle_ious = []
    target_literal_in_candidates = 0
    no_gt = 0
    for row in rows:
        gt_box = get_gt_box(row["scan_id"], row["target_id"])
        if gt_box is None:
            no_gt += 1
            oracle_ious.append(0.0)
            continue
        _labels, locs, _scores, _center = get_seg(row["scan_id"])
        candidates = [int(x) for x in row.get("candidate_ids") or []]
        if int(row["target_id"]) in candidates:
            target_literal_in_candidates += 1
        best_iou = max(
            [float(calc_iou(np.asarray(locs[c], dtype=float), gt_box)) for c in candidates if 0 <= c < len(locs)]
            or [0.0]
        )
        oracle_ious.append(best_iou)

    def summarize_group(group: list[dict], idxs: list[int]) -> dict:
        vals = [oracle_ious[i] for i in idxs]
        return {
            "n": len(group),
            "acc25": mean([bool(r.get("acc25")) for r in group]),
            "acc50": mean([bool(r.get("acc50")) for r in group]),
            "miou": mean([float(r.get("iou") or 0.0) for r in group]),
            "candidate_oracle_acc25": mean([v >= 0.25 for v in vals]),
            "candidate_oracle_acc50": mean([v >= 0.50 for v in vals]),
            "candidate_oracle_miou": mean(vals),
        }

    by_route = {}
    route_to_idxs = defaultdict(list)
    for idx, row in enumerate(rows):
        route_to_idxs[row.get("route")].append(idx)
    for route, idxs in route_to_idxs.items():
        group = [rows[i] for i in idxs]
        by_route[ROUTE_DISPLAY.get(str(route), str(route))] = summarize_group(group, idxs)

    by_status = {}
    status_to_idxs = defaultdict(list)
    for idx, row in enumerate(rows):
        status_to_idxs[row.get("status")].append(idx)
    for status, idxs in status_to_idxs.items():
        group = [rows[i] for i in idxs]
        by_status[str(status)] = summarize_group(group, idxs)

    summary = {
        "results_path": str(args.results_jsonl),
        "num_rows": len(rows),
        "status_counts": dict(Counter(row.get("status") for row in rows)),
        "route_counts_raw": dict(Counter(row.get("route") for row in rows)),
        "route_counts_display": dict(Counter(ROUTE_DISPLAY.get(str(row.get("route")), str(row.get("route"))) for row in rows)),
        "target_literal_in_candidates": target_literal_in_candidates,
        "no_gt": no_gt,
        "overall": summarize_group(rows, list(range(len(rows)))),
        "by_status": by_status,
        "by_route": by_route,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
