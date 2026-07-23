#!/usr/bin/env python3
"""Reparse saved materialized-run answers and recompute metrics.

Useful when response parsing changes but API answers are already cached in a
results.jsonl file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BOSUNG = REPO.parents[0]
SEQVLM = BOSUNG / "SeqVLM"

sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(SEQVLM))
sys.path.insert(0, str(SEQVLM / "seqvlm"))

from run_materialized_vlm_eval import parse_e0_answer, parse_key_answer  # noqa: E402
from seqvlm.utils import calc_iou, load_pc, load_seg_inst  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mean(vals: list[float | bool]) -> float:
    return float(sum(vals) / max(len(vals), 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument("--out-summary", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.results_jsonl)
    gt_cache = {}
    seg_cache = {}

    def gt_box(scene_id: str, target_id: int):
        if scene_id not in gt_cache:
            gt_cache[scene_id] = load_pc(scene_id)
        obj_ids, _labels, locs = gt_cache[scene_id]
        target_id = int(target_id)
        if target_id not in obj_ids:
            return None
        return np.asarray(locs[obj_ids.index(target_id)], dtype=float)

    def seg_locs(scene_id: str):
        if scene_id not in seg_cache:
            seg_cache[scene_id] = load_seg_inst(scene_id)
        return seg_cache[scene_id][1]

    changed = 0
    reparsed = []
    for row in rows:
        new = dict(row)
        candidate_ids = [int(x) for x in row.get("candidate_ids") or []]
        pred = row.get("pred_instance")
        if candidate_ids and row.get("status") == "ok":
            if row.get("vlm_input_kind") == "e0_rgb_canvas_tournament":
                raw_answers = row.get("raw_answers") or []
                raw = raw_answers[-1] if raw_answers else row.get("raw_answer") or ""
                local_idx = parse_e0_answer(raw, len(candidate_ids))
                pred = candidate_ids[local_idx] if 0 <= local_idx < len(candidate_ids) else None
            else:
                local_idx = parse_key_answer(row.get("raw_answer") or "", len(candidate_ids))
                pred = candidate_ids[local_idx] if local_idx is not None else None
                new["selected_local_index_reparsed"] = local_idx
            if pred != row.get("pred_instance"):
                changed += 1
        gt = gt_box(row["scan_id"], row["target_id"])
        if pred is None or gt is None:
            iou = 0.0
        else:
            locs = seg_locs(row["scan_id"])
            iou = float(calc_iou(np.asarray(locs[int(pred)], dtype=float), gt)) if 0 <= int(pred) < len(locs) else 0.0
        new.update(
            {
                "pred_instance_reparsed": pred,
                "iou_reparsed": round(iou, 6),
                "acc25_reparsed": bool(iou >= 0.25),
                "acc50_reparsed": bool(iou >= 0.50),
            }
        )
        reparsed.append(new)

    summary = {
        "n": len(reparsed),
        "changed_predictions": changed,
        "old_acc25": mean([bool(r.get("acc25")) for r in rows]),
        "old_acc50": mean([bool(r.get("acc50")) for r in rows]),
        "old_miou": mean([float(r.get("iou") or 0.0) for r in rows]),
        "reparsed_acc25": mean([bool(r.get("acc25_reparsed")) for r in reparsed]),
        "reparsed_acc50": mean([bool(r.get("acc50_reparsed")) for r in reparsed]),
        "reparsed_miou": mean([float(r.get("iou_reparsed") or 0.0) for r in reparsed]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out_jsonl:
        write_jsonl(args.out_jsonl, reparsed)
    if args.out_summary:
        write_json(args.out_summary, summary)


if __name__ == "__main__":
    main()
