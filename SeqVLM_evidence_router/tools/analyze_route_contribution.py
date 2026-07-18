#!/usr/bin/env python3
"""Analyze route-level recovery/regression for final evidence router."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from analysis_common import SourcePack, evaluate_policy, route_final, write_json, write_jsonl  # noqa: E402

DEFAULT_INPUTS = REPO / "inputs"
DEFAULT_OUT = REPO / "experiments/ablation/route_contribution"


def label(row: dict) -> str:
    if (not row["e0_acc25"]) and row["acc25"]:
        return "recovery"
    if row["e0_acc25"] and not row["acc25"]:
        return "regression"
    if row["acc25"]:
        return "both_correct"
    return "both_wrong"


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["route"])].append(row)
    out = {}
    for (dataset, route), vals in sorted(groups.items()):
        recover = sum(label(v) == "recovery" for v in vals)
        regress = sum(label(v) == "regression" for v in vals)
        out.setdefault(dataset, {})[route] = {
            "routed_n": len(vals),
            "e0_correct": sum(v["e0_acc25"] for v in vals),
            "route_correct": sum(v["acc25"] for v in vals),
            "recovery": recover,
            "regression": regress,
            "net": recover - regress,
            "acc_iou25": round(sum(v["acc25"] for v in vals) / len(vals), 4),
            "e0_acc_iou25": round(sum(v["e0_acc25"] for v in vals) / len(vals), 4),
            "source_unavailable_fallbacks": sum(bool(v.get("source_unavailable_fallback_e0")) for v in vals),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    pack = SourcePack(args.input_dir)
    rows = []
    for dataset in ("scanrefer", "nr3d"):
        rows.extend(evaluate_policy(pack, dataset, "full_router", route_final))
    transitions = []
    for row in rows:
        transition = {
            **row,
            "transition": label(row),
        }
        transitions.append(transition)

    summary = summarize(transitions)
    write_jsonl(args.out_dir / "transitions.jsonl", transitions)
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
