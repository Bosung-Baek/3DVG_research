#!/usr/bin/env python3
"""Compare end-to-end VLM rerun against final recomposition outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RERUN = REPO / "experiments/end_to_end_nr3d_final_router_openrouter_qwen/nr3d_query_type_routed_vlm_results.jsonl"
DEFAULT_FINAL = REPO / "outputs/universal_evidence_router/nr3d_universal_evidence_routed_results.jsonl"
DEFAULT_E0 = REPO / "inputs/nr3d/official_e0_nr3d_openrouter_qwen_250.jsonl"
DEFAULT_OUT = REPO / "experiments/end_to_end_nr3d_final_router_openrouter_qwen/consistency_summary.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def by_case(rows: list[dict]) -> dict[int, dict]:
    return {int(r["case"]): r for r in rows}


def acc(row: dict) -> bool:
    return bool(row.get("acc25") if "acc25" in row else row.get("success_iou25"))


def mean(rows: list[dict], key: str) -> float:
    return sum(float(r.get(key, 0) or 0) for r in rows) / max(len(rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", type=Path, default=DEFAULT_RERUN)
    ap.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    ap.add_argument("--e0", type=Path, default=DEFAULT_E0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rerun = by_case(load_jsonl(args.rerun))
    final = by_case(load_jsonl(args.final))
    e0 = by_case(load_jsonl(args.e0))
    rows = [rerun[i] for i in sorted(rerun)]

    route_summary = {}
    for route in sorted(set(r["route"] for r in rows)):
        ids = [int(r["case"]) for r in rows if r["route"] == route]
        route_summary[route] = {
            "num_queries": len(ids),
            "rerun_acc25": round(sum(acc(rerun[i]) for i in ids) / len(ids), 4),
            "final_recomposition_acc25": round(sum(acc(final[i]) for i in ids) / len(ids), 4),
            "e0_acc25": round(sum(acc(e0[i]) for i in ids) / len(ids), 4),
            "final_correct_rerun_wrong": sum(acc(final[i]) and not acc(rerun[i]) for i in ids),
            "final_wrong_rerun_correct": sum((not acc(final[i])) and acc(rerun[i]) for i in ids),
        }

    changed = []
    for i in sorted(rerun):
        if acc(rerun[i]) != acc(final[i]):
            changed.append(
                {
                    "case": i,
                    "route": rerun[i]["route"],
                    "query_type": rerun[i].get("query_type"),
                    "caption": rerun[i].get("caption"),
                    "final_acc25": acc(final[i]),
                    "rerun_acc25": acc(rerun[i]),
                    "final_iou": float(final[i].get("iou", 0) or 0),
                    "rerun_iou": float(rerun[i].get("iou", 0) or 0),
                }
            )

    out = {
        "num_queries": len(rows),
        "rerun_acc25": round(sum(acc(r) for r in rows) / len(rows), 4),
        "final_recomposition_acc25": round(sum(acc(final[i]) for i in rerun) / len(rows), 4),
        "e0_acc25": round(sum(acc(e0[i]) for i in rerun) / len(rows), 4),
        "rerun_mean_iou": round(mean(rows, "iou"), 4),
        "route_counts": dict(Counter(r["route"] for r in rows)),
        "route_summary": route_summary,
        "acc25_transition_counts": {
            str(k): v for k, v in Counter((acc(final[i]), acc(rerun[i]), rerun[i]["route"]) for i in rerun).items()
        },
        "num_acc25_changed_vs_final": len(changed),
        "changed_cases": changed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
