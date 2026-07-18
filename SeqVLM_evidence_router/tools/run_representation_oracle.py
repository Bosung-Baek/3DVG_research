#!/usr/bin/env python3
"""Compute available representation oracle upper bounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from analysis_common import ROUTES, SourcePack, route_final, success25, success50, write_json, write_jsonl  # noqa: E402

DEFAULT_INPUTS = REPO / "inputs"
DEFAULT_OUT = REPO / "experiments/ablation/representation_oracle"


def scanrefer_oracle(pack: SourcePack) -> tuple[list[dict], dict]:
    rows = []
    for item in pack.iter_items("scanrefer"):
        case = item["case"]
        e0 = pack.e0_row("scanrefer", case)
        candidates = {"E0": e0}
        for route in ROUTES:
            if route == "E0":
                continue
            source, _ = pack.source_row("scanrefer", case, route)
            candidates[route] = source
        best_route, best = max(candidates.items(), key=lambda kv: float(kv[1].get("iou", 0) or 0))
        final_route, _ = route_final(item)
        final_source, _ = pack.source_row("scanrefer", case, final_route)
        rows.append(
            {
                "dataset": "scanrefer",
                "case": case,
                "query_type": item["query_type"],
                "query": item["query"],
                "e0_acc25": success25(e0),
                "router_acc25": success25(final_source),
                "oracle_acc25": success25(best),
                "oracle_route": best_route,
                "oracle_iou": float(best.get("iou", 0) or 0),
                "router_route": final_route,
                "router_iou": float(final_source.get("iou", 0) or 0),
                "e0_iou": float(e0.get("iou", 0) or 0),
            }
        )
    return rows, oracle_summary(rows)


def nr3d_available_oracle(pack: SourcePack) -> tuple[list[dict], dict]:
    rows = []
    for item in pack.iter_items("nr3d"):
        case = item["case"]
        e0 = pack.e0_row("nr3d", case)
        available = pack.nr_routed[case]
        candidates = {"E0": e0, str(available.get("route")): available}
        best_route, best = max(candidates.items(), key=lambda kv: float(kv[1].get("iou", 0) or 0))
        final_route, _ = route_final(item)
        final_source, _ = pack.source_row("nr3d", case, final_route)
        rows.append(
            {
                "dataset": "nr3d",
                "case": case,
                "query_type": item["query_type"],
                "query": item["query"],
                "available_non_e0_route": available.get("route"),
                "e0_acc25": success25(e0),
                "router_acc25": success25(final_source),
                "oracle_acc25": success25(best),
                "oracle_route": best_route,
                "oracle_iou": float(best.get("iou", 0) or 0),
                "router_route": final_route,
                "router_iou": float(final_source.get("iou", 0) or 0),
                "e0_iou": float(e0.get("iou", 0) or 0),
            }
        )
    return rows, oracle_summary(rows)


def oracle_summary(rows: list[dict]) -> dict:
    n = len(rows)
    e0 = sum(r["e0_acc25"] for r in rows)
    router = sum(r["router_acc25"] for r in rows)
    oracle = sum(r["oracle_acc25"] for r in rows)
    recoverable = sum((not r["e0_acc25"]) and r["oracle_acc25"] for r in rows)
    recovered = sum((not r["e0_acc25"]) and r["router_acc25"] for r in rows)
    missed = sum((not r["e0_acc25"]) and r["oracle_acc25"] and not r["router_acc25"] for r in rows)
    return {
        "num_queries": n,
        "e0_acc25": round(e0 / n, 4),
        "router_acc25": round(router / n, 4),
        "oracle_acc25": round(oracle / n, 4),
        "e0_fail_oracle_success": recoverable,
        "router_recovered_from_oracle_pool": recovered,
        "missed_oracle_recoveries": missed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    pack = SourcePack(args.input_dir)
    scan_rows, scan_summary = scanrefer_oracle(pack)
    nr_rows, nr_summary = nr3d_available_oracle(pack)
    write_jsonl(args.out_dir / "scanrefer_oracle_cases.jsonl", scan_rows)
    write_jsonl(args.out_dir / "nr3d_available_oracle_cases.jsonl", nr_rows)
    summary = {"scanrefer": scan_summary, "nr3d_available_source": nr_summary}
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
