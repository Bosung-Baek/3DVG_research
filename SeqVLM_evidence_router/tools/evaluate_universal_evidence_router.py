#!/usr/bin/env python3
"""Evaluate the standalone universal evidence router.

This script is intentionally self-contained for the extracted evidence-router
repo. It does not call a VLM. It recomposes completed E0 and alternate-input
outputs using the shared dictionary/rule router in tools/query_type_router.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from query_type_router import route_for_row_universal  # noqa: E402


DEFAULT_INPUTS = REPO / "inputs"
DEFAULT_OUT = REPO / "outputs/universal_evidence_router"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def row_id(row: dict) -> int:
    for key in ("case", "case_id", "query_id"):
        if key in row:
            return int(str(row[key]).lstrip("0") or "0")
    raise KeyError(f"No case id in row keys: {sorted(row)}")


def success25(row: dict) -> bool:
    if "acc25" in row:
        return bool(row["acc25"])
    if "success_iou25" in row:
        return bool(row["success_iou25"])
    return float(row.get("iou", 0) or 0) >= 0.25


def success50(row: dict) -> bool:
    if "acc50" in row:
        return bool(row["acc50"])
    if "success_iou50" in row:
        return bool(row["success_iou50"])
    return float(row.get("iou", 0) or 0) >= 0.50


def compact_result(case: int, source: str, source_row: dict, route_row: dict, route_reason: str) -> dict:
    return {
        "case": case,
        "dataset": route_row["dataset"],
        "query_type": route_row["query_type"],
        "route": source,
        "route_reason": route_reason,
        "caption": route_row.get("caption", ""),
        "iou": float(source_row.get("iou", 0) or 0),
        "acc25": success25(source_row),
        "acc50": success50(source_row),
        "e0_iou": float(route_row["e0_row"].get("iou", 0) or 0),
        "e0_acc25": success25(route_row["e0_row"]),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    e0_recover = sum((not r["e0_acc25"]) and r["acc25"] for r in rows)
    e0_regress = sum(r["e0_acc25"] and not r["acc25"] for r in rows)
    return {
        "num_queries": n,
        "acc_iou25": round(sum(r["acc25"] for r in rows) / n, 4),
        "acc_iou50": round(sum(r["acc50"] for r in rows) / n, 4),
        "mean_iou": round(sum(r["iou"] for r in rows) / n, 4),
        "route_counts": dict(Counter(r["route"] for r in rows)),
        "route_reason_counts": dict(Counter(r["route_reason"] for r in rows)),
        "query_type_counts": dict(Counter(r["query_type"] for r in rows)),
        "e0_recoveries": e0_recover,
        "e0_regressions": e0_regress,
        "net_vs_e0": e0_recover - e0_regress,
    }


def grouped_summary(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "NA"))].append(row)
    return {name: summarize(vals) for name, vals in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))}


def scanrefer_rows(input_root: Path) -> list[dict]:
    scanrefer_inputs = input_root / "scanrefer"
    e0 = {row_id(r): r for r in load_jsonl(scanrefer_inputs / "full_E0_baseline_qwen72b.jsonl")}
    formats = {
        "bev_raw_labeled": {row_id(r): r for r in load_jsonl(scanrefer_inputs / "bev_raw_labeled/results.jsonl")},
        "seeground_ablation_spatial_only": {
            row_id(r): r for r in load_jsonl(scanrefer_inputs / "seeground_ablation_spatial_only/results.jsonl")
        },
        "seeground_ablation_3dpos_only": {
            row_id(r): r for r in load_jsonl(scanrefer_inputs / "seeground_ablation_3dpos_only/results.jsonl")
        },
    }
    labels = formats["bev_raw_labeled"]
    rows = []
    for case in sorted(e0):
        label = labels[case]
        route_input = {
            "dataset": "scanrefer_250",
            "query_type": label.get("query_type", "uncategorized"),
            "query": label.get("query") or label.get("caption") or e0[case].get("caption", ""),
            "e0_row": e0[case],
        }
        query_type, route, reason = route_for_row_universal(route_input)
        route_input["query_type"] = query_type
        route_input["caption"] = route_input["query"]
        source_row = e0[case] if route == "E0" else formats[route][case]
        rows.append(compact_result(case, route, source_row, route_input, reason))
    return rows


def nr3d_rows(input_root: Path) -> list[dict]:
    nr3d_inputs = input_root / "nr3d"
    nr3d_route = (
        nr3d_inputs
        / "nr3d_query_type_routed_vlm_bev_exact_scanrefer_format"
        / "nr3d_query_type_routed_vlm_results.jsonl"
    )
    e0 = {row_id(r): r for r in load_jsonl(nr3d_inputs / "official_e0_nr3d_openrouter_qwen_250.jsonl")}
    parse = {row_id(r): r for r in load_jsonl(nr3d_inputs / "nr3d_dfrc_llm_parse.jsonl")}
    routed = {row_id(r): r for r in load_jsonl(nr3d_route)}
    rows = []
    for case in sorted(e0):
        parse_row = parse[case]
        route_input = {
            "dataset": "nr3d_250",
            "relation_source": parse_row.get("relation_source", "uncategorized"),
            "caption": parse_row.get("caption") or e0[case].get("caption", ""),
            "e0_row": e0[case],
        }
        query_type, route, reason = route_for_row_universal(route_input)
        route_input["query_type"] = query_type
        source_row = e0[case] if route == "E0" else routed[case]
        rows.append(compact_result(case, route, source_row, route_input, reason))
    return rows


def write_dataset(out_dir: Path, name: str, rows: list[dict]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{name}_universal_evidence_routed_results.jsonl"
    with result_path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out_dir / f"{name}_per_query_type_summary.json").write_text(
        json.dumps(grouped_summary(rows, "query_type"), indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    scanrefer = scanrefer_rows(args.input_dir)
    nr3d = nr3d_rows(args.input_dir)
    summaries = {
        "policy": "universal_evidence_router_v2_proximity_first",
        "note": "Dataset-agnostic recomposition. No VLM calls.",
        "scanrefer": write_dataset(args.out_dir, "scanrefer", scanrefer),
        "nr3d": write_dataset(args.out_dir, "nr3d", nr3d),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
