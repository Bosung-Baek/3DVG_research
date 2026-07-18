#!/usr/bin/env python3
"""Analyze completed all-route Nr3D branch fill-in outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/ablation/nr3d_missing_branches/openrouter_qwen"

SOURCES = {
    "E0": REPO / "inputs/nr3d/official_e0_nr3d_openrouter_qwen_250.jsonl",
    "spatial_only_text": OUT / "spatial_only_all/nr3d_query_type_routed_vlm_results.jsonl",
    "3d_position_text": OUT / "3dpos_all/nr3d_query_type_routed_vlm_results.jsonl",
    "bev_labeled_layout": OUT / "bev_all/nr3d_query_type_routed_vlm_results.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def row_id(row: dict) -> int:
    return int(row.get("case", row.get("case_id", 0)))


def acc25(row: dict) -> bool:
    return bool(row.get("acc25") if "acc25" in row else row.get("success_iou25", False))


def acc50(row: dict) -> bool:
    return bool(row.get("acc50") if "acc50" in row else row.get("success_iou50", False))


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "num_queries": n,
        "acc_iou25": round(sum(acc25(r) for r in rows) / max(n, 1), 4),
        "acc_iou50": round(sum(acc50(r) for r in rows) / max(n, 1), 4),
        "mean_iou": round(sum(float(r.get("iou", 0) or 0) for r in rows) / max(n, 1), 4),
    }


def main() -> None:
    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing branch outputs:\n" + "\n".join(missing))

    rows_by_source = {name: {row_id(r): r for r in load_jsonl(path)} for name, path in SOURCES.items()}
    cases = sorted(rows_by_source["E0"])
    overall = {name: metrics([rows_by_source[name][case] for case in cases]) for name in SOURCES}

    qtypes = {case: rows_by_source["spatial_only_text"][case].get("query_type", "NA") for case in cases}
    by_type = {}
    for qtype in sorted(set(qtypes.values())):
        subset = [case for case in cases if qtypes[case] == qtype]
        by_type[qtype] = {
            name: metrics([rows_by_source[name][case] for case in subset])
            for name in SOURCES
        }

    oracle_rows = []
    contribution = Counter()
    for case in cases:
        candidates = []
        for name in SOURCES:
            row = rows_by_source[name][case]
            candidates.append((float(row.get("iou", 0) or 0), name, row))
        best_iou, best_name, best_row = max(candidates, key=lambda item: item[0])
        e0 = rows_by_source["E0"][case]
        oracle_rows.append(
            {
                "case": case,
                "query_type": qtypes[case],
                "caption": best_row.get("caption") or e0.get("caption"),
                "best_source": best_name,
                "best_iou": best_iou,
                "oracle_acc25": best_iou >= 0.25,
                "oracle_acc50": best_iou >= 0.50,
                "e0_iou": float(e0.get("iou", 0) or 0),
                "e0_acc25": acc25(e0),
                "e0_acc50": acc50(e0),
            }
        )
        contribution[best_name] += 1

    oracle_summary = metrics([{"iou": r["best_iou"], "acc25": r["oracle_acc25"], "acc50": r["oracle_acc50"]} for r in oracle_rows])
    e0_fail_oracle_success = sum((not r["e0_acc25"]) and r["oracle_acc25"] for r in oracle_rows)
    result = {
        "dataset": "nr3d_250",
        "model": "openrouter-qwen",
        "overall_by_input_format": overall,
        "by_query_type": by_type,
        "full_representation_oracle": {
            **oracle_summary,
            "best_source_counts": dict(contribution),
            "e0_fail_oracle_success": e0_fail_oracle_success,
        },
        "sources": {name: str(path.relative_to(REPO)) for name, path in SOURCES.items()},
    }
    write_json(OUT / "summary.json", result)
    write_json(OUT / "full_oracle_cases.json", oracle_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
