#!/usr/bin/env python3
"""Run policy cumulative and router-component ablations by recomposition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from analysis_common import (  # noqa: E402
    POLICY_VARIANTS,
    SourcePack,
    evaluate_policy,
    grouped_summary,
    summarize,
    write_json,
    write_jsonl,
)

DEFAULT_INPUTS = REPO / "inputs"
DEFAULT_OUT = REPO / "experiments/ablation/policy_ablation"

CUMULATIVE = [
    "e0_only",
    "proximity_only",
    "proximity_plus_ordinal",
    "proximity_plus_geometric",
    "proximity_plus_ordinal_geometric",
    "full_router",
]

COMPONENTS = [
    "full_router",
    "without_visual_fallback",
    "without_purity_constraint",
    "without_viewpoint_fallback",
    "without_priority_ordering",
]


def run_group(pack: SourcePack, names: list[str], out_dir: Path) -> dict:
    summary = {}
    for name in names:
        router = POLICY_VARIANTS[name]
        rows = []
        for dataset in ("scanrefer", "nr3d"):
            ds_rows = evaluate_policy(pack, dataset, name, router)
            rows.extend(ds_rows)
            write_jsonl(out_dir / name / f"{dataset}_results.jsonl", ds_rows)
            write_json(out_dir / name / f"{dataset}_by_route.json", grouped_summary(ds_rows, "route"))
            summary.setdefault(name, {})[dataset] = summarize(ds_rows)
        write_jsonl(out_dir / name / "combined_results.jsonl", rows)
        summary[name]["combined"] = summarize(rows)
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    pack = SourcePack(args.input_dir)
    cumulative = run_group(pack, CUMULATIVE, args.out_dir / "cumulative")
    components = run_group(pack, COMPONENTS, args.out_dir / "components")
    final = {"cumulative": cumulative, "components": components}
    write_json(args.out_dir / "summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
