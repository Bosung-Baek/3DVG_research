#!/usr/bin/env python3
"""Validate materialized VLM input packages and rewrite accurate summaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_dataset(root: Path, dataset: str) -> dict[str, Any]:
    dataset_root = root / dataset
    manifests = sorted(dataset_root.glob("case_*/manifest.json"))
    status = Counter()
    route = Counter()
    kind = Counter()
    missing_prompt_refs = 0
    missing_image_refs = 0
    errors = []

    for path in manifests:
        try:
            row = json.loads(path.read_text())
        except Exception as exc:
            errors.append({"manifest": str(path), "error": repr(exc)})
            continue
        status[row.get("status")] += 1
        route[row.get("route")] += 1
        kind[row.get("vlm_input_kind")] += 1

        for key in ("system_path", "prompt_path", "prompt_template_path"):
            rel = row.get(key)
            if rel and not (path.parent / rel).exists():
                missing_prompt_refs += 1
                if len(errors) < 20:
                    errors.append({"case": row.get("case"), "missing": rel})

        for rel in row.get("image_paths") or []:
            img = Path(rel) if str(rel).startswith("/") else path.parent / rel
            if not img.exists():
                missing_image_refs += 1
                if len(errors) < 20:
                    errors.append({"case": row.get("case"), "missing_image": rel})

    summary = {
        "dataset": dataset,
        "num_manifests": len(manifests),
        "status_counts": dict(status),
        "route_counts": dict(route),
        "vlm_input_kind_counts": dict(kind),
        "missing_prompt_refs": missing_prompt_refs,
        "missing_image_refs": missing_image_refs,
        "errors": errors,
    }
    write_json(dataset_root / "materialization_validation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/knuvi/bosung/evidence_router_vlm_inputs"))
    parser.add_argument("--datasets", default="scanrefer,nr3d")
    parser.add_argument("--rewrite-main-summary", action="store_true")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summaries = {dataset: validate_dataset(args.root, dataset) for dataset in datasets}
    write_json(args.root / "materialization_validation_summary.json", summaries)
    if args.rewrite_main_summary:
        write_json(args.root / "materialization_summary.json", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
