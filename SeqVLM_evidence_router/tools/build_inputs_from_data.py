#!/usr/bin/env python3
"""Build evaluator inputs from the repo-local data pack.

This is the data-generation step for the standalone evidence-router repo. It
copies the portable source records under data/source_outputs into the canonical
inputs/ layout consumed by tools/evaluate_universal_evidence_router.py.

The data pack stores completed source outputs, not the full ScanNet/Mask3D raw
asset tree. This keeps the repo portable while allowing inputs and final metrics
to be regenerated without the original SeqVLM workspace.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO / "data/source_outputs"
DEFAULT_OUT = REPO / "inputs"


REQUIRED_FILES = (
    ("scanrefer/full_E0_baseline_qwen72b.jsonl", "scanrefer/full_E0_baseline_qwen72b.jsonl"),
    ("scanrefer/bev_raw_labeled/results.jsonl", "scanrefer/bev_raw_labeled/results.jsonl"),
    (
        "scanrefer/seeground_ablation_spatial_only/results.jsonl",
        "scanrefer/seeground_ablation_spatial_only/results.jsonl",
    ),
    (
        "scanrefer/seeground_ablation_3dpos_only/results.jsonl",
        "scanrefer/seeground_ablation_3dpos_only/results.jsonl",
    ),
    ("nr3d/official_e0_nr3d_openrouter_qwen_250.jsonl", "nr3d/official_e0_nr3d_openrouter_qwen_250.jsonl"),
    ("nr3d/nr3d_dfrc_llm_parse.jsonl", "nr3d/nr3d_dfrc_llm_parse.jsonl"),
    (
        "nr3d/nr3d_query_type_routed_vlm_bev_exact_scanrefer_format/nr3d_query_type_routed_vlm_results.jsonl",
        "nr3d/nr3d_query_type_routed_vlm_bev_exact_scanrefer_format/nr3d_query_type_routed_vlm_results.jsonl",
    ),
)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing data source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_inputs(data_root: Path, out_dir: Path, clean: bool = False) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    for src_rel, dst_rel in REQUIRED_FILES:
        copy_file(data_root / src_rel, out_dir / dst_rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clean", action="store_true", help="Remove output inputs directory before rebuilding.")
    args = ap.parse_args()

    build_inputs(args.data_root, args.out_dir, clean=args.clean)
    print(f"Built inputs from {args.data_root} -> {args.out_dir}")


if __name__ == "__main__":
    main()
