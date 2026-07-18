#!/usr/bin/env python3
"""Run the standalone evidence-router pipeline from data pack to metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=REPO / "data/source_outputs")
    ap.add_argument("--input-dir", type=Path, default=REPO / "inputs")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/universal_evidence_router")
    ap.add_argument("--clean-inputs", action="store_true")
    args = ap.parse_args()

    build_cmd = [
        sys.executable,
        str(REPO / "tools/build_inputs_from_data.py"),
        "--data-root",
        str(args.data_root),
        "--out-dir",
        str(args.input_dir),
    ]
    if args.clean_inputs:
        build_cmd.append("--clean")
    subprocess.run(build_cmd, check=True, cwd=REPO)

    eval_cmd = [
        sys.executable,
        str(REPO / "tools/evaluate_universal_evidence_router.py"),
        "--input-dir",
        str(args.input_dir),
        "--out-dir",
        str(args.out_dir),
    ]
    subprocess.run(eval_cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()
