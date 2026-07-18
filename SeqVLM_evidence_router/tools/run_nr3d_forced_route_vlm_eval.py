#!/usr/bin/env python3
"""Run the original SeqVLM Nr3D VLM runner with one forced non-E0 route.

This is used for missing-branch fill-in: generate completed source outputs for
spatial-only, BEV, or 3D-position on all 250 Nr3D cases so representation
oracle and input-format ablations are not limited to the final-router branches.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SEQVLM = Path("/home/knuvi/bosung/SeqVLM")
ORIGINAL = SEQVLM / "tools/run_nr3d_query_type_routed_vlm_eval.py"


VALID_ROUTES = {
    "seeground_ablation_spatial_only",
    "bev_raw_labeled",
    "seeground_ablation_3dpos_only",
}


def parse_wrapper_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force-route", required=True, choices=sorted(VALID_ROUTES))
    args, rest = parser.parse_known_args()
    return args.force_route, rest


def load_original_module():
    spec = importlib.util.spec_from_file_location("seqvlm_nr3d_forced_route_runner", ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load original runner: {ORIGINAL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    forced_route, rest = parse_wrapper_args()
    sys.argv = [sys.argv[0], *rest]

    module = load_original_module()

    def forced_route_for_row(row: dict) -> tuple[str, str]:
        query_type = row.get("relation_source") or row.get("query_type") or "uncategorized"
        return str(query_type), forced_route

    module.route_for_row = forced_route_for_row
    module.ROUTE_POLICY = {
        "policy": "forced_route_fill_in",
        "forced_route": forced_route,
        "patched_runner": str(ORIGINAL),
    }
    module.main()


if __name__ == "__main__":
    main()
