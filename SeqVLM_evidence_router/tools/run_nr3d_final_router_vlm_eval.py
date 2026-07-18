#!/usr/bin/env python3
"""Run the original SeqVLM Nr3D route-first VLM runner with final router policy.

This wrapper reuses SeqVLM's VLM API, ScanNet loading, BEV rendering, spatial
prompt construction, and IoU evaluation. It only replaces the broad query-type
router inside `SeqVLM/tools/run_nr3d_query_type_routed_vlm_eval.py` with the
standalone final evidence-aware router.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEQVLM = Path("/home/knuvi/bosung/SeqVLM")
ORIGINAL = SEQVLM / "tools/run_nr3d_query_type_routed_vlm_eval.py"

sys.path.insert(0, str(REPO / "tools"))
from query_type_router import route_for_row_universal  # noqa: E402


def load_original_module():
    spec = importlib.util.spec_from_file_location("seqvlm_nr3d_route_runner", ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load original runner: {ORIGINAL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def final_route_for_row(row: dict) -> tuple[str, str]:
    query_type, route, _reason = route_for_row_universal(
        {
            "relation_source": row.get("relation_source"),
            "query_type": row.get("query_type"),
            "caption": row.get("caption"),
            "query": row.get("query"),
        }
    )
    return query_type, route


def main() -> None:
    module = load_original_module()
    module.route_for_row = final_route_for_row
    module.ROUTE_POLICY = {
        "policy": "universal_evidence_router_v2_proximity_first",
        "implementation": str(REPO / "tools/query_type_router.py"),
        "patched_runner": str(ORIGINAL),
    }
    module.main()


if __name__ == "__main__":
    main()
