#!/usr/bin/env python3
"""Run final-router Nr3D with an OpenRouter model override.

The original SeqVLM API accepts config aliases only. This wrapper injects a
temporary alias that reuses the existing openrouter-qwen key/base-url while
changing the model id, then runs the final evidence router wrapper.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SEQVLM = Path("/home/knuvi/bosung/SeqVLM")
ORIGINAL = SEQVLM / "tools/run_nr3d_query_type_routed_vlm_eval.py"

sys.path.insert(0, str(REPO / "tools"))
from query_type_router import route_for_row_universal  # noqa: E402


def parse_wrapper_args() -> tuple[str, str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--openrouter-model-id", required=True)
    parser.add_argument("--alias", default="openrouter-override")
    args, rest = parser.parse_known_args()
    return args.openrouter_model_id, args.alias, rest


def load_original_module():
    spec = importlib.util.spec_from_file_location("seqvlm_nr3d_model_override_runner", ORIGINAL)
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
    model_id, alias, rest = parse_wrapper_args()
    sys.argv = [sys.argv[0], "--model", alias, *rest]

    module = load_original_module()
    import seqvlm.api as seqvlm_api

    base_cfg = dict(seqvlm_api.configs["openrouter-qwen"])
    base_cfg["model"] = model_id
    seqvlm_api.configs[alias] = base_cfg

    module.route_for_row = final_route_for_row
    module.ROUTE_POLICY = {
        "policy": "universal_evidence_router_v2_proximity_first",
        "implementation": str(REPO / "tools/query_type_router.py"),
        "patched_runner": str(ORIGINAL),
        "model_override": model_id,
        "alias": alias,
    }
    module.main()


if __name__ == "__main__":
    main()
