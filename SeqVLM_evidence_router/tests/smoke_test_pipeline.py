#!/usr/bin/env python3
"""Smoke-test the standalone evidence-router pipeline.

The test covers three layers:
1. VLM input-format generation modules can be imported and build prompts.
2. The final query router maps representative queries to expected routes.
3. The final evaluator reproduces the locked output files exactly.

The input-format smoke test uses synthetic scene metadata. It does not require
external ScanNet/Mask3D assets. BEV rendering may have no image without those
assets, but the BEV module must still build the coordinate prompt and metadata.
"""

from __future__ import annotations

import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from input_formats import get_format  # noqa: E402
from tools.query_type_router import route_for_row_universal  # noqa: E402


EXPECTED = {
    "scanrefer": {"acc_iou25": 0.52, "acc_iou50": 0.468, "mean_iou": 0.4455},
    "nr3d": {"acc_iou25": 0.652, "acc_iou50": 0.648, "mean_iou": 0.6514},
}


def synthetic_scene() -> dict:
    return {
        "labels": ["chair", "chair", "desk", "cup"],
        "locs": np.asarray(
            [
                [0.0, 0.0, 0.5, 0.6, 0.6, 1.0],
                [1.0, 0.0, 0.5, 0.6, 0.6, 1.0],
                [0.0, 1.5, 0.4, 1.5, 0.8, 0.8],
                [0.5, 0.5, 1.0, 0.2, 0.2, 0.2],
            ],
            dtype=float,
        ),
        "ins_pcds": np.asarray(
            [
                np.asarray([[0.0, 0.0, 0.0, 120, 120, 120], [0.2, 0.2, 1.0, 120, 120, 120]], dtype=float),
                np.asarray([[1.0, 0.0, 0.0, 140, 140, 140], [1.2, 0.2, 1.0, 140, 140, 140]], dtype=float),
                np.asarray([[0.0, 1.5, 0.0, 80, 80, 80], [0.4, 1.8, 0.8, 80, 80, 80]], dtype=float),
                np.asarray([[0.5, 0.5, 1.0, 200, 200, 200], [0.6, 0.6, 1.2, 200, 200, 200]], dtype=float),
            ],
            dtype=object,
        ),
    }


def ensure_smoke_canvas() -> None:
    canvas_path = REPO / "data/scanrefer_preprocessed/smoke_scene/0/canvas.jpg"
    canvas_path.parent.mkdir(parents=True, exist_ok=True)
    if canvas_path.exists():
        return
    img = Image.new("RGB", (320, 240), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([90, 50, 230, 190], outline=(220, 20, 20), width=6)
    draw.text((12, 12), "smoke canvas candidate 0", fill=(0, 0, 0))
    img.save(canvas_path, "JPEG", quality=90)


def assert_format_builds() -> None:
    ensure_smoke_canvas()
    scene_data = synthetic_scene()
    common = {
        "query": "the chair nearest the desk",
        "scene_id": "smoke_scene",
        "candidates": [0, 1],
        "anchors": [2],
        "relation_info": {},
        "frame_data": {},
        "scene_data": scene_data,
        "config": {},
    }

    expected = {
        "baseline_e0": {"has_system": True, "uses_real_rgb": True},
        "seeground_ablation_spatial_only": {"has_system": True, "uses_spatial_text": True},
        "seeground_ablation_3dpos_only": {"has_system": True, "uses_spatial_text": True},
        "bev_raw_labeled": {"has_system": True, "uses_spatial_text": True},
    }
    for name, checks in expected.items():
        built = get_format(name).build(**common)
        assert "prompt" in built and built["prompt"], f"{name}: missing prompt"
        assert "metadata" in built, f"{name}: missing metadata"
        if name == "baseline_e0":
            assert built["images"][0], "baseline_e0: expected generated smoke canvas image"
        if checks.get("has_system"):
            assert "system" in built and built["system"], f"{name}: missing system prompt"
        for key, value in checks.items():
            if key == "has_system":
                continue
            assert built["metadata"].get(key) == value, f"{name}: expected metadata {key}={value}"


def assert_router_routes() -> None:
    examples = [
        (
            {"query_type": "proximity_derived", "query": "the red chair near the desk"},
            ("proximity_derived", "seeground_ablation_spatial_only", "pure_proximity_spatial"),
        ),
        (
            {"query_type": "ordinal", "query": "the cup in the middle"},
            ("ordinal", "bev_raw_labeled", "pure_ordinal_bev"),
        ),
        (
            {"query_type": "ordinal", "query": "the blue cup in the middle"},
            ("ordinal", "E0", "visual_attribute_default_e0"),
        ),
        (
            {"query_type": "geometric", "query": "the table between the couches"},
            ("geometric", "seeground_ablation_3dpos_only", "pure_geometric_3dpos"),
        ),
        (
            {"query_type": "viewpoint_guided", "query": "while facing the doorway, the desk on the right"},
            ("viewpoint_guided", "E0", "viewpoint_needs_local_frame_default_e0"),
        ),
    ]
    for row, expected in examples:
        actual = route_for_row_universal(row)
        assert actual == expected, f"route mismatch for {row}: {actual} != {expected}"


def assert_evaluator_reproduces_locked_outputs() -> None:
    with tempfile.TemporaryDirectory(prefix="evidence_router_smoke_") as tmp:
        input_dir = Path(tmp) / "inputs"
        out_dir = Path(tmp) / "universal_evidence_router"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools/build_inputs_from_data.py"),
                "--data-root",
                str(REPO / "data/source_outputs"),
                "--out-dir",
                str(input_dir),
                "--clean",
            ],
            check=True,
            cwd=REPO,
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools/evaluate_universal_evidence_router.py"),
                "--input-dir",
                str(input_dir),
                "--out-dir",
                str(out_dir),
            ],
            check=True,
            cwd=REPO,
        )

        summary = json.loads((out_dir / "summary.json").read_text())
        for dataset, metrics in EXPECTED.items():
            for key, value in metrics.items():
                actual = summary[dataset][key]
                assert actual == value, f"{dataset} {key}: {actual} != {value}"

        locked = REPO / "outputs/universal_evidence_router"
        for name in (
            "summary.json",
            "scanrefer_universal_evidence_routed_results.jsonl",
            "nr3d_universal_evidence_routed_results.jsonl",
        ):
            assert filecmp.cmp(locked / name, out_dir / name, shallow=False), f"locked output differs: {name}"


def assert_experiment_suite_runs() -> None:
    with tempfile.TemporaryDirectory(prefix="evidence_router_suite_") as tmp:
        out_dir = Path(tmp) / "experiments"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools/run_experiment_suite.py"),
                "--out-dir",
                str(out_dir),
            ],
            check=True,
            cwd=REPO,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads((out_dir / "summary.json").read_text())
        main_table = summary["main_table"]
        assert main_table[1]["acc_iou25"] == 0.52
        assert main_table[3]["acc_iou25"] == 0.652
        assert (out_dir / "ablation/input_format_overall_scanrefer.json").exists()
        assert (out_dir / "ablation/input_format_by_query_type_scanrefer.json").exists()


def assert_llm_router_ablation_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="evidence_router_llm_") as tmp:
        out_dir = Path(tmp) / "llm_router_mock"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools/run_llm_router_ablation.py"),
                "--mock-dictionary",
                "--max-samples",
                "8",
                "--out-dir",
                str(out_dir),
                "--quiet",
            ],
            check=True,
            cwd=REPO,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["route_diff_vs_dictionary"]["num_changed"] == 0
        assert (out_dir / "scanrefer_llm_router_results.jsonl").exists()


def main() -> None:
    assert_format_builds()
    assert_router_routes()
    assert_evaluator_reproduces_locked_outputs()
    assert_experiment_suite_runs()
    assert_llm_router_ablation_smoke()
    print("SMOKE TEST PASSED: input generation, routing, and final outputs are valid.")


if __name__ == "__main__":
    main()
