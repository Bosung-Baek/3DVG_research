#!/usr/bin/env python3
"""Prepare API-free validation-protocol inputs and diagnostics.

This script does not call any VLM. It builds the official ScanRefer val and
Nr3D validation annotation files with target object names, derives a
deterministic relation_source from caption/program cues, applies the final
evidence-aware router, and checks candidate/canvas coverage against the local
Mask3D/SeqVLM assets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOSUNG = REPO.parents[0]
SEQVLM = BOSUNG / "SeqVLM"

sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(SEQVLM))
sys.path.insert(0, str(SEQVLM / "tools"))
sys.path.insert(0, str(SEQVLM / "seqvlm"))

from query_type_router import route_for_row_universal  # noqa: E402
from p1p4_pool_utils import normalize_label  # noqa: E402
from run_nr3d_mask3d_oracle import normalize_singular  # noqa: E402
from seqvlm.utils import load_pc, load_seg_inst  # noqa: E402


DEFAULT_OUT = REPO / "experiments" / "full_dataset_preprocessing"
DEFAULT_SCANREFER_OFFICIAL = Path("/data/knuvi/bosung/scanrefer/ScanRefer_filtered_val.json")


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_first_loc_object(program: str | None) -> str:
    m = re.search(r"LOC\(object='([^']+)'\)", program or "")
    return m.group(1).strip() if m else ""


def derive_relation_source(row: dict) -> str:
    program = str(row.get("program") or "").upper()
    text = str(row.get("caption") or row.get("query") or "").lower().replace("-", " ")
    if row.get("view_dep"):
        return "viewpoint_guided"
    if any(op in program for op in ("CLOSEST", "FARTHEST")):
        return "proximity_derived"
    if any(term in text for term in ("near", "next to", "beside", "closest", "farthest", "farther", "nearer", "away from")):
        return "proximity_derived"
    if any(op in program for op in ("BETWEEN", "ABOVE", "BELOW", "UNDER", "INSIDE")):
        return "geometric"
    if any(term in text for term in ("between", "under", "below", "above", "inside", "surrounded")):
        return "geometric"
    if any(term in text for term in ("leftmost", "rightmost", "middle", "center", "centre", "second", "third", "first", "last", "northwestern", "northeastern", "southwestern", "southeastern")):
        return "ordinal"
    if any(op in program for op in ("LEFT", "RIGHT", "FRONT", "BEHIND")):
        return "explicit_direction"
    return "none"


def derive_anchor_class(row: dict) -> str:
    program = str(row.get("program") or "")
    locs = re.findall(r"LOC\(object='([^']+)'\)", program)
    return locs[1].strip() if len(locs) > 1 else ""


def relation_name(row: dict) -> str:
    program = str(row.get("program") or "").upper()
    for op in ("CLOSEST", "FARTHEST", "BETWEEN", "ABOVE", "BELOW", "UNDER", "INSIDE", "LEFT", "RIGHT", "FRONT", "BEHIND"):
        if op in program:
            return op
    return "NONE"


def build_scanrefer_official_val(path: Path) -> list[dict]:
    rows = read_json(path)
    out = []
    for case_id, row in enumerate(rows):
        converted = {
            "scan_id": row["scene_id"],
            "target_id": row["object_id"],
            "caption": row["description"],
            "obj_name": row["object_name"],
            "ann_id": row.get("ann_id"),
            "token": row.get("token", []),
        }
        relation_source = derive_relation_source(converted)
        query_type, route, reason = route_for_row_universal(
            {"relation_source": relation_source, "caption": converted["caption"]}
        )
        out.append(
            {
                "case": case_id,
                "dataset": "scanrefer_official_val_9508",
                **converted,
                "raw_scene_id": row["scene_id"],
                "raw_object_id": row["object_id"],
                "raw_object_name": row["object_name"],
                "raw_ann_id": row.get("ann_id"),
                "relation": relation_name(row),
                "anchor_class": derive_anchor_class(row),
                "relation_source": relation_source,
                "query_type": query_type,
                "route": route,
                "route_reason": reason,
            }
        )
    return out


def build_nr3d_full() -> list[dict]:
    rows = read_json(SEQVLM / "data" / "nr3d_val_with_obj_name.json")
    out = []
    for case_id, row in enumerate(rows):
        relation_source = derive_relation_source(row)
        query_type, route, reason = route_for_row_universal(
            {"relation_source": relation_source, "caption": row["caption"]}
        )
        out.append(
            {
                "case": case_id,
                "dataset": "nr3d_full_val",
                **row,
                "relation": relation_name(row),
                "anchor_class": derive_anchor_class(row),
                "relation_source": relation_source,
                "query_type": query_type,
                "route": route,
                "route_reason": reason,
            }
        )
    return out


def attach_candidate_coverage(rows: list[dict], canvas_root: Path, normalizer_name: str) -> dict:
    scene_cache: dict[str, tuple] = {}
    normalizer = normalize_singular if normalizer_name == "nr3d" else normalize_label
    counts = Counter()
    total_candidates = 0
    total_canvas_candidates = 0
    scenes = set()
    examples = []

    for row in rows:
        scene_id = row["scan_id"]
        scenes.add(scene_id)
        obj_name = row.get("obj_name") or ""
        if not obj_name:
            counts["no_category"] += 1
            row["candidate_coverage_status"] = "no_category"
            row["pre_e0_same_class_candidate_ids"] = []
            row["pre_e0_canvas_candidate_ids"] = []
            row["num_same_class_candidates"] = 0
            row["num_canvas_candidates"] = 0
            continue
        try:
            if scene_id not in scene_cache:
                if normalizer_name == "nr3d":
                    obj_ids, labels, locs = load_pc(scene_id)
                    scene_cache[scene_id] = (obj_ids, labels, locs)
                else:
                    labels, locs, scores, center = load_seg_inst(scene_id)
                    scene_cache[scene_id] = (list(range(len(labels))), labels, locs)
            obj_ids, labels, _locs = scene_cache[scene_id]
        except Exception as exc:
            counts["missing_scene"] += 1
            row["candidate_coverage_status"] = "missing_scene"
            row["pre_e0_same_class_candidate_ids"] = []
            row["pre_e0_canvas_candidate_ids"] = []
            row["num_same_class_candidates"] = 0
            row["num_canvas_candidates"] = 0
            if len(examples) < 8:
                examples.append({"case": row["case"], "scene_id": scene_id, "issue": "missing_scene", "error": repr(exc)})
            continue

        target = normalizer(obj_name)
        candidates = [int(obj_id) for obj_id, label in zip(obj_ids, labels) if normalizer(label) == target]
        row["pre_e0_same_class_candidate_ids"] = candidates
        row["num_same_class_candidates"] = len(candidates)
        if not candidates:
            counts["no_candidate"] += 1
            row["candidate_coverage_status"] = "no_candidate"
            row["pre_e0_canvas_candidate_ids"] = []
            row["num_canvas_candidates"] = 0
            if len(examples) < 8:
                examples.append({"case": row["case"], "scene_id": scene_id, "obj_name": obj_name, "issue": "no_candidate"})
            continue
        canvas_candidates = [idx for idx in candidates if (canvas_root / scene_id / str(idx) / "canvas.jpg").exists()]
        row["pre_e0_canvas_candidate_ids"] = canvas_candidates
        row["num_canvas_candidates"] = len(canvas_candidates)
        total_candidates += len(candidates)
        total_canvas_candidates += len(canvas_candidates)
        if canvas_candidates:
            counts["ok_with_canvas"] += 1
            row["candidate_coverage_status"] = "ok_with_canvas"
        else:
            counts["no_canvas_candidate"] += 1
            row["candidate_coverage_status"] = "no_canvas_candidate"
            if len(examples) < 8:
                examples.append({"case": row["case"], "scene_id": scene_id, "obj_name": obj_name, "issue": "no_canvas_candidate", "n_candidates": len(candidates)})

    denom = counts["ok_with_canvas"] + counts["no_canvas_candidate"]
    return {
        "num_queries": len(rows),
        "num_scenes": len(scenes),
        "num_canvas_scenes": len([p for p in canvas_root.iterdir() if p.is_dir()]) if canvas_root.exists() else 0,
        **dict(counts),
        "avg_candidates_when_found": round(total_candidates / max(denom, 1), 4),
        "avg_canvas_candidates_when_found": round(total_canvas_candidates / max(denom, 1), 4),
        "examples": examples,
    }


def summarize(rows: list[dict], coverage: dict) -> dict:
    return {
        "num_queries": len(rows),
        "route_counts": dict(Counter(row["route"] for row in rows)),
        "query_type_counts": dict(Counter(row["query_type"] for row in rows)),
        "route_reason_counts": dict(Counter(row["route_reason"] for row in rows)),
        "coverage": coverage,
    }


def sample_per_route(rows: list[dict], n: int) -> list[dict]:
    if n <= 0:
        return rows
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["route"], []).append(row)
    sampled = []
    for route in sorted(buckets):
        sampled.extend(buckets[route][:n])
    return sorted(sampled, key=lambda row: row["case"])


def coverage_misses(rows: list[dict]) -> list[dict]:
    keep_keys = [
        "case",
        "dataset",
        "scan_id",
        "target_id",
        "obj_name",
        "caption",
        "query_type",
        "route",
        "route_reason",
        "candidate_coverage_status",
        "num_same_class_candidates",
        "num_canvas_candidates",
        "pre_e0_same_class_candidate_ids",
        "pre_e0_canvas_candidate_ids",
    ]
    return [
        {key: row.get(key) for key in keep_keys}
        for row in rows
        if row.get("candidate_coverage_status") != "ok_with_canvas"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scanrefer-official-path", type=Path, default=DEFAULT_SCANREFER_OFFICIAL)
    parser.add_argument(
        "--sample-per-route",
        type=int,
        default=0,
        help="If positive, write only the first N rows for each routed input format.",
    )
    args = parser.parse_args()

    scanrefer = build_scanrefer_official_val(args.scanrefer_official_path)
    nr3d = build_nr3d_full()
    scanrefer_source_count = len(scanrefer)
    nr3d_source_count = len(nr3d)

    scanrefer = sample_per_route(scanrefer, args.sample_per_route)
    nr3d = sample_per_route(nr3d, args.sample_per_route)

    scan_cov = attach_candidate_coverage(scanrefer, SEQVLM / "data" / "scanrefer_preprocessed", "scanrefer")
    nr3d_cov = attach_candidate_coverage(nr3d, SEQVLM / "data" / "preprocessed_nr3d", "nr3d")

    write_jsonl(args.out_dir / "scanrefer_official_val_preprocessed.jsonl", scanrefer)
    write_jsonl(args.out_dir / "nr3d_full_preprocessed.jsonl", nr3d)
    write_jsonl(args.out_dir / "scanrefer_official_val_coverage_misses.jsonl", coverage_misses(scanrefer))
    write_jsonl(args.out_dir / "nr3d_full_coverage_misses.jsonl", coverage_misses(nr3d))
    write_json(args.out_dir / "scanrefer_official_val_summary.json", summarize(scanrefer, scan_cov))
    write_json(args.out_dir / "nr3d_full_summary.json", summarize(nr3d, nr3d_cov))
    write_json(
        args.out_dir / "summary.json",
        {
            "stage": "api_free_full_dataset_preprocessing",
            "sample_per_route": args.sample_per_route,
            "source_counts": {
                "scanrefer": scanrefer_source_count,
                "nr3d": nr3d_source_count,
            },
            "scanrefer": summarize(scanrefer, scan_cov),
            "nr3d": summarize(nr3d, nr3d_cov),
            "notes": [
                "No VLM/API calls are made in this preprocessing stage.",
                f"ScanRefer uses the official raw val annotation at {args.scanrefer_official_path} with {scanrefer_source_count} source rows.",
                "ScanRefer target category uses the official object_name field.",
                "ScanRefer relation_source is deterministic from caption cues because the official raw annotation has no ZSVG symbolic program.",
                "Nr3D relation_source is deterministic from ZSVG program, view_dep flag, and caption cues.",
            ],
        },
    )
    print(json.dumps(json.loads((args.out_dir / "summary.json").read_text()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
