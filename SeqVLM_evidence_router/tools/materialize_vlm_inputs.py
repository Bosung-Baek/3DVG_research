#!/usr/bin/env python3
"""Materialize VLM input prompts and image assets for full-dataset runs.

This is API-free preprocessing. It consumes the route-plan JSONL files produced
by `prepare_full_dataset_preprocessing.py` and writes per-query VLM input
packages:

  case_000123/
    manifest.json
    system.txt
    prompt.txt or prompt_template.txt
    images/*.jpg  (only for newly rendered assets such as BEV)

For E0, existing SeqVLM canvas images are referenced by absolute path rather
than copied, because copying all candidate canvases would duplicate many GB of
already-rendered data. BEV images are rendered and saved because they are
generated inputs.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BOSUNG = REPO.parents[0]
SEQVLM = BOSUNG / "SeqVLM"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(SEQVLM))
sys.path.insert(0, str(SEQVLM / "seqvlm"))

from input_formats.baseline_e0 import SYSTEM_PROMPT as E0_SYSTEM_PROMPT  # noqa: E402
from input_formats.baseline_e0 import USER_PROMPT as E0_USER_PROMPT  # noqa: E402
from input_formats.bev_raw_labeled import BevRawLabeledFormat  # noqa: E402
from seqvlm.utils import load_pc, load_seg_inst  # noqa: E402


DEFAULT_PREPROCESS_ROOT = REPO / "experiments" / "full_dataset_preprocessing"
DEFAULT_OUT_ROOT = Path("/data/knuvi/bosung/evidence_router_vlm_inputs")
ROUTE_E0 = "E0"
ROUTE_SPATIAL = "seeground_ablation_spatial_only"
ROUTE_3DPOS = "seeground_ablation_3dpos_only"
ROUTE_BEV = "bev_raw_labeled"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SCENE_CACHE: dict[str, dict[str, Any]] = {}
SCENE_LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def decode_b64_to_file(path: Path, b64: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))


def norm(value: str | None) -> str:
    return (value or "").lower().replace("-", " ").replace("_", " ").strip()


def dataset_canvas_root(dataset: str) -> Path:
    if dataset.startswith("nr3d"):
        return SEQVLM / "data" / "preprocessed_nr3d"
    return SEQVLM / "data" / "scanrefer_preprocessed"


def case_dir(out_root: Path, dataset_name: str, case_id: int) -> Path:
    return out_root / dataset_name / f"case_{case_id:06d}"


def load_scene_data(scene_id: str, dataset_name: str) -> dict[str, Any]:
    cache_key = f"{dataset_name}:{scene_id}"
    with SCENE_LOCK:
        if cache_key in SCENE_CACHE:
            return SCENE_CACHE[cache_key]
    if dataset_name == "nr3d":
        obj_ids, labels, locs = load_pc(scene_id)
        loc_by_id = {int(obj_id): np.asarray(loc, dtype=float) for obj_id, loc in zip(obj_ids, locs)}
        label_by_id = {int(obj_id): str(label) for obj_id, label in zip(obj_ids, labels)}
        max_id = max([int(x) for x in obj_ids] or [-1])
        indexed_labels = ["?"] * (max_id + 1)
        indexed_locs = [np.zeros(6, dtype=float) for _ in range(max_id + 1)]
        for obj_id in obj_ids:
            indexed_labels[int(obj_id)] = label_by_id[int(obj_id)]
            indexed_locs[int(obj_id)] = loc_by_id[int(obj_id)]
        scene = {
            "obj_ids": [int(x) for x in obj_ids],
            "labels": indexed_labels,
            "locs": indexed_locs,
            "loc_by_id": loc_by_id,
            "label_by_id": label_by_id,
            "scores": [],
            "center": None,
        }
    else:
        labels, locs, scores, center = load_seg_inst(scene_id)
        scene = {
            "obj_ids": list(range(len(labels))),
            "labels": labels,
            "locs": locs,
            "loc_by_id": {idx: np.asarray(loc, dtype=float) for idx, loc in enumerate(locs)},
            "label_by_id": {idx: str(label) for idx, label in enumerate(labels)},
            "scores": scores,
            "center": center,
        }
    with SCENE_LOCK:
        SCENE_CACHE[cache_key] = scene
    return scene


def find_anchors(row: dict[str, Any], scene: dict[str, Any], candidates: list[int], limit: int = 5) -> list[int]:
    anchor_class = norm(row.get("anchor_class"))
    if not anchor_class:
        return []
    cand_set = {int(x) for x in candidates}
    out = []
    for obj_id in scene["obj_ids"]:
        label = scene["label_by_id"].get(int(obj_id), "")
        label_norm = norm(label)
        if int(obj_id) in cand_set:
            continue
        if anchor_class == label_norm or anchor_class in label_norm or label_norm in anchor_class:
            out.append(int(obj_id))
            if len(out) >= limit:
                break
    return out


def format_box(loc: np.ndarray) -> str:
    c = loc[:3]
    s = loc[3:6]
    return f"center=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) size=({s[0]:.2f}x{s[1]:.2f}x{s[2]:.2f})"


def spatial_prompt(
    caption: str,
    candidates: list[int],
    anchors: list[int],
    loc_by_id: dict[int, np.ndarray],
    label_by_id: dict[int, str],
    mode: str,
) -> tuple[str, str]:
    if mode == "spatial_only":
        system = (
            "You are selecting an object using only spatial relationships, "
            "distances, ordering, object categories, and 3D geometry. Ignore "
            "visual appearance such as color, material, and texture."
        )
        lines = [
            f'Query: "{caption}"',
            "",
            "Use only relative spatial relationships, distances, ordering, and object categories.",
            "Ignore color, material, texture, and other visual appearance words.",
            "",
        ]
    else:
        system = (
            "You are selecting an object using structured 3D positions, sizes, "
            "heights, distances, global layout, and object categories."
        )
        lines = [
            f'Query: "{caption}"',
            "",
            "Use 3D object positions, sizes, height, distances, global layout, and object categories.",
            "Ignore texture/color unless it is needed only as a quoted part of the query.",
            "",
        ]

    anchor_locs = [np.asarray(loc_by_id[a], dtype=float) for a in anchors if a in loc_by_id]
    anchor_center = np.mean([a[:3] for a in anchor_locs], axis=0) if anchor_locs else None
    if anchors:
        lines.append("Anchors:")
        for aid in anchors:
            if aid in loc_by_id:
                lines.append(f"  id{aid} {label_by_id.get(aid, 'object')}: {format_box(np.asarray(loc_by_id[aid]))}")
        lines.append("")

    lines.append("Candidates:")
    for rank, cid in enumerate(candidates):
        letter = LETTERS[rank] if rank < len(LETTERS) else str(rank)
        if cid not in loc_by_id:
            continue
        loc = np.asarray(loc_by_id[cid], dtype=float)
        extra = ""
        if anchor_center is not None:
            delta = loc[:3] - anchor_center
            xy_dist = float(np.linalg.norm(delta[:2]))
            xyz_dist = float(np.linalg.norm(delta))
            extra = (
                f" anchor_delta=({delta[0]:+.2f},{delta[1]:+.2f},{delta[2]:+.2f})"
                f" xy_dist={xy_dist:.2f}m xyz_dist={xyz_dist:.2f}m"
            )
        lines.append(f"  {letter}. id{cid} {label_by_id.get(cid, 'object')}: {format_box(loc)}{extra}")
    lines.append("")
    letters = ", ".join(LETTERS[: min(len(candidates), len(LETTERS))])
    lines.append(f"Return only JSON: {{\"letter\": \"<one of {letters}>\"}}")
    return system, "\n".join(lines)


def link_or_reference_image(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
        return str(dst.relative_to(dst.parents[1]))
    except Exception:
        return str(src.resolve())


def build_e0(row: dict[str, Any], out_dir: Path, candidates: list[int]) -> dict[str, Any]:
    root = dataset_canvas_root(row["dataset"])
    image_paths = []
    missing = []
    for rank, cid in enumerate(candidates):
        p = root / row["scan_id"] / str(cid) / "canvas.jpg"
        if p.exists():
            rel = Path("images") / f"candidate_{rank:03d}_id{cid}.jpg"
            image_paths.append(link_or_reference_image(p, out_dir / rel))
        else:
            missing.append(cid)
    write_text(out_dir / "system.txt", E0_SYSTEM_PROMPT)
    write_text(out_dir / "prompt_template.txt", E0_USER_PROMPT)
    return {
        "vlm_input_kind": "e0_rgb_canvas_tournament",
        "system_path": "system.txt",
        "prompt_template_path": "prompt_template.txt",
        "prompt_template": E0_USER_PROMPT,
        "query": row["caption"],
        "needs_tournament_batching": True,
        "image_paths": image_paths,
        "missing_image_candidate_ids": missing,
    }


def build_text_only(row: dict[str, Any], out_dir: Path, scene: dict[str, Any], candidates: list[int], anchors: list[int], mode: str) -> dict[str, Any]:
    system, prompt = spatial_prompt(row["caption"], candidates, anchors, scene["loc_by_id"], scene["label_by_id"], mode)
    write_text(out_dir / "system.txt", system)
    write_text(out_dir / "prompt.txt", prompt)
    return {
        "vlm_input_kind": mode,
        "system_path": "system.txt",
        "prompt_path": "prompt.txt",
        "image_paths": [],
    }


def build_bev(row: dict[str, Any], out_dir: Path, scene: dict[str, Any], candidates: list[int], anchors: list[int]) -> dict[str, Any]:
    fmt = BevRawLabeledFormat()
    relation_info = {
        "relation": row.get("relation", "NONE"),
        "relation_source": row.get("relation_source", "none"),
        "anchor_class": row.get("anchor_class", ""),
    }
    package = fmt.build(
        row["caption"],
        row["scan_id"],
        candidates,
        anchors,
        relation_info,
        {},
        scene,
        {"bev_img_size": 900},
    )
    write_text(out_dir / "system.txt", package.get("system", ""))
    write_text(out_dir / "prompt.txt", package.get("prompt", ""))
    image_paths = []
    for idx, b64 in enumerate(package.get("images") or []):
        rel = Path("images") / f"bev_{idx}.jpg"
        decode_b64_to_file(out_dir / rel, b64)
        image_paths.append(str(rel))
    return {
        "vlm_input_kind": "bev_labeled_layout",
        "system_path": "system.txt",
        "prompt_path": "prompt.txt",
        "image_paths": image_paths,
        "metadata": package.get("metadata", {}),
    }


def choose_candidates(row: dict[str, Any], route: str) -> list[int]:
    if route == ROUTE_E0:
        return [int(x) for x in row.get("pre_e0_canvas_candidate_ids") or []]
    return [int(x) for x in row.get("pre_e0_same_class_candidate_ids") or []]


def materialize_one(row: dict[str, Any], out_root: Path, dataset_name: str, overwrite: bool = False) -> dict[str, Any]:
    cid = int(row["case"])
    route = row["route"]
    out_dir = case_dir(out_root, dataset_name, cid)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return {"case": cid, "status": "skipped_existing", "route": route}

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = choose_candidates(row, route)
    status = "ok"
    error = ""
    input_payload: dict[str, Any]
    anchors: list[int] = []

    try:
        if not candidates:
            status = "no_candidates"
            input_payload = {
                "vlm_input_kind": "unavailable",
                "system_path": None,
                "prompt_path": None,
                "image_paths": [],
            }
        elif route == ROUTE_E0:
            input_payload = build_e0(row, out_dir, candidates)
        else:
            scene = load_scene_data(row["scan_id"], dataset_name)
            anchors = find_anchors(row, scene, candidates)
            if route == ROUTE_SPATIAL:
                input_payload = build_text_only(row, out_dir, scene, candidates, anchors, "spatial_only")
            elif route == ROUTE_3DPOS:
                input_payload = build_text_only(row, out_dir, scene, candidates, anchors, "3d_position_text")
            elif route == ROUTE_BEV:
                input_payload = build_bev(row, out_dir, scene, candidates, anchors)
                if not input_payload.get("image_paths"):
                    status = "bev_render_missing"
            else:
                status = "unknown_route"
                input_payload = {"vlm_input_kind": "unknown", "image_paths": []}
    except Exception as exc:
        status = "error"
        error = repr(exc)
        input_payload = {"vlm_input_kind": "error", "image_paths": []}

    manifest = {
        "case": cid,
        "dataset": row["dataset"],
        "dataset_name": dataset_name,
        "scan_id": row["scan_id"],
        "target_id": row["target_id"],
        "obj_name": row.get("obj_name"),
        "caption": row["caption"],
        "query_type": row["query_type"],
        "route": route,
        "route_reason": row["route_reason"],
        "candidate_coverage_status": row.get("candidate_coverage_status"),
        "candidate_ids": candidates,
        "anchor_ids": anchors,
        "status": status,
        "error": error,
        **input_payload,
    }
    write_json(manifest_path, manifest)
    return {
        "case": cid,
        "status": status,
        "route": route,
        "num_candidates": len(candidates),
        "num_images": len(manifest.get("image_paths") or []),
        "error": error,
    }


def dataset_paths(preprocess_root: Path) -> dict[str, Path]:
    return {
        "scanrefer": preprocess_root / "scanrefer_official_val_preprocessed.jsonl",
        "nr3d": preprocess_root / "nr3d_full_preprocessed.jsonl",
    }


def run_dataset(args: argparse.Namespace, dataset_name: str, path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if args.sample_per_route > 0:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(row["route"], []).append(row)
        rows = sorted(
            [row for route in sorted(buckets) for row in buckets[route][: args.sample_per_route]],
            key=lambda row: int(row["case"]),
        )
    if args.limit > 0:
        rows = rows[: args.limit]

    summary_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(materialize_one, row, args.out_root, dataset_name, args.overwrite)
            for row in rows
        ]
        for fut in as_completed(futures):
            summary_rows.append(fut.result())

    summary_rows.sort(key=lambda r: int(r["case"]))
    out_dir = args.out_root / dataset_name
    write_json(out_dir / "materialization_summary.json", {
        "dataset": dataset_name,
        "input_path": str(path),
        "num_rows": len(rows),
        "workers": args.workers,
        "status_counts": dict(Counter(r["status"] for r in summary_rows)),
        "route_counts": dict(Counter(r["route"] for r in summary_rows)),
    })
    with (out_dir / "materialization_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return json.loads((out_dir / "materialization_summary.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--datasets", default="scanrefer,nr3d")
    parser.add_argument("--workers", type=int, default=max(4, min(16, os.cpu_count() or 4)))
    parser.add_argument("--sample-per-route", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = dataset_paths(args.preprocess_root)
    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for dataset_name in selected:
        if dataset_name not in paths:
            raise ValueError(f"Unknown dataset {dataset_name!r}; expected one of {sorted(paths)}")
        summaries[dataset_name] = run_dataset(args, dataset_name, paths[dataset_name])
    write_json(args.out_root / "materialization_summary.json", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
