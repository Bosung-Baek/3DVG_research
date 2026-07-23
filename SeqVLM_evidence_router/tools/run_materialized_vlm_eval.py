#!/usr/bin/env python3
"""Run VLM evaluation from materialized full-dataset input packages.

The runner is resumable and writes rolling metrics while it runs:

  results.jsonl
  rolling_summary.json
  run.log

Modes:
  final_router: use each materialized manifest's routed input.
  e0_only: force E0 canvas tournament for every query that has E0 canvas
           candidates available.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
import time
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

from input_formats.baseline_e0 import IMAGE_ID_INVALID_PROMPT, WRONG_FORMAT_PROMPT  # noqa: E402
from input_formats.baseline_e0 import SYSTEM_PROMPT as E0_SYSTEM_PROMPT  # noqa: E402
from input_formats.baseline_e0 import USER_PROMPT as E0_USER_PROMPT  # noqa: E402
from input_formats.base import parse_selected_key  # noqa: E402
from seqvlm.api import invoke_api  # noqa: E402
from seqvlm.utils import calc_iou, load_pc, load_seg_inst  # noqa: E402


DEFAULT_INPUT_ROOT = Path("/data/knuvi/bosung/evidence_router_vlm_inputs")
DEFAULT_OUT_ROOT = Path("/data/knuvi/bosung/evidence_router_full_runs")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANVAS_ROOTS = {
    "scanrefer": SEQVLM / "data" / "scanrefer_preprocessed",
    "nr3d": SEQVLM / "data" / "preprocessed_nr3d",
}
ROUTE_DISPLAY_NAMES = {
    "E0": "E0 RGB canvas",
    "seeground_ablation_spatial_only": "spatial-only text",
    "seeground_ablation_3dpos_only": "3D position text",
    "bev_raw_labeled": "BEV labeled layout",
}

WRITE_LOCK = threading.Lock()
GT_LOCK = threading.Lock()
SEG_LOCK = threading.Lock()
GT_CACHE: dict[str, tuple[list[int], list[str], list[np.ndarray]]] = {}
SEG_CACHE: dict[str, tuple[list[str], list[np.ndarray], list[float], np.ndarray]] = {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            done.add(int(row["case"]))
    return done


def image_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def get_gt(scene_id: str):
    with GT_LOCK:
        if scene_id in GT_CACHE:
            return GT_CACHE[scene_id]
    gt = load_pc(scene_id)
    with GT_LOCK:
        GT_CACHE[scene_id] = gt
    return gt


def get_seg(scene_id: str):
    with SEG_LOCK:
        if scene_id in SEG_CACHE:
            return SEG_CACHE[scene_id]
    seg = load_seg_inst(scene_id)
    with SEG_LOCK:
        SEG_CACHE[scene_id] = seg
    return seg


def gt_box_for(scene_id: str, target_id: Any) -> np.ndarray | None:
    obj_ids, _labels, locs = get_gt(scene_id)
    tid = int(target_id)
    if tid not in obj_ids:
        return None
    return np.asarray(locs[obj_ids.index(tid)], dtype=float)


def calc_pred_iou(scene_id: str, pred_instance: int | None, target_id: Any, dataset: str) -> float:
    if pred_instance is None:
        return 0.0
    gt_box = gt_box_for(scene_id, target_id)
    if gt_box is None:
        return 0.0
    if dataset == "nr3d":
        obj_ids, _labels, locs = get_gt(scene_id)
        pred_instance = int(pred_instance)
        if pred_instance not in obj_ids:
            return 0.0
        return float(calc_iou(np.asarray(locs[obj_ids.index(pred_instance)], dtype=float), gt_box))
    _labels, locs, _scores, _center = get_seg(scene_id)
    if pred_instance < 0 or pred_instance >= len(locs):
        return 0.0
    return float(calc_iou(np.asarray(locs[pred_instance], dtype=float), gt_box))


def rel_or_abs(case_dir: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else case_dir / p


def make_image_part(path: Path) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_to_b64(path)}",
            "detail": "high",
        },
    }


def invoke_with_retry(model: str, messages: list[dict[str, Any]], max_retry: int) -> tuple[str, str, int, int, float]:
    error = ""
    prompt_tokens = 0
    completion_tokens = 0
    t0 = time.time()
    for attempt in range(max_retry):
        try:
            out = invoke_api(model, messages)
            prompt_tokens += int(out.get("prompt_tokens") or 0)
            completion_tokens += int(out.get("completion_tokens") or 0)
            return out.get("answer", ""), "", prompt_tokens, completion_tokens, time.time() - t0
        except Exception as exc:
            error = repr(exc)
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
    return "", error, prompt_tokens, completion_tokens, time.time() - t0


def parse_e0_answer(raw: str, n_images: int) -> int:
    try:
        obj = json.loads(raw)
        image_id = obj.get("image_id", -1)
        if isinstance(image_id, int) and -1 <= image_id < n_images:
            return image_id
    except Exception:
        pass
    m = re.search(r'"image_id"\s*:\s*(-?\d+)', raw)
    if m:
        val = int(m.group(1))
        if -1 <= val < n_images:
            return val
    return -2


def select_e0_tournament(manifest: dict[str, Any], case_dir: Path, model: str, max_batch: int, max_retry: int) -> dict[str, Any]:
    candidate_ids = [int(x) for x in manifest.get("candidate_ids") or []]
    image_paths = [rel_or_abs(case_dir, rel) for rel in manifest.get("image_paths") or []]
    pairs = [(cid, path) for cid, path in zip(candidate_ids, image_paths) if path.exists()]
    if not pairs:
        return {"pred_instance": None, "raw_answer": "", "n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "api_seconds": 0.0, "error": "no_e0_images"}

    remain = pairs
    raw_answers = []
    n_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    api_seconds = 0.0
    errors = []
    while len(remain) > 1:
        new_remain = []
        for start in range(0, len(remain), max_batch):
            group = remain[start:start + max_batch]
            user_text = E0_USER_PROMPT.format(query=manifest["caption"], n_images=len(group))
            messages = [
                {"role": "system", "content": E0_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[make_image_part(path) for _cid, path in group],
                    ],
                },
            ]
            answer = ""
            local_idx = -2
            for retry in range(max_retry):
                answer, error, pt, ct, sec = invoke_with_retry(model, messages, 1)
                prompt_tokens += pt
                completion_tokens += ct
                api_seconds += sec
                n_calls += 1
                raw_answers.append(answer)
                if error:
                    errors.append(error)
                local_idx = parse_e0_answer(answer, len(group))
                if -1 <= local_idx < len(group):
                    break
                guide = IMAGE_ID_INVALID_PROMPT.format(image_id=local_idx) if local_idx != -2 else WRONG_FORMAT_PROMPT
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content": guide})
            if 0 <= local_idx < len(group):
                new_remain.append(group[local_idx])
        remain = new_remain
        if not remain:
            break
    pred = int(remain[0][0]) if remain else None
    return {
        "pred_instance": pred,
        "raw_answer": raw_answers[-1] if raw_answers else "",
        "raw_answers": raw_answers,
        "n_calls": n_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "api_seconds": round(api_seconds, 4),
        "error": "; ".join(errors[-3:]),
    }


def parse_key_answer(raw: str, n_candidates: int) -> int | None:
    letters = [LETTERS[i] if i < len(LETTERS) else str(i) for i in range(n_candidates)]
    raw = raw or ""
    cleaned = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.I | re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        obj = json.loads(cleaned)
        letter = str(obj.get("letter") or obj.get("selected_key") or "").strip().upper()
        if letter in letters:
            return letters.index(letter)
    except Exception:
        pass
    m = re.search(r'"(?:letter|selected_key)"\s*:\s*"([A-Z0-9])"', raw, re.I)
    if m and m.group(1).upper() in letters:
        return letters.index(m.group(1).upper())
    key = parse_selected_key(raw, set(letters))
    if key and key in letters:
        return letters.index(key)
    return None


def select_single_prompt(manifest: dict[str, Any], case_dir: Path, model: str, max_retry: int) -> dict[str, Any]:
    candidate_ids = [int(x) for x in manifest.get("candidate_ids") or []]
    prompt_path = manifest.get("prompt_path")
    system_path = manifest.get("system_path")
    if not candidate_ids or not prompt_path:
        return {"pred_instance": None, "raw_answer": "", "n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "api_seconds": 0.0, "error": "missing_prompt_or_candidates"}
    system = (case_dir / system_path).read_text() if system_path else ""
    prompt = (case_dir / prompt_path).read_text()
    content = [{"type": "text", "text": prompt}]
    for rel in manifest.get("image_paths") or []:
        path = rel_or_abs(case_dir, rel)
        if path.exists():
            content.append(make_image_part(path))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    answer, error, pt, ct, sec = invoke_with_retry(model, messages, max_retry)
    local_idx = parse_key_answer(answer, len(candidate_ids))
    pred = candidate_ids[local_idx] if local_idx is not None else None
    return {
        "pred_instance": pred,
        "raw_answer": answer,
        "selected_local_index": local_idx,
        "n_calls": 1 if answer or error else 0,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "api_seconds": round(sec, 4),
        "error": error,
    }


def force_e0_manifest(manifest: dict[str, Any], case_dir: Path, dataset: str) -> dict[str, Any]:
    scan_id = manifest["scan_id"]
    root = CANVAS_ROOTS[dataset]
    candidate_ids = []
    image_paths = []
    # Prefer the route-plan's E0 coverage candidates if present. Otherwise fall
    # back to the routed candidate IDs and keep only candidates with canvas.
    source_ids = manifest.get("candidate_ids") or []
    for cid in source_ids:
        path = root / scan_id / str(cid) / "canvas.jpg"
        if path.exists():
            candidate_ids.append(int(cid))
            image_paths.append(str(path.resolve()))
    return {
        **manifest,
        "route": "E0",
        "route_reason": "forced_e0_only",
        "vlm_input_kind": "e0_rgb_canvas_tournament",
        "candidate_ids": candidate_ids,
        "image_paths": image_paths,
    }


def run_case(manifest_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = manifest_path.parent
    manifest = read_json(manifest_path)
    if args.mode == "e0_only":
        manifest = force_e0_manifest(manifest, case_dir, args.dataset)

    rec = {
        "case": int(manifest["case"]),
        "dataset": manifest.get("dataset"),
        "dataset_name": args.dataset,
        "scan_id": manifest["scan_id"],
        "target_id": manifest["target_id"],
        "obj_name": manifest.get("obj_name"),
        "caption": manifest["caption"],
        "query_type": manifest.get("query_type"),
        "route": manifest.get("route"),
        "route_reason": manifest.get("route_reason"),
        "vlm_input_kind": manifest.get("vlm_input_kind"),
        "candidate_ids": manifest.get("candidate_ids") or [],
        "candidate_coverage_status": manifest.get("candidate_coverage_status"),
        "status": "pending",
    }

    if not rec["candidate_ids"]:
        pred = None
        sel = {"raw_answer": "", "n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "api_seconds": 0.0, "error": "no_candidates"}
    elif manifest.get("vlm_input_kind") == "e0_rgb_canvas_tournament":
        sel = select_e0_tournament(manifest, case_dir, args.model, args.max_batch, args.max_retry)
        pred = sel["pred_instance"]
    else:
        sel = select_single_prompt(manifest, case_dir, args.model, args.max_retry)
        pred = sel["pred_instance"]

    iou = calc_pred_iou(rec["scan_id"], pred, rec["target_id"], args.dataset)
    rec.update({
        "pred_instance": pred,
        "iou": round(float(iou), 6),
        "acc25": bool(iou >= 0.25),
        "acc50": bool(iou >= 0.50),
        "status": "ok" if not sel.get("error") else "error" if sel.get("error") not in {"no_candidates", "no_e0_images", "missing_prompt_or_candidates"} else sel.get("error"),
        **sel,
    })
    return rec


def summarize_results(rows: list[dict[str, Any]], total_expected: int, started_at: float) -> dict[str, Any]:
    n = len(rows)
    acc25 = sum(bool(r.get("acc25")) for r in rows)
    acc50 = sum(bool(r.get("acc50")) for r in rows)
    miou = sum(float(r.get("iou") or 0.0) for r in rows)
    by_route = {}
    for route, group_count in Counter(r.get("route") for r in rows).items():
        group = [r for r in rows if r.get("route") == route]
        by_route[str(route)] = {
            "n": len(group),
            "acc25": sum(bool(r.get("acc25")) for r in group) / max(len(group), 1),
            "acc50": sum(bool(r.get("acc50")) for r in group) / max(len(group), 1),
            "miou": sum(float(r.get("iou") or 0.0) for r in group) / max(len(group), 1),
        }
    display_routes = [ROUTE_DISPLAY_NAMES.get(str(r.get("route")), str(r.get("route"))) for r in rows]
    by_route_display = {}
    for display_name in Counter(display_routes):
        group = [
            r for r in rows
            if ROUTE_DISPLAY_NAMES.get(str(r.get("route")), str(r.get("route"))) == display_name
        ]
        by_route_display[display_name] = {
            "n": len(group),
            "acc25": sum(bool(r.get("acc25")) for r in group) / max(len(group), 1),
            "acc50": sum(bool(r.get("acc50")) for r in group) / max(len(group), 1),
            "miou": sum(float(r.get("iou") or 0.0) for r in group) / max(len(group), 1),
        }
    return {
        "processed": n,
        "total_expected": total_expected,
        "remaining": max(total_expected - n, 0),
        "acc25": acc25 / max(n, 1),
        "acc50": acc50 / max(n, 1),
        "miou": miou / max(n, 1),
        "status_counts": dict(Counter(r.get("status") for r in rows)),
        "route_counts": dict(Counter(r.get("route") for r in rows)),
        "route_display_counts": dict(Counter(display_routes)),
        "by_route": by_route,
        "by_route_display": by_route_display,
        "n_api_calls": sum(int(r.get("n_calls") or 0) for r in rows),
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
        "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }


def load_existing_rows(out_path: Path) -> list[dict[str, Any]]:
    if not out_path.exists():
        return []
    return [json.loads(line) for line in out_path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scanrefer", "nr3d"], required=True)
    parser.add_argument("--mode", choices=["final_router", "e0_only"], default="final_router")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model", default="openrouter-qwen")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-end", type=int, default=0)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.input_root / args.dataset
    manifests = sorted(dataset_dir.glob("case_*/manifest.json"), key=lambda p: int(p.parent.name.split("_")[-1]))
    if args.case_start:
        manifests = [p for p in manifests if int(p.parent.name.split("_")[-1]) >= args.case_start]
    if args.case_end:
        manifests = [p for p in manifests if int(p.parent.name.split("_")[-1]) < args.case_end]
    if args.limit:
        manifests = manifests[: args.limit]

    out_dir = args.out_root / args.dataset / args.mode / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    log_path = out_dir / "run.log"
    summary_path = out_dir / "rolling_summary.json"
    if args.overwrite and out_path.exists():
        out_path.unlink()

    done = load_done(out_path)
    todo = [p for p in manifests if int(p.parent.name.split("_")[-1]) not in done]
    existing_rows = load_existing_rows(out_path)
    all_rows = list(existing_rows)
    started_at = time.time()

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"dataset={args.dataset} mode={args.mode} model={args.model} total={len(manifests)} cached={len(done)} todo={len(todo)} workers={args.workers}")
    write_json(summary_path, summarize_results(all_rows, len(manifests), started_at))

    completed_since_log = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_case, path, args): path for path in todo}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                case = int(path.parent.name.split("_")[-1])
                rec = {"case": case, "dataset_name": args.dataset, "status": "exception", "error": repr(exc), "acc25": False, "acc50": False, "iou": 0.0}
            with WRITE_LOCK:
                append_jsonl(out_path, rec)
                all_rows.append(rec)
                completed_since_log += 1
                if completed_since_log >= args.progress_every or len(all_rows) == len(manifests):
                    summary = summarize_results(all_rows, len(manifests), started_at)
                    write_json(summary_path, summary)
                    log(
                        f"processed={summary['processed']}/{summary['total_expected']} "
                        f"acc25={summary['acc25']:.4f} acc50={summary['acc50']:.4f} "
                        f"miou={summary['miou']:.4f} calls={summary['n_api_calls']}"
                    )
                    completed_since_log = 0

    final_summary = summarize_results(all_rows, len(manifests), started_at)
    write_json(summary_path, final_summary)
    write_json(out_dir / "summary.json", final_summary)
    log(f"done acc25={final_summary['acc25']:.4f} acc50={final_summary['acc50']:.4f} miou={final_summary['miou']:.4f}")


if __name__ == "__main__":
    main()
