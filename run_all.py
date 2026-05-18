"""
run_all.py — Full ablation pipeline for SeqVLM
================================================
1. Preprocess all missing scenes (tqdm + log file)
2. Run E0, E_V, E_VF sequentially (tqdm per-query + log file)

Usage (from /home/knuvi/bosung/SeqVLM):
  conda run -n tsdsr python run_all.py [--skip-preprocess] [--skip-eval]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── dual logging: tee stdout+stderr to log file ───────────────────────────────

class _Tee:
    """Write to both stream and file simultaneously."""
    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def fileno(self):
        return self.stream.fileno()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    logger = logging.getLogger("run_all")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT      = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "seqvlm"))

DATA_PATH      = REPO_ROOT / "data/scanrefer_250.json"
IMAGE_PATH     = REPO_ROOT / "data/scanrefer_preprocessed"
OUT_DIR        = REPO_ROOT / "experiments/full_ablation/outputs"
LOG_DIR        = REPO_ROOT / "experiments/full_ablation/logs"
GEO_EVID_DIR   = REPO_ROOT / "experiments/full_ablation/geo_evidence"

SCANNET_DIR        = Path("/data/knuvi/bosung/scannet")
SCANNET_ORIGIN_DIR = Path("/data/knuvi/bosung/scannet_origin")
MASK3D_DIR         = Path("/data/knuvi/bosung/Mask3d/scannet200")

VIS_THRESH = 0.25
CUT_BOUND  = 0
MAX_FRAMES = 10000
TOP_K      = 5


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def load_axis_alignment(scene_id: str):
    import numpy as np
    txt = SCANNET_ORIGIN_DIR / scene_id / f"{scene_id}.txt"
    with open(txt) as f:
        for line in f:
            if line.startswith("axisAlignment"):
                vals = [float(x) for x in line.split("=")[1].split()]
                return np.array(vals, dtype=np.float64).reshape(4, 4)
    return np.eye(4)


def compute_mapping(c2w, coords, depth, image_dim, intrinsic, vis_thresh, cut_bound):
    import numpy as np
    H, W = image_dim
    mapping = np.zeros((3, coords.shape[0]), dtype=int)
    coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1).T

    w2c = np.linalg.inv(c2w)
    p = w2c @ coords_h
    p[0] = p[0] * intrinsic[0, 0] / p[2] + intrinsic[0, 2]
    p[1] = p[1] * intrinsic[1, 1] / p[2] + intrinsic[1, 2]
    pi = np.round(p).astype(int)

    inside = (pi[0] >= cut_bound) & (pi[1] >= cut_bound) & \
             (pi[0] < W - cut_bound) & (pi[1] < H - cut_bound)

    if depth is not None:
        depth_at = depth[pi[1][inside], pi[0][inside]]
        occ = np.abs(depth_at - p[2][inside]) <= vis_thresh * depth_at
        inside[inside] = occ
    else:
        inside &= (p[2] > 0)

    mapping[0][inside] = pi[1][inside]
    mapping[1][inside] = pi[0][inside]
    mapping[2][inside] = 1
    return mapping.T


def process_scene(scene_id: str, pbar=None) -> str:
    """Process one scene. Returns status string."""
    import cv2
    import numpy as np
    import random as _rnd
    from collections import defaultdict

    t0 = time.time()
    scene_dir = SCANNET_DIR / scene_id
    npz_path  = MASK3D_DIR / f"{scene_id}.npz"

    if not npz_path.exists():
        return f"SKIP (no npz)"

    raw      = np.load(str(npz_path), allow_pickle=True)
    ins_pcds = list(raw["ins_pcds"])
    N_inst   = len(ins_pcds)

    axis_align = load_axis_alignment(scene_id)
    inv_align  = np.linalg.inv(axis_align).astype(np.float32)

    K4 = np.loadtxt(scene_dir / "intrinsic" / "intrinsic_color.txt")
    K  = K4[:3, :3].astype(np.float32)

    pose_dir  = scene_dir / "pose"
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"

    frame_ids = sorted([p.stem for p in pose_dir.glob("*.txt")], key=lambda x: int(x))
    if len(frame_ids) > MAX_FRAMES:
        frame_ids = frame_ids[:: len(frame_ids) // MAX_FRAMES]

    inst_pts_world = []
    for pcd in ins_pcds:
        pts  = np.asarray(pcd, dtype=np.float32)[:, :3]
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h   = np.concatenate([pts, ones], axis=1)
        pts_raw = (inv_align @ pts_h.T).T[:, :3]
        inst_pts_world.append(pts_raw)

    views = defaultdict(list)

    for fid in frame_ids:
        color_path = color_dir / f"{fid}.jpg"
        depth_path = depth_dir / f"{fid}.png"
        pose_path  = pose_dir  / f"{fid}.txt"

        if not color_path.exists() or not pose_path.exists():
            continue

        img = cv2.imread(str(color_path))
        if img is None:
            continue
        H_img, W_img = img.shape[:2]

        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            depth = cv2.resize(depth, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        else:
            depth = None

        c2w = np.loadtxt(str(pose_path)).astype(np.float32)
        if c2w.shape != (4, 4) or np.any(np.isinf(c2w)):
            continue

        for obj_id, pts_raw in enumerate(inst_pts_world):
            link  = compute_mapping(c2w, pts_raw, depth, (H_img, W_img), K, VIS_THRESH, CUT_BOUND)
            valid = link[link[:, 2] != 0]
            if len(valid) < 20:
                continue

            rows, cols = valid[:, 0], valid[:, 1]
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            h_c, w_c = r_max - r_min, c_max - c_min
            if h_c < 20 or w_c < 20:
                continue

            frame = img.copy()
            cv2.rectangle(frame, (c_min, r_min), (c_max, r_max), (0, 0, 255), 3)
            views[obj_id].append((frame, int(fid), h_c * w_c))

    for obj_id, view_list in views.items():
        if not view_list:
            continue
        selected = view_list if len(view_list) <= TOP_K else _rnd.sample(view_list, TOP_K)

        obj_out = IMAGE_PATH / scene_id / str(obj_id)
        obj_out.mkdir(parents=True, exist_ok=True)

        imgs_for_canvas = []
        for idx, (frame, fid, _) in enumerate(selected):
            cv2.imwrite(str(obj_out / f"frame_{idx:02d}.jpg"), frame)
            imgs_for_canvas.append(frame)

        if not imgs_for_canvas:
            continue

        max_w  = max(i.shape[1] for i in imgs_for_canvas)
        canvas = np.zeros((sum(i.shape[0] for i in imgs_for_canvas), max_w, 3), np.uint8)
        y = 0
        for img_c in imgs_for_canvas:
            h, w = img_c.shape[:2]
            canvas[y:y+h, :w] = img_c
            y += h

        cv2.imwrite(str(obj_out / "canvas.jpg"), canvas)

    n_canvas = sum(1 for _ in (IMAGE_PATH / scene_id).rglob("canvas.jpg"))
    elapsed  = time.time() - t0
    return f"OK  {N_inst} instances → {n_canvas} canvas files  ({elapsed:.1f}s)"


def run_preprocessing(scenes: list[str], logger: logging.Logger) -> None:
    from tqdm import tqdm

    already = set(p.name for p in IMAGE_PATH.iterdir() if p.is_dir()) if IMAGE_PATH.exists() else set()
    todo    = [s for s in scenes if s not in already]

    logger.info(f"Preprocessing: {len(todo)} scenes to process ({len(scenes)-len(todo)} already done)")

    if not todo:
        logger.info("All scenes already preprocessed. Skipping.")
        return

    pbar = tqdm(todo, desc="Preprocessing", unit="scene", dynamic_ncols=True, file=sys.stdout)
    for scene_id in pbar:
        pbar.set_postfix(scene=scene_id)
        status = process_scene(scene_id, pbar)
        logger.info(f"  [{scene_id}] {status}")

    logger.info("Preprocessing complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — evaluation
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    {
        "name":         "full_E0_baseline",
        "ablation_axis":"baseline",
        "input_format": "seqvlm_canvas",
        "view_source":  "seqvlm_canvas",
    },
    {
        "name":         "full_E_V_viewpoint",
        "ablation_axis":"viewpoint",
        "input_format": "geo_raw_frame",
        "view_source":  "geo_frame_selection",
    },
    {
        "name":         "full_E_VF_system",
        "ablation_axis":"viewpoint",
        "input_format": "geo_bbox_overlay",
        "view_source":  "geo_frame_selection",
    },
]


def run_eval_experiment(exp: dict, eval_data: list, logger: logging.Logger) -> None:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from seqvlm.utils import load_seg_inst, load_pc, calc_iou, encode_image_to_base64, create_logger
    from seqvlm.adaptive_predictor import AdpativePredictor

    name        = exp["name"]
    output_path = OUT_DIR / f"{name}.jsonl"
    geo_dir     = GEO_EVID_DIR / name

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {name}")
    logger.info(f"  ablation_axis={exp['ablation_axis']}, input_format={exp['input_format']}, view_source={exp['view_source']}")
    logger.info(f"  output → {output_path}")
    logger.info(f"{'='*60}")

    # load already-completed cases for resume support
    done_cases: set[int] = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path) as f:
            for line in f:
                try:
                    done_cases.add(json.loads(line)["case"])
                except Exception:
                    pass
        if done_cases:
            logger.info(f"  Resuming: {len(done_cases)} cases already done, skipping them")
    else:
        output_path.write_text("")

    vlm_configs = {
        "image_path":      str(IMAGE_PATH),
        "vlm_model":       "local-qwen",
        "max_retry":       3,
        "max_batch_size":  4,
        "max_vlm_props":   40,
        "ablation_axis":   exp["ablation_axis"],
        "input_format":    exp["input_format"],
        "view_source":     exp["view_source"],
        "force_program_first": False,
        "use_spatial_filter":  False,
        "geo_n_views":     3,
        "geo_max_frames":  80,
        "geo_evidence_dir": str(geo_dir),
    }

    predictor = AdpativePredictor(**vlm_configs)

    label_map_file = REPO_ROOT / "data/scannetv2-labels.combined.tsv"
    labels_pd = pd.read_csv(str(label_map_file), sep="\t", header=0)

    correct_25 = correct_50 = 0
    unique_25  = unique_50  = 0
    total = vlm_total = unique_total = except_total = 0
    eps   = 1e-6

    pbar = tqdm(
        enumerate(eval_data),
        total=len(eval_data),
        desc=name,
        unit="query",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for i, task in pbar:
        if i in done_cases:
            continue

        scene_id  = task["scan_id"]
        obj_id    = task["target_id"]
        caption   = task["caption"]
        prog_str  = task["program"]
        obj_name  = task["obj_name"]

        obj_ids, obj_labels, obj_locs = load_pc(scene_id)
        mapped_class_ids = []
        for lbl in obj_labels:
            ids = labels_pd[labels_pd["raw_category"] == lbl]["nyu40id"]
            mapped_class_ids.append(int(ids.iloc[0]) if len(ids) > 0 else 0)

        index        = obj_ids.index(int(obj_id))
        target_box   = obj_locs[index]
        target_cls   = mapped_class_ids[index]

        total  += 1
        unique  = (np.array(mapped_class_ids) == target_cls).sum() == 1
        if unique:
            unique_total += 1

        pred_box, use_vlm = predictor.execute(scene_id, obj_name, caption, prog_str)
        iou = 0.0
        if use_vlm:
            vlm_total += 1

        if pred_box is None:
            except_total += 1
        else:
            iou = calc_iou(pred_box, target_box)
            if iou >= 0.25:
                correct_25 += 1
                if unique:
                    unique_25 += 1
            if iou >= 0.5:
                correct_50 += 1
                if unique:
                    unique_50 += 1

        acc25 = correct_25 / total
        acc50 = correct_50 / total
        pbar.set_postfix(Acc25=f"{acc25:.3f}", Acc50=f"{acc50:.3f}")

        record = {
            "case": i, "scene_id": scene_id, "obj_id": int(obj_id),
            "obj_name": obj_name, "caption": caption,
            "ablation_axis": exp["ablation_axis"],
            "input_format":  exp["input_format"],
            "view_source":   exp["view_source"],
            "use_vlm":   bool(use_vlm),
            "pred_box":  pred_box.tolist() if pred_box is not None else None,
            "target_box":target_box.tolist(),
            "iou":    float(iou),
            "acc25":  bool(iou >= 0.25),
            "acc50":  bool(iou >= 0.5),
            "unique": bool(unique),
            "trace":  predictor.last_trace,
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if (i + 1) % 10 == 0:
            mult_25 = (correct_25 - unique_25) / (total - unique_total + eps)
            mult_50 = (correct_50 - unique_50) / (total - unique_total + eps)
            logger.info(
                f"[{i+1}/{len(eval_data)}] "
                f"Ov@25={acc25:.3f}  Ov@50={acc50:.3f}  "
                f"Uniq@25={unique_25/(unique_total+eps):.3f}  "
                f"Mult@25={mult_25:.3f}  "
                f"Except={except_total}/{total}"
            )

    # final summary
    mult_25 = (correct_25 - unique_25) / (total - unique_total + eps)
    mult_50 = (correct_50 - unique_50) / (total - unique_total + eps)
    logger.info(f"\n{'─'*60}")
    logger.info(f"FINAL RESULTS — {name}")
    logger.info(f"  Overall@25  : {correct_25/total:.3f}  ({correct_25}/{total})")
    logger.info(f"  Overall@50  : {correct_50/total:.3f}  ({correct_50}/{total})")
    logger.info(f"  Unique@25   : {unique_25/(unique_total+eps):.3f}  ({unique_25}/{unique_total})")
    logger.info(f"  Unique@50   : {unique_50/(unique_total+eps):.3f}  ({unique_50}/{unique_total})")
    logger.info(f"  Multiple@25 : {mult_25:.3f}  ({correct_25-unique_25}/{total-unique_total})")
    logger.info(f"  Multiple@50 : {mult_50:.3f}  ({correct_50-unique_50}/{total-unique_total})")
    logger.info(f"  VLM usage   : {vlm_total}/{total}")
    logger.info(f"  Exceptions  : {except_total}/{total}")
    logger.info(f"{'─'*60}")


def run_evaluation(eval_data: list, exps: list, logger: logging.Logger) -> None:
    for exp in exps:
        run_eval_experiment(exp, eval_data, logger)
    logger.info("\nAll experiments complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full SeqVLM ablation pipeline")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing step (use existing canvas.jpg files)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation step (only run preprocessing)")
    parser.add_argument("--exp", nargs="+", choices=["E0", "E_V", "E_VF"],
                        default=["E0", "E_V", "E_VF"],
                        help="Which experiments to run (default: all)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_all_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger   = setup_logging(log_path)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Repo root: {REPO_ROOT}")

    with open(DATA_PATH) as f:
        eval_data = json.load(f)

    scenes_needed = sorted(set(t["scan_id"] for t in eval_data))
    logger.info(f"Data: {len(eval_data)} queries, {len(scenes_needed)} unique scenes")

    # ── step 1: preprocess ──
    if not args.skip_preprocess:
        run_preprocessing(scenes_needed, logger)

    # ── step 2: evaluate ──
    if not args.skip_eval:
        exp_map = {"E0": EXPERIMENTS[0], "E_V": EXPERIMENTS[1], "E_VF": EXPERIMENTS[2]}
        selected_exps = [exp_map[k] for k in args.exp]
        run_evaluation(eval_data, selected_exps, logger)

    logger.info(f"\nDone. Log saved to: {log_path}")


if __name__ == "__main__":
    main()
