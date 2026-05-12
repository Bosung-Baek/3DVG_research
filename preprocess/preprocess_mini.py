"""
Mini preprocessing script for SeqVLM — adapts our ScanNet data layout.

Our data:
  /data/knuvi/bosung/scannet/{scene_id}/color/{frame_id}.jpg
  /data/knuvi/bosung/scannet/{scene_id}/depth/{frame_id}.png
  /data/knuvi/bosung/scannet/{scene_id}/pose/{frame_id}.txt
  /data/knuvi/bosung/scannet/{scene_id}/intrinsic/intrinsic_color.txt
  /data/knuvi/bosung/scannet_origin/{scene_id}/{scene_id}.txt  (axisAlignment)
  /data/knuvi/bosung/Mask3d/scannet200/{scene_id}.npz

Output:
  data/scanrefer_preprocessed/{scene_id}/{obj_id}/canvas.jpg

Usage:
  conda run -n sam3 python preprocess/preprocess_mini.py \
      --scenes scene0606_00 scene0221_00 scene0329_00
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SCANNET_DIR        = Path("/data/knuvi/bosung/scannet")
SCANNET_ORIGIN_DIR = Path("/data/knuvi/bosung/scannet_origin")
MASK3D_DIR         = Path("/data/knuvi/bosung/Mask3d/scannet200")
OUTPUT_DIR         = Path("/home/knuvi/bosung/SeqVLM/data/scanrefer_preprocessed")

VIS_THRESH = 0.25
CUT_BOUND  = 0
MAX_FRAMES = 20    # sample at most 20 frames per scene
TOP_K      = 5     # keep 5 best views per instance


# ── geometry ──────────────────────────────────────────────────────────────────

def load_axis_alignment(scene_id: str) -> np.ndarray:
    txt = SCANNET_ORIGIN_DIR / scene_id / f"{scene_id}.txt"
    with open(txt) as f:
        for line in f:
            if line.startswith("axisAlignment"):
                vals = [float(x) for x in line.split("=")[1].split()]
                return np.array(vals, dtype=np.float64).reshape(4, 4)
    return np.eye(4)


def compute_mapping(c2w, coords, depth, image_dim, intrinsic, vis_thresh, cut_bound):
    """Returns (N, 3) mapping array: [row, col, valid]."""
    H, W = image_dim
    mapping = np.zeros((3, coords.shape[0]), dtype=int)
    coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1).T  # (4,N)

    w2c = np.linalg.inv(c2w)
    p = w2c @ coords_h  # (4,N)
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

    mapping[0][inside] = pi[1][inside]  # row
    mapping[1][inside] = pi[0][inside]  # col
    mapping[2][inside] = 1
    return mapping.T  # (N, 3)


# ── per-scene processing ──────────────────────────────────────────────────────

def process_scene(scene_id: str) -> None:
    t0 = time.time()
    print(f"\n[{scene_id}] Processing …")

    scene_dir  = SCANNET_DIR / scene_id
    npz_path   = MASK3D_DIR / f"{scene_id}.npz"

    if not npz_path.exists():
        print(f"  SKIP: no npz at {npz_path}")
        return

    # Load Mask3D predictions
    raw       = np.load(str(npz_path), allow_pickle=True)
    ins_pcds  = list(raw["ins_pcds"])    # list of (P,6) XYZ+RGB
    N_inst    = len(ins_pcds)
    print(f"  {N_inst} instances loaded from npz")

    # axis alignment: axis-aligned → raw world
    axis_align = load_axis_alignment(scene_id)
    inv_align  = np.linalg.inv(axis_align).astype(np.float32)

    # intrinsic
    K4 = np.loadtxt(scene_dir / "intrinsic" / "intrinsic_color.txt")
    K  = K4[:3, :3].astype(np.float32)

    # list frames, sample evenly
    color_dir  = scene_dir / "color"
    depth_dir  = scene_dir / "depth"
    pose_dir   = scene_dir / "pose"
    frame_ids  = sorted(
        [p.stem for p in pose_dir.glob("*.txt")],
        key=lambda x: int(x)
    )
    step = max(1, len(frame_ids) // MAX_FRAMES)
    frame_ids = frame_ids[::step][:MAX_FRAMES]
    print(f"  Using {len(frame_ids)} frames (from {len(frame_ids)*step} total)")

    # prepare instance coords in raw-world space
    inst_pts_world = []
    for pcd in ins_pcds:
        pts = np.asarray(pcd, dtype=np.float32)[:, :3]
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=1)
        pts_raw = (inv_align @ pts_h.T).T[:, :3]
        inst_pts_world.append(pts_raw)

    # accumulate full-frame bbox overlays per instance
    views  = defaultdict(list)   # obj_id -> list of (frame_img, frame_id, area)

    for fid in frame_ids:
        color_path = color_dir / f"{fid}.jpg"
        depth_path = depth_dir / f"{fid}.png"
        pose_path  = pose_dir  / f"{fid}.txt"

        if not color_path.exists() or not pose_path.exists():
            continue

        img   = cv2.imread(str(color_path))
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
            link = compute_mapping(c2w, pts_raw, depth, (H_img, W_img), K, VIS_THRESH, CUT_BOUND)
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

    print(f"  Annotated {len(views)} instances")

    # for each instance: select top-K full frames, stitch into canvas
    for obj_id, view_list in views.items():
        if not view_list:
            continue

        import random as _random
        # random sample to match original SeqVLM paper (random.sample, not area-based)
        if len(view_list) <= TOP_K:
            selected = view_list
        else:
            selected = _random.sample(view_list, TOP_K)

        obj_out = OUTPUT_DIR / scene_id / str(obj_id)
        obj_out.mkdir(parents=True, exist_ok=True)

        # save individual annotated frames + build canvas
        imgs_for_canvas = []
        for idx, (frame, fid, _) in enumerate(selected):
            cv2.imwrite(str(obj_out / f"frame_{idx:02d}.jpg"), frame)
            imgs_for_canvas.append(frame)

        if not imgs_for_canvas:
            continue

        # stitch vertically
        max_w = max(i.shape[1] for i in imgs_for_canvas)
        canvas = np.zeros((sum(i.shape[0] for i in imgs_for_canvas), max_w, 3), np.uint8)
        y = 0
        for img_c in imgs_for_canvas:
            h, w = img_c.shape[:2]
            canvas[y:y+h, :w] = img_c
            y += h

        cv2.imwrite(str(obj_out / "canvas.jpg"), canvas)

    n_canvas = sum(1 for _ in OUTPUT_DIR.glob(f"{scene_id}/*/canvas.jpg"))
    print(f"  Saved {n_canvas} canvas.jpg files → {OUTPUT_DIR}/{scene_id}/")
    print(f"  Time: {time.time()-t0:.1f}s")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args()

    for scene_id in args.scenes:
        process_scene(scene_id)

    print("\nDone. Output at:", OUTPUT_DIR)
