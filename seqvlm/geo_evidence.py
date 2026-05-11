import os
from pathlib import Path

import cv2
import numpy as np

from seqvlm.ablation import parse_geo_structured


SCANNET_ROOT = Path("/data/knuvi/bosung/scannet")
SCANNET_ORIGIN = Path("/data/knuvi/bosung/scannet_origin")
MASK3D_ROOT = Path("/data/knuvi/bosung/Mask3d/scannet200")

COLORS_BGR = [
    (40, 40, 230),    # A red
    (230, 90, 40),    # B blue-ish
    (40, 180, 80),    # C green
    (0, 190, 230),    # D yellow
    (180, 80, 200),
    (80, 200, 200),
]
ANCHOR_BGR = (0, 220, 255)


def build_geo_evidence_images(
    scene_id,
    candidate_indices,
    caption,
    obj_name,
    ins_labels,
    ins_locs,
    ins_scores,
    class_predictor,
    out_dir,
    input_format="geo_bbox_overlay",
    n_views=3,
    max_frames=80,
):
    """Build geo_ground_v2-style per-candidate visual evidence images.

    geo_ground_v2 presents top selected frames per candidate to the VLM, either
    as raw frames or bbox overlays with candidate letters. SeqVLM's predictor
    expects one image per candidate, so this function stitches the selected
    frames for each candidate into one evidence canvas while preserving the
    candidate-letter and overlay semantics.
    """
    if not candidate_indices:
        return [], {}

    scene = _load_scene(scene_id, max_frames=max_frames)
    if scene is None:
        return [], {"error": "scene_load_failed"}

    npz = np.load(MASK3D_ROOT / f"{scene_id}.npz", allow_pickle=True)
    ins_pcds = list(npz["ins_pcds"])

    parsed = parse_geo_structured(caption, obj_name)
    anchor_indices = _resolve_anchor_indices(
        parsed, ins_labels, ins_scores, class_predictor
    )
    anchor_idx = anchor_indices[0] if anchor_indices else None
    anchor_pts_raw = None
    if anchor_idx is not None and anchor_idx < len(ins_pcds):
        anchor_pts_raw = _aligned_to_raw(np.asarray(ins_pcds[anchor_idx])[:, :3], scene["axis_inv"])

    out_dir = Path(out_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    trace = {
        "view_source": "geo_frame_selection",
        "input_format": input_format,
        "n_views": int(n_views),
        "max_frames": int(max_frames),
        "geo_parse": parsed,
        "anchor_indices": [int(x) for x in anchor_indices],
        "candidates": [],
    }

    for local_i, inst_idx in enumerate(candidate_indices):
        if inst_idx >= len(ins_pcds):
            continue
        pts_aligned = np.asarray(ins_pcds[inst_idx])[:, :3]
        pts_raw = _aligned_to_raw(pts_aligned, scene["axis_inv"])
        selected = _select_candidate_frames(
            pts_raw=pts_raw,
            anchor_pts_raw=anchor_pts_raw,
            frames=scene["frames"],
            K=scene["K"],
            n_views=n_views,
        )
        if not selected:
            continue

        letter = chr(ord("A") + local_i)
        color = COLORS_BGR[local_i % len(COLORS_BGR)]
        tiles = []
        frame_trace = []
        for frame, target_proj, anchor_proj, score in selected:
            img = cv2.imread(str(frame["image_path"]))
            if img is None:
                continue
            if input_format == "geo_bbox_overlay":
                img = _draw_overlay(
                    img,
                    target_proj["xyxy"],
                    color,
                    f"{letter} bbox",
                    anchor_proj["xyxy"] if anchor_proj else None,
                )
            img = _add_header(img, f"Candidate {letter} | frame {frame['frame_id']}")
            tiles.append(_resize_keep(img, width=640))
            frame_trace.append({
                "frame_id": int(frame["frame_id"]),
                "score": float(score),
                "target_visible_points": int(target_proj["n_visible"]),
                "anchor_visible_points": int(anchor_proj["n_visible"]) if anchor_proj else 0,
            })

        if not tiles:
            continue
        canvas = _stack_vertical(tiles)
        path = out_dir / f"cand_{local_i:02d}_inst_{inst_idx}_{input_format}.jpg"
        cv2.imwrite(str(path), canvas)
        paths.append(str(path))
        trace["candidates"].append({
            "candidate_local_index": int(local_i),
            "instance_index": int(inst_idx),
            "letter": letter,
            "frames": frame_trace,
            "evidence_path": str(path),
        })

    return paths, trace


def _resolve_anchor_indices(parsed, ins_labels, ins_scores, class_predictor):
    anchor = parsed.get("anchor") or ""
    if not anchor:
        return []
    try:
        pred_cls = class_predictor(anchor, ins_labels)
    except Exception:
        return []
    candidates = [
        (i, float(score))
        for i, (label, score) in enumerate(zip(ins_labels, ins_scores))
        if label == pred_cls
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in candidates[:2]]


def _load_scene(scene_id, max_frames=80):
    scene_dir = SCANNET_ROOT / scene_id
    if not scene_dir.exists():
        return None

    K4 = np.loadtxt(scene_dir / "intrinsic" / "intrinsic_color.txt")
    K = K4[:3, :3].astype(np.float32)

    axis = np.eye(4, dtype=np.float32)
    txt = SCANNET_ORIGIN / scene_id / f"{scene_id}.txt"
    if txt.exists():
        for line in txt.read_text().splitlines():
            if line.startswith("axisAlignment"):
                axis = np.array([float(x) for x in line.split("=")[1].split()], dtype=np.float32).reshape(4, 4)
                break
    axis_inv = np.linalg.inv(axis).astype(np.float32)

    color_dir = scene_dir / "color"
    fids = sorted([int(p.stem) for p in color_dir.glob("*.jpg")])
    if len(fids) > max_frames:
        keep = np.linspace(0, len(fids) - 1, max_frames, dtype=int)
        fids = [fids[i] for i in keep]

    frames = []
    for fid in fids:
        pose_path = scene_dir / "pose" / f"{fid}.txt"
        image_path = scene_dir / "color" / f"{fid}.jpg"
        if not pose_path.exists() or not image_path.exists():
            continue
        try:
            pose = np.loadtxt(pose_path).astype(np.float32)
        except Exception:
            continue
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            continue
        frames.append({
            "frame_id": fid,
            "pose": pose,
            "image_path": image_path,
        })
    return {"K": K, "axis_inv": axis_inv, "frames": frames}


def _aligned_to_raw(points, axis_inv):
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    raw = (axis_inv @ np.concatenate([pts, ones], axis=1).T).T[:, :3]
    return raw.astype(np.float32)


def _select_candidate_frames(pts_raw, anchor_pts_raw, frames, K, n_views):
    scored = []
    for frame in frames:
        target = _project_points(pts_raw, frame["pose"], K)
        if target is None or target["n_visible"] < 20:
            continue
        anchor = None
        score = target["score"]
        if anchor_pts_raw is not None and len(anchor_pts_raw) > 0:
            anchor = _project_points(anchor_pts_raw, frame["pose"], K)
            if anchor is not None:
                score = 0.65 * target["score"] + 0.35 * anchor["score"]
        scored.append((frame, target, anchor, score))
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:n_views]


def _project_points(pts_raw, c2w, K):
    if pts_raw is None or len(pts_raw) == 0:
        return None
    pts = np.asarray(pts_raw, dtype=np.float32)
    if len(pts) > 5000:
        step = max(1, len(pts) // 5000)
        pts = pts[::step]
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1).T
    cam = (w2c @ pts_h).T[:, :3]
    front = cam[:, 2] > 0.05
    if not np.any(front):
        return None
    cam = cam[front]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    h, w = 480, 640
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    if not np.any(inside):
        return None
    uv_in = uv[inside]
    x1, y1 = uv_in.min(axis=0)
    x2, y2 = uv_in.max(axis=0)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area_score = min(1.0, (bw * bh) / (w * h * 0.20))
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    dist = np.linalg.norm(center - np.array([w / 2, h / 2], dtype=np.float32))
    center_score = max(0.0, 1.0 - dist / (w * 0.75))
    visible_frac = len(uv_in) / max(1, len(pts))
    score = 0.45 * min(1.0, visible_frac * 8.0) + 0.35 * center_score + 0.20 * area_score
    return {
        "xyxy": _clip_xyxy((x1, y1, x2, y2), h, w),
        "n_visible": int(len(uv_in)),
        "score": float(score),
    }


def _clip_xyxy(xyxy, h, w):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = int(np.clip(np.floor(min(x1, x2)), 0, w - 1))
    x2 = int(np.clip(np.ceil(max(x1, x2)), 0, w - 1))
    y1 = int(np.clip(np.floor(min(y1, y2)), 0, h - 1))
    y2 = int(np.clip(np.ceil(max(y1, y2)), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _draw_overlay(img, target_xyxy, color_bgr, label, anchor_xyxy=None):
    out = img.copy()
    if target_xyxy is not None:
        _draw_labeled_rect(out, target_xyxy, color_bgr, label)
    if anchor_xyxy is not None:
        _draw_labeled_rect(out, anchor_xyxy, ANCHOR_BGR, "Anchor")
    return out


def _draw_labeled_rect(img, xyxy, color, label):
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.65
    (tw, th), _ = cv2.getTextSize(label, font, fs, 2)
    ty = max(y1 - 8, th + 8)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
    txt = (0, 0, 0) if sum(color) > 420 else (255, 255, 255)
    cv2.putText(img, label, (x1 + 4, ty), font, fs, txt, 2, cv2.LINE_AA)


def _add_header(img, text):
    h, w = img.shape[:2]
    header = np.zeros((34, w, 3), dtype=np.uint8)
    header[:] = (28, 28, 28)
    cv2.putText(header, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack([header, img])


def _resize_keep(img, width=640):
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _stack_vertical(images):
    width = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        if img.shape[1] < width:
            pad = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)
    return np.vstack(padded)

