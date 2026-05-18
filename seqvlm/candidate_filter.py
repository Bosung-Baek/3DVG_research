"""
Geometric candidate filters for geo_evidence ablation.
All filter functions take (candidates, record, ins_locs, scene_info) and return
a filtered/reranked candidate list. Pass-through when len(candidates) <= PASS_THRESH.
"""
import json
import numpy as np
from pathlib import Path

PASS_THRESH = 10   # don't filter if already <= this many candidates
SCANNET_DIR = Path('/data/knuvi/bosung/scannet')
MASK3D_DIR  = Path('/data/knuvi/bosung/Mask3d/scannet200')

# ──────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────

def _load_ins_locs(scene_id: str) -> np.ndarray:
    """Compute ins_locs [cx,cy,cz,sx,sy,sz] from Mask3D NPZ ins_pcds."""
    npz = np.load(MASK3D_DIR / f'{scene_id}.npz', allow_pickle=True)
    ins_pcds = npz['ins_pcds']
    locs = []
    for pcd in ins_pcds:
        p = np.asarray(pcd, np.float32)
        if p.shape[0] == 0:
            locs.append(np.zeros(6, np.float32))
            continue
        mn, mx = p[:, :3].min(0), p[:, :3].max(0)
        locs.append(np.concatenate([(mn + mx) / 2, mx - mn]))
    return np.array(locs, dtype=np.float32)


def _load_pose(scene_id: str, frame_id: int) -> np.ndarray:
    """Load 4x4 camera-to-world pose matrix."""
    p = SCANNET_DIR / scene_id / 'pose' / f'{frame_id}.txt'
    return np.loadtxt(str(p)).astype(np.float64)


def _load_axis(scene_id: str) -> np.ndarray:
    """Load 4x4 axisAlignment matrix."""
    txt = Path('/data/knuvi/bosung/scannet_origin') / scene_id / f'{scene_id}.txt'
    M = np.eye(4, dtype=np.float64)
    if txt.exists():
        for line in txt.read_text().splitlines():
            if line.startswith('axisAlignment'):
                M = np.array([float(x) for x in line.split('=')[1].split()],
                              dtype=np.float64).reshape(4, 4)
    return M


def _center(ins_locs: np.ndarray, idx: int) -> np.ndarray:
    return ins_locs[idx, :3].astype(np.float64)


def _valid_cands(candidates, ins_locs):
    """Return candidates whose instance_index is in range."""
    n = len(ins_locs)
    return [c for c in candidates if 0 <= c.get('instance_index', -1) < n]


def _fallback(candidates, k=PASS_THRESH):
    """Return first k candidates as fallback."""
    return candidates[:k]


# ──────────────────────────────────────────────────────────────
# Filter: TopK (score-ranked cutoff)
# ──────────────────────────────────────────────────────────────

def filter_topk(candidates, record, ins_locs, k=PASS_THRESH):
    """Keep first k candidates (already score-ranked by geo_evidence)."""
    if len(candidates) <= k:
        return candidates, 'pass_through'
    return candidates[:k], 'topk'


# ──────────────────────────────────────────────────────────────
# Filter: Vertical (HIGHER / LOWER / UNDER / BELOW / ABOVE)
# ──────────────────────────────────────────────────────────────

def filter_vertical(candidates, record, ins_locs):
    """Filter by Z-axis relation relative to anchor centroid."""
    ge = (record.get('trace') or {}).get('geo_evidence') or {}
    gp = ge.get('geo_parse') or {}
    rel = (gp.get('relation') or '').upper()
    anchor_indices = ge.get('anchor_indices') or []

    if rel not in ('HIGHER', 'LOWER', 'UNDER', 'BELOW', 'ABOVE') or not anchor_indices:
        return candidates, 'skip'

    if len(candidates) <= PASS_THRESH:
        return candidates, 'pass_through'

    anchor_zs = [ins_locs[ai, 2] for ai in anchor_indices if 0 <= ai < len(ins_locs)]
    if not anchor_zs:
        return _fallback(candidates), 'fallback_no_anchor'
    anchor_z = float(np.mean(anchor_zs))

    if rel in ('HIGHER', 'ABOVE'):
        keep = [c for c in _valid_cands(candidates, ins_locs)
                if ins_locs[c['instance_index'], 2] > anchor_z]
    else:
        keep = [c for c in _valid_cands(candidates, ins_locs)
                if ins_locs[c['instance_index'], 2] < anchor_z]

    if not keep:
        return _fallback(candidates), 'fallback_empty'
    return keep, 'vertical'


# ──────────────────────────────────────────────────────────────
# Filter: Distance (CLOSEST / NEAR)
# ──────────────────────────────────────────────────────────────

def filter_distance(candidates, record, ins_locs, top_k=3):
    """Keep top_k candidates nearest to anchor centroid (euclidean in aligned space)."""
    ge = (record.get('trace') or {}).get('geo_evidence') or {}
    gp = ge.get('geo_parse') or {}
    rel = (gp.get('relation') or '').upper()
    anchor_indices = ge.get('anchor_indices') or []

    if rel not in ('CLOSEST', 'NEAR', 'BESIDE', 'NEXT') or not anchor_indices:
        return candidates, 'skip'

    if len(candidates) <= PASS_THRESH:
        return candidates, 'pass_through'

    anchor_centers = [ins_locs[ai, :3] for ai in anchor_indices if 0 <= ai < len(ins_locs)]
    if not anchor_centers:
        return _fallback(candidates), 'fallback_no_anchor'
    anchor_mean = np.mean(anchor_centers, axis=0)

    valid = _valid_cands(candidates, ins_locs)
    dists = [np.linalg.norm(ins_locs[c['instance_index'], :3] - anchor_mean) for c in valid]
    order = np.argsort(dists)
    keep = [valid[i] for i in order[:top_k]]

    if not keep:
        return _fallback(candidates), 'fallback_empty'
    return keep, 'distance'


# ──────────────────────────────────────────────────────────────
# Filter: Direction (LEFT / RIGHT / FRONT / BEHIND)
# ──────────────────────────────────────────────────────────────

def filter_direction(candidates, record, ins_locs):
    """Viewpoint-aware half-plane filter for directional relations."""
    ge = (record.get('trace') or {}).get('geo_evidence') or {}
    gp = ge.get('geo_parse') or {}
    rel = (gp.get('relation') or '').upper()
    anchor_indices = ge.get('anchor_indices') or []

    if rel not in ('LEFT', 'RIGHT', 'FRONT', 'BEHIND') or not anchor_indices:
        return candidates, 'skip'

    if len(candidates) <= PASS_THRESH:
        return candidates, 'pass_through'

    scene_id = record['scene_id']
    M_axis = _load_axis(scene_id)

    best_frame_id = None
    for c in candidates:
        frames = c.get('frames') or []
        if frames:
            best_frame_id = frames[0]['frame_id']
            break
    if best_frame_id is None:
        return _fallback(candidates), 'fallback_no_frame'

    try:
        pose_c2w = _load_pose(scene_id, best_frame_id)
    except Exception:
        return _fallback(candidates), 'fallback_pose_load'

    anchor_centers_aln = np.array([ins_locs[ai, :3] for ai in anchor_indices
                                    if 0 <= ai < len(ins_locs)], dtype=np.float64)
    if len(anchor_centers_aln) == 0:
        return _fallback(candidates), 'fallback_no_anchor'

    Minv = np.linalg.inv(M_axis)

    def to_cam(center_aln):
        h = np.append(center_aln, 1.0)
        raw = Minv @ h
        cam = np.linalg.inv(pose_c2w) @ raw
        return cam[:3]

    anchor_cam = np.mean([to_cam(c) for c in anchor_centers_aln], axis=0)

    def signed_offset(cand_idx):
        cand_cam = to_cam(ins_locs[cand_idx, :3].astype(np.float64))
        delta = cand_cam - anchor_cam
        if rel in ('LEFT', 'RIGHT'):
            return float(delta[0])
        else:
            return float(delta[2])

    expected_positive = rel in ('RIGHT', 'BEHIND')

    valid = _valid_cands(candidates, ins_locs)
    keep = []
    for c in valid:
        off = signed_offset(c['instance_index'])
        if expected_positive and off > 0:
            keep.append(c)
        elif (not expected_positive) and off < 0:
            keep.append(c)

    if not keep:
        return _fallback(candidates), 'fallback_empty'
    return keep, 'direction'


# ──────────────────────────────────────────────────────────────
# Filter: Diversity pruning
# ──────────────────────────────────────────────────────────────

def filter_diversity(candidates, record, ins_locs, k=PASS_THRESH, cell_size=1.0):
    """Spatial diversity prune: top-1 per 1m^2 XY cell, then top-k."""
    ge = (record.get('trace') or {}).get('geo_evidence') or {}
    gp = ge.get('geo_parse') or {}
    rel = (gp.get('relation') or '').upper()
    anchor_indices = ge.get('anchor_indices') or []

    if anchor_indices or rel in ('HIGHER', 'LOWER', 'UNDER', 'BELOW', 'ABOVE',
                                  'CLOSEST', 'NEAR', 'BESIDE', 'NEXT',
                                  'LEFT', 'RIGHT', 'FRONT', 'BEHIND', 'BETWEEN'):
        return candidates, 'skip'

    if len(candidates) <= k:
        return candidates, 'pass_through'

    valid = _valid_cands(candidates, ins_locs)
    seen_cells = {}
    for c in valid:
        cx, cy = ins_locs[c['instance_index'], :2]
        cell = (int(cx // cell_size), int(cy // cell_size))
        if cell not in seen_cells:
            seen_cells[cell] = c

    keep = list(seen_cells.values())[:k]
    if not keep:
        return _fallback(candidates), 'fallback_empty'
    return keep, 'diversity'


# ──────────────────────────────────────────────────────────────
# Filter: Between (BETWEEN -- two anchors)
# ──────────────────────────────────────────────────────────────

def filter_between(candidates, record, ins_locs):
    """Keep candidates whose center lies in the corridor between two anchor centers."""
    ge = (record.get('trace') or {}).get('geo_evidence') or {}
    gp = ge.get('geo_parse') or {}
    rel = (gp.get('relation') or '').upper()
    anchor_indices = ge.get('anchor_indices') or []

    if rel != 'BETWEEN' or len(anchor_indices) < 2:
        return candidates, 'skip'

    if len(candidates) <= PASS_THRESH:
        return candidates, 'pass_through'

    a = ins_locs[anchor_indices[0], :3].astype(np.float64)
    b = ins_locs[anchor_indices[1], :3].astype(np.float64)
    ab = b - a
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-6:
        return _fallback(candidates), 'fallback_zero_len'

    ab_unit = ab / ab_len
    lat_thresh = max(
        np.max(ins_locs[anchor_indices[0], 3:6]) * 0.75,
        np.max(ins_locs[anchor_indices[1], 3:6]) * 0.75,
        0.3,
    )

    valid = _valid_cands(candidates, ins_locs)
    keep = []
    for c in valid:
        p = ins_locs[c['instance_index'], :3].astype(np.float64)
        ap = p - a
        proj = float(np.dot(ap, ab_unit))
        lateral = np.linalg.norm(ap - proj * ab_unit)
        if 0.0 <= proj <= ab_len and lateral <= lat_thresh:
            keep.append(c)

    if not keep:
        return _fallback(candidates), 'fallback_empty'
    return keep, 'between'


# ──────────────────────────────────────────────────────────────
# Composite: Full filter
# ──────────────────────────────────────────────────────────────

def filter_full(candidates, record, ins_locs):
    """Apply the appropriate relation-specific filter, fallback to diversity."""
    ge = (record.get('trace') or {}).get('geo_evidence') or {}
    gp = ge.get('geo_parse') or {}
    rel = (gp.get('relation') or '').upper()

    if len(candidates) <= PASS_THRESH:
        return candidates, 'pass_through'

    if rel in ('HIGHER', 'ABOVE', 'LOWER', 'UNDER', 'BELOW'):
        kept, tag = filter_vertical(candidates, record, ins_locs)
    elif rel in ('CLOSEST', 'NEAR', 'BESIDE', 'NEXT'):
        kept, tag = filter_distance(candidates, record, ins_locs)
    elif rel in ('LEFT', 'RIGHT', 'FRONT', 'BEHIND'):
        kept, tag = filter_direction(candidates, record, ins_locs)
    elif rel == 'BETWEEN':
        kept, tag = filter_between(candidates, record, ins_locs)
    else:
        kept, tag = filter_diversity(candidates, record, ins_locs)

    return kept, tag


FILTER_REGISTRY = {
    'topk':      filter_topk,
    'vertical':  filter_vertical,
    'distance':  filter_distance,
    'direction': filter_direction,
    'diversity': filter_diversity,
    'between':   filter_between,
    'full':      filter_full,
}
