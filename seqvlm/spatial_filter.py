"""
BEV spatial filter for SeqVLM ablation.

Filters target candidate instances using 3D spatial relations parsed from
the query caption. Reduces the candidate set before VLM selection so the
VLM only sees geometrically plausible candidates.
"""

import numpy as np
from seqvlm.ablation import parse_geo_structured


def bev_spatial_filter(
    candidate_indices,
    ins_locs,
    caption,
    obj_name,
    ins_labels,
    ins_scores,
    class_predictor,
    fallback_on_empty=True,
):
    if not candidate_indices:
        return candidate_indices

    parsed = parse_geo_structured(caption, obj_name)
    relation = parsed.get("relation", "LOC")
    anchor_text = parsed.get("anchor", "")

    if relation == "LOC" or not anchor_text:
        return candidate_indices

    anchor_locs = _get_anchor_locs(parsed, ins_locs, ins_labels, ins_scores, class_predictor)
    if not anchor_locs:
        return candidate_indices

    anchor_centroid = np.mean([loc[:3] for loc in anchor_locs], axis=0)

    filtered = _apply_relation(
        relation=relation,
        candidate_indices=candidate_indices,
        ins_locs=ins_locs,
        anchor_locs=anchor_locs,
        anchor_centroid=anchor_centroid,
    )

    if not filtered and fallback_on_empty:
        return candidate_indices
    return filtered


def _get_anchor_locs(parsed, ins_locs, ins_labels, ins_scores, class_predictor):
    anchor_text = parsed.get("anchor", "")
    if not anchor_text:
        return []
    try:
        anchor_cls = class_predictor(anchor_text, ins_labels)
    except Exception:
        return []
    candidates = [
        (i, float(ins_scores[i]))
        for i, label in enumerate(ins_labels)
        if label == anchor_cls and float(ins_scores[i]) > 0.2
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in candidates[:2]]
    return [np.asarray(ins_locs[i], dtype=np.float32) for i in top_indices]


def _apply_relation(relation, candidate_indices, ins_locs, anchor_locs, anchor_centroid):
    filtered = []

    if relation in ("LEFT", "RIGHT", "FRONT", "BEHIND", "HIGHER", "LOWER"):
        for idx in candidate_indices:
            loc = np.asarray(ins_locs[idx], dtype=np.float32)
            rel = loc[:3] - anchor_centroid
            keep = False
            if relation == "LEFT":
                keep = rel[0] < 0
            elif relation == "RIGHT":
                keep = rel[0] > 0
            elif relation == "FRONT":
                keep = rel[1] > 0
            elif relation == "BEHIND":
                keep = rel[1] < 0
            elif relation == "HIGHER":
                keep = rel[2] > 0
            elif relation == "LOWER":
                keep = rel[2] < 0
            if keep:
                filtered.append(idx)

    elif relation == "CLOSEST":
        dists = [(idx, float(np.linalg.norm(np.asarray(ins_locs[idx], dtype=np.float32)[:3] - anchor_centroid)))
                 for idx in candidate_indices]
        dists.sort(key=lambda x: x[1])
        filtered = [idx for idx, _ in dists[:3]]

    elif relation == "FARTHEST":
        dists = [(idx, float(np.linalg.norm(np.asarray(ins_locs[idx], dtype=np.float32)[:3] - anchor_centroid)))
                 for idx in candidate_indices]
        dists.sort(key=lambda x: x[1], reverse=True)
        filtered = [idx for idx, _ in dists[:3]]

    elif relation == "BETWEEN":
        anchor_xy = np.stack([loc[:2] for loc in anchor_locs], axis=0)
        xy_min = anchor_xy.min(axis=0)
        xy_max = anchor_xy.max(axis=0)
        margin = (xy_max - xy_min) * 0.5
        lo = xy_min - margin
        hi = xy_max + margin
        for idx in candidate_indices:
            loc = np.asarray(ins_locs[idx], dtype=np.float32)
            if np.all(loc[:2] >= lo) and np.all(loc[:2] <= hi):
                filtered.append(idx)

    elif relation == "MID":
        x_vals = [(idx, float(np.asarray(ins_locs[idx], dtype=np.float32)[0]))
                  for idx in candidate_indices]
        x_vals.sort(key=lambda x: x[1])
        n = len(x_vals)
        lo_i = n // 3
        hi_i = n - n // 3
        filtered = [idx for idx, _ in x_vals[lo_i:hi_i]] or [x_vals[n // 2][0]]

    elif relation == "MIN_SIZE":
        sizes = [(idx, float(np.asarray(ins_locs[idx], dtype=np.float32)[3]) *
                       float(np.asarray(ins_locs[idx], dtype=np.float32)[4]))
                 for idx in candidate_indices]
        sizes.sort(key=lambda x: x[1])
        filtered = [idx for idx, _ in sizes[:3]]

    elif relation == "MAX_SIZE":
        sizes = [(idx, float(np.asarray(ins_locs[idx], dtype=np.float32)[3]) *
                       float(np.asarray(ins_locs[idx], dtype=np.float32)[4]))
                 for idx in candidate_indices]
        sizes.sort(key=lambda x: x[1], reverse=True)
        filtered = [idx for idx, _ in sizes[:3]]

    else:
        filtered = list(candidate_indices)

    return filtered
