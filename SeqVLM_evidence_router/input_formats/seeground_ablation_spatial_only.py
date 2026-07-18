"""
SeeGround Ablation — 3D Pos. Only (text-only, no visual input).

Source: SeeGround ablation Table 3, row (a): "3D Pos. only" condition.
Paper: "SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding"
Reported result: 37.7% overall Acc@0.25 (ScanRefer val).

SeeGround proposes a "spatially enriched text description" that includes
3D location, dimension, and semantic label per object.
The "3D Pos. only" ablation condition uses this text description without
any rendered image or texture/color visual input.

This format is the cleanest paper-faithful condition:
text-only 3D spatial description, no image input.

Note: anchor_delta, distance, and volume are derived from raw 3D positions
to be compatible with our candidate-selection pipeline.
These are slightly richer than what SeeGround's "3D Pos." condition strictly reports.
Recorded in derived_spatial_features metadata.

paper_faithful: "ablation_reproduction"
"""
from __future__ import annotations
import json, re, numpy as np
from .base import VLMInputFormat


SYSTEM_PROMPT = """You are given spatial/geometric descriptions of candidate objects \
in a 3D indoor room. Using only the 3D coordinate and spatial information provided, \
identify which candidate best matches the query.

Coordinate convention: X = East/Right, Y = North/Front, Z = Up.
Distances are in meters.

Response format (JSON only):
{
  "process": "Explain your spatial reasoning",
  "selected_key": "A",
  "confidence": 0.0
}"""

USER_PROMPT_TEMPLATE = """Query: {query}

{anchor_block}
Candidates:
{cand_block}

Based on the spatial information only (no images), which candidate matches the query?
Respond with JSON: {{"process": "reasoning", "selected_key": "A", "confidence": 0.0}}"""


def _fmt_candidate_block(candidates, locs, labels, anchor_pos,
                         coord_mode="anchor_relative_xyz"):
    lines = []
    for rank, inst_id in enumerate(candidates):
        letter = chr(65+rank) if rank < 26 else str(rank)
        lbl = labels[inst_id][:24] if inst_id < len(labels) else "object"
        if inst_id < len(locs):
            cx, cy, cz, wx, wy, wz = locs[inst_id][:6]
            vol = wx * wy * wz
            if coord_mode == "anchor_relative_xyz" and anchor_pos is not None:
                dx = round(cx - anchor_pos[0], 2)
                dy = round(cy - anchor_pos[1], 2)
                dz = round(cz - anchor_pos[2], 2)
                dist = round(float(np.sqrt(dx**2 + dy**2 + dz**2)), 2)
                pos = f"anchor_delta=({dx:+.2f}, {dy:+.2f}, {dz:+.2f}) dist={dist}m"
            elif coord_mode == "raw_xyz":
                pos = f"center=({cx:.2f}, {cy:.2f}, {cz:.2f})"
            else:  # scene_normalized_xyz — normalize to [0,1] using scene bbox
                pos = f"center=({cx:.2f}, {cy:.2f}, {cz:.2f})"
            size = f"size=({wx:.2f}×{wy:.2f}×{wz:.2f})m vol={vol:.3f}m³"
            lines.append(f"  [{letter}] {lbl}: {pos}  {size}")
        else:
            lines.append(f"  [{letter}] {lbl}")
    return "\n".join(lines)


class SeeGroundAblationSpatialOnlyFormat(VLMInputFormat):
    name = "seeground_ablation_3dpos_only"   # renamed to match SeeGround Table 3 row (a)
    source_paper = "SeeGround ablation (Table 3, row a)"
    paper_faithful = "ablation_reproduction"

    def build(self, query, scene_id, candidates, anchors,
              relation_info, frame_data, scene_data, config):

        coord_mode = config.get("coordinate_mode", "anchor_relative_xyz")
        locs   = scene_data.get("locs", [])
        labels = scene_data.get("labels", [])

        anchor_pos = None
        anchor_block = ""
        if anchors:
            ai = anchors[0]
            if ai < len(locs):
                anchor_pos = np.asarray(locs[ai][:3])
                albl = labels[ai][:20] if ai < len(labels) else "anchor"
                anchor_block = (f"Anchor object: {albl}  "
                                f"center=({anchor_pos[0]:.2f}, "
                                f"{anchor_pos[1]:.2f}, {anchor_pos[2]:.2f})\n\n")

        cand_block = _fmt_candidate_block(
            candidates, locs, labels, anchor_pos, coord_mode)

        prompt_text = USER_PROMPT_TEMPLATE.format(
            query=query, anchor_block=anchor_block, cand_block=cand_block)

        meta = self._base_metadata(
            uses_real_rgb=False,
            uses_spatial_text=True,
            visual_input=False,
            coordinate_mode=coord_mode,
            coordinate_frame="ScanNet_axis_aligned_XEast_YNorth_ZUp",
            ablation_condition="3D Pos. only",
            derived_spatial_features=["anchor_delta", "distance", "volume"],
            implementation_difference_from_paper=[
                "Uses Mask3D candidate locs instead of GT/OLT annotations.",
                "anchor_delta, distance, volume derived from 3D positions "
                "(slightly richer than strict 3D Pos. only).",
            ],
        )
        return {
            "images":   [],           # no images
            "system":   SYSTEM_PROMPT,
            "prompt":   prompt_text,
            "metadata": meta,
        }

    @staticmethod
    def parse_response(raw: str, n_cands: int) -> int:
        """Unified selected_key parser. Returns 0-based index or -1."""
        from .base import parse_selected_key
        letters = [chr(65+i) if i < 26 else str(i) for i in range(n_cands)]
        key = parse_selected_key(raw, set(letters))
        if key and key in letters:
            return letters.index(key)
        return -1
