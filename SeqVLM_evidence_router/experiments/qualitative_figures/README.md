# Qualitative Figures

These figures use real query/result records and pre-rendered SeqVLM canvas
images from the local workspace.

| Figure | File stem | Description |
|---|---|---|
| 1 | `qual1_3d_scene_query_candidate_boxes` | 3D scene coordinate view + natural-language query + candidate boxes. |
| 2 | `qual2_e0_rgb_canvas_candidate_overlay` | E0 RGB canvas candidates with candidate letter/id overlay. |
| 3 | `qual3_e0_failure_vs_spatial_success` | E0 failure vs spatial-only success on the same NR3D query. |
| 4 | `qual4_geometric_e0_vs_3d_position` | Geometric query comparison: E0 prediction vs 3D-position prediction. |

Regenerate:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /home/knuvi/anaconda3/envs/sam3/bin/python tools/create_qualitative_figures.py
```
