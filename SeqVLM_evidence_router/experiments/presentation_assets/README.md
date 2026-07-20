# Presentation Assets

Slide-ready qualitative assets generated from real locked query/result records.
Each asset is saved as both PNG and PDF.

| Asset | File stem | Use |
|---|---|---|
| 1 | `asset_01_e0_failure_bev_overlay` | E0 failure case with prediction/GT overlaid on BEV. |
| 2 | `asset_02_spatial_success_bev_overlay` | Spatial-only success case for the same query, overlaid on BEV. |
| 3 | `asset_03_failure_vs_success_3d_overlay` | Same failure/success pair as a 3D box rendering. |
| 4 | `asset_04_geometric_e0_vs_3d_position_2x2` | 2x2 comparison: E0 query/input vs 3D-position query/input, with BEV overlays. |

Regenerate:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /home/knuvi/anaconda3/envs/sam3/bin/python tools/create_presentation_assets.py
```
