# SeqVLM Setup

This repo has a few machine-specific paths. Update them before running on a new server.

## Conda Environment

Create or activate the `sam3` environment used by the evaluation scripts:

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install -r requirements.txt
```

If your environment is already provisioned, run commands through:

```bash
conda run -n sam3 python ...
```

## Per-Server Constants

Update these constants after moving the repo or data:

| File | Constants |
| --- | --- |
| `seqvlm/utils.py` | `_SCANNET_ORIGIN_DIR`, `_MASK3D_DIR` |
| `preprocess/preprocess_mini.py` | `SCANNET_DIR`, `SCANNET_ORIGIN_DIR`, `MASK3D_DIR`, `OUTPUT_DIR` |
| `seqvlm/api.py` | `LOCAL_QWEN_PATH` |

## Required Data Layout

Mask3D predictions:

```text
{MASK3D_DIR}/{scene_id}.npz
```

Each NPZ should contain the instance arrays consumed by `seqvlm/utils.py` and `preprocess/preprocess_mini.py`, including `ins_pcds`, `ins_labels`, and `ins_scores` where applicable.

ScanNet RGB-D frames:

```text
{SCANNET_DIR}/{scene_id}/color/{frame_id}.jpg
{SCANNET_DIR}/{scene_id}/depth/{frame_id}.png
{SCANNET_DIR}/{scene_id}/pose/{frame_id}.txt
{SCANNET_DIR}/{scene_id}/intrinsic/intrinsic_color.txt
```

ScanNet original scene metadata:

```text
{SCANNET_ORIGIN_DIR}/{scene_id}/{scene_id}.txt
{SCANNET_ORIGIN_DIR}/{scene_id}/{scene_id}_vh_clean_2.ply
{SCANNET_ORIGIN_DIR}/{scene_id}/{scene_id}_vh_clean_2.0.010000.segs.json
{SCANNET_ORIGIN_DIR}/{scene_id}/{scene_id}.aggregation.json
```

The `{scene_id}.txt` file must include the `axisAlignment` matrix.

Qwen2-VL-7B weights:

```text
/home/knuvi/bosung/models/Qwen2-VL-7B-Instruct
```

Set `seqvlm/api.py:LOCAL_QWEN_PATH` to the local weights directory. If that directory is missing, `seqvlm/api.py` falls back to `Qwen/Qwen2-VL-7B-Instruct`.

## Preprocessing

Generate mini ScanRefer canvases:

```bash
PYTHONPATH=. conda run -n sam3 python preprocess/preprocess_mini.py \
  --scenes scene0606_00 scene0221_00 scene0329_00
```

The default output is:

```text
data/scanrefer_preprocessed/{scene_id}/{obj_id}/canvas.jpg
```

## Ablation Runs

Run from the repo root.

E0 baseline:

```bash
PYTHONPATH=. conda run -n sam3 python seqvlm/evaluate.py \
  --exp_name E0_baseline \
  --data_path data/mini_test.json \
  --image_path data/scanrefer_preprocessed \
  --vlm_model local-qwen \
  --ablation_axis baseline \
  --input_format seqvlm_canvas \
  --view_source seqvlm_canvas
```

E2 geo_raw:

```bash
PYTHONPATH=. conda run -n sam3 python seqvlm/evaluate.py \
  --exp_name E2_geo_raw \
  --data_path data/mini_test.json \
  --image_path data/scanrefer_preprocessed \
  --vlm_model local-qwen \
  --ablation_axis input_format \
  --input_format geo_raw_frame \
  --view_source geo_frame_selection
```

E3 geo_bbox:

```bash
PYTHONPATH=. conda run -n sam3 python seqvlm/evaluate.py \
  --exp_name E3_geo_bbox \
  --data_path data/mini_test.json \
  --image_path data/scanrefer_preprocessed \
  --vlm_model local-qwen \
  --ablation_axis input_format \
  --input_format geo_bbox_overlay \
  --view_source geo_frame_selection
```

E_parsing:

```bash
PYTHONPATH=. conda run -n sam3 python seqvlm/evaluate.py \
  --exp_name E_parsing \
  --data_path data/mini_test.json \
  --image_path data/scanrefer_preprocessed \
  --vlm_model local-qwen \
  --ablation_axis parsing \
  --input_format seqvlm_canvas \
  --view_source seqvlm_canvas
```

E_viewpoint:

```bash
PYTHONPATH=. conda run -n sam3 python seqvlm/evaluate.py \
  --exp_name E_viewpoint \
  --data_path data/mini_test.json \
  --image_path data/scanrefer_preprocessed \
  --vlm_model local-qwen \
  --ablation_axis viewpoint \
  --input_format seqvlm_canvas \
  --view_source geo_frame_selection
```

**E_P - BEV spatial filter (parsing ablation):**
```bash
PYTHONPATH=. conda run -n sam3 python seqvlm/evaluate.py   --exp_name spatial_filter_E_P   --data_path data/mini_test.json   --image_path data/scanrefer_preprocessed   --vlm_model local-qwen   --ablation_axis baseline   --input_format seqvlm_canvas   --view_source seqvlm_canvas   --use_spatial_filter
```

**E_PVF - Full proposed system (BEV filter + geo frame + bbox overlay):**
```bash
PYTHONPATH=. conda run -n sam3 python seqvlm/evaluate.py   --exp_name full_system_E_PVF   --data_path data/mini_test.json   --image_path data/scanrefer_preprocessed   --vlm_model local-qwen   --ablation_axis baseline   --input_format geo_bbox_overlay   --view_source geo_frame_selection   --use_spatial_filter
```

## Key Modified Files

| File | Purpose |
| --- | --- |
| `seqvlm/evaluate.py` | Evaluation entrypoint and ablation flags |
| `seqvlm/api.py` | Local Qwen2-VL invocation and local weight auto-detection |
| `seqvlm/feat_handler.py` | Visual feature handling |
| `seqvlm/utils.py` | ScanNet and Mask3D loading utilities |
| `seqvlm/ablation.py` | Ablation helpers |
| `seqvlm/geo_evidence.py` | Geometry evidence generation |
| `seqvlm/adaptive_predictor.py` | Adaptive prediction flow |
| `visprog/view_interpreters.py` | View interpreter updates |
| `preprocess/preprocess_mini.py` | Mini preprocessing pipeline |
| `data/mini_test.json` | Mini evaluation split |
| `SETUP.md` | Portability and run documentation |
| `PROGRESS.md` | Project progress notes |
