#!/bin/bash
set -e
cd /home/knuvi/bosung/SeqVLM
export PYTHONPATH=.
CONDA=/home/knuvi/anaconda3/bin/conda
log() { echo "[$(date '+%H:%M:%S')] $*"; }
OUT=experiments/ablation_seqvlm_3scenes/outputs
LOG=experiments/ablation_seqvlm_3scenes/logs

log "Deleting old preprocessed data..."
rm -rf data/scanrefer_preprocessed/scene0606_00
rm -rf data/scanrefer_preprocessed/scene0221_00
rm -rf data/scanrefer_preprocessed/scene0329_00

log "Re-preprocessing with area-based top-K..."
$CONDA run -n tsdsr python preprocess/preprocess_mini.py \
    --scenes scene0606_00 scene0221_00 scene0329_00
log "Preprocessing done."

run_exp() {
    local name=$1; shift
    log "Starting: $name"
    $CONDA run -n tsdsr python seqvlm/evaluate.py \
        --exp_name "$name" \
        --data_path data/mini_test.json \
        --image_path data/scanrefer_preprocessed \
        --vlm_model local-qwen \
        --output_path "${OUT}/${name}.jsonl" \
        "$@" > "${LOG}/${name}.log" 2>&1
    log "Done: $name"
}

run_exp mini_E0_area \
    --ablation_axis baseline \
    --input_format seqvlm_canvas \
    --view_source seqvlm_canvas

run_exp mini_E_V_viewpoint \
    --ablation_axis baseline \
    --input_format geo_raw_frame \
    --view_source geo_frame_selection

run_exp mini_E_VF_system \
    --ablation_axis baseline \
    --input_format geo_bbox_overlay \
    --view_source geo_frame_selection

log "All done."
