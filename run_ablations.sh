#!/bin/bash
set -e
cd /home/knuvi/bosung/SeqVLM
export PYTHONPATH=.

run_exp() {
    local name=$1
    shift
    echo "========== Starting: $name =========="
    python seqvlm/evaluate.py \
        --exp_name "$name" \
        --data_path data/mini_test.json \
        --image_path data/scanrefer_preprocessed \
        --vlm_model local-qwen \
        --output_path "experiments/ablation_seqvlm_3scenes/outputs/${name}.jsonl" \
        "$@" \
        > "experiments/ablation_seqvlm_3scenes/logs/${name}.log" 2>&1
    echo "========== Done: $name =========="
}

# E0: baseline (full-frame + red bbox canvas, no filter)
run_exp baseline_E0 \
    --ablation_axis baseline \
    --input_format seqvlm_canvas \
    --view_source seqvlm_canvas

# E_P: BEV spatial filter only
run_exp spatial_filter_E_P \
    --ablation_axis baseline \
    --input_format seqvlm_canvas \
    --view_source seqvlm_canvas \
    --use_spatial_filter

# E_PVF: full system (BEV filter + geo frame + bbox overlay)
run_exp full_system_E_PVF \
    --ablation_axis baseline \
    --input_format geo_bbox_overlay \
    --view_source geo_frame_selection \
    --use_spatial_filter

echo "All ablations complete."
