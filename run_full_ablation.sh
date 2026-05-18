#!/bin/bash
set -e
cd /home/knuvi/bosung/SeqVLM
export PYTHONPATH=.
CONDA=/home/knuvi/anaconda3/bin/conda
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Preprocess missing scenes in batches of 20 ────────────────────────
log "Checking scenes to preprocess..."
python3 -c "
import json
from pathlib import Path
data = json.load(open('data/scanrefer_250.json'))
scenes = sorted(set(d['scan_id'] for d in data))
done = set(p.name for p in Path('data/scanrefer_preprocessed').iterdir() if p.is_dir())
missing = [s for s in scenes if s not in done]
print(f'Missing: {len(missing)} scenes')
Path('/tmp/missing_scenes.txt').write_text('\n'.join(missing))
"

if [ -s /tmp/missing_scenes.txt ]; then
    # Process in batches of 20 to avoid arg limit issues
    batch=()
    count=0
    while IFS= read -r scene; do
        batch+=("$scene")
        count=$((count+1))
        if [ $count -eq 20 ]; then
            log "Preprocessing batch: ${batch[*]:0:1}..."
            $CONDA run -n tsdsr python preprocess/preprocess_mini.py --scenes "${batch[@]}"
            batch=()
            count=0
        fi
    done < /tmp/missing_scenes.txt
    # Process remaining
    if [ ${#batch[@]} -gt 0 ]; then
        log "Preprocessing final batch: ${batch[*]:0:1}..."
        $CONDA run -n tsdsr python preprocess/preprocess_mini.py --scenes "${batch[@]}"
    fi
    log "All preprocessing done."
else
    log "All scenes already preprocessed."
fi

# ── Step 2: Full ablation experiments ─────────────────────────────────────────
mkdir -p experiments/full_ablation/outputs experiments/full_ablation/logs

run_exp() {
    local name=$1; shift
    log "Starting: $name"
    $CONDA run -n tsdsr python seqvlm/evaluate.py \
        --exp_name "$name" \
        --data_path data/scanrefer_250.json \
        --image_path data/scanrefer_preprocessed \
        --vlm_model local-qwen \
        --output_path "experiments/full_ablation/outputs/${name}.jsonl" \
        "$@" \
        > "experiments/full_ablation/logs/${name}.log" 2>&1
    log "Done: $name"
}

run_exp full_E0_baseline \
    --ablation_axis baseline \
    --input_format seqvlm_canvas \
    --view_source seqvlm_canvas

run_exp full_E_V_viewpoint \
    --ablation_axis baseline \
    --input_format geo_raw_frame \
    --view_source geo_frame_selection

run_exp full_E_VF_system \
    --ablation_axis baseline \
    --input_format geo_bbox_overlay \
    --view_source geo_frame_selection

log "All ablations complete."
