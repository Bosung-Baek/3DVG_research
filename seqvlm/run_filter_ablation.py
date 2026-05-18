"""
Filter ablation runner.

Usage:
    conda run -n tsdsr python3 seqvlm/run_filter_ablation.py \
        --input experiments/full_ablation/outputs/full_E_V_viewpoint.jsonl \
        --variants topk vertical distance direction diversity full \
        --out_dir experiments/full_ablation/outputs/filter_ablation
"""
import argparse
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'seqvlm'))

from seqvlm.candidate_filter import FILTER_REGISTRY, _load_ins_locs

# ── Metrics helpers ────────────────────────────────────────────

def calc_iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    amin, amax = a[:3] - a[3:] / 2, a[:3] + a[3:] / 2
    bmin, bmax = b[:3] - b[3:] / 2, b[:3] + b[3:] / 2
    inter = np.maximum(0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
    iv = float(np.prod(inter))
    av = float(np.prod(np.maximum(0, amax - amin)))
    bv = float(np.prod(np.maximum(0, bmax - bmin)))
    u = av + bv - iv
    return iv / u if u > 0 else 0.0


def best_oracle_iou(candidates, ins_locs, target_box):
    best = 0.0
    for c in candidates:
        ii = c.get('instance_index', -1)
        if 0 <= ii < len(ins_locs):
            best = max(best, calc_iou(target_box, ins_locs[ii]))
    return best


def compute_metrics(results, label=''):
    n = len(results)
    if n == 0:
        return {}
    acc25  = sum(r['acc25']    for r in results) / n
    acc50  = sum(r['acc50']    for r in results) / n
    ora25  = sum(r['oracle25'] for r in results) / n
    ora50  = sum(r['oracle50'] for r in results) / n
    severe = sum(1 for r in results if r['iou'] == 0.0)
    ncands = [r['n_cands_after'] for r in results]
    unique = [r for r in results if r.get('unique')]
    multi  = [r for r in results if not r.get('unique')]
    return {
        'label':       label,
        'n':           n,
        'acc25':       round(acc25, 4),
        'acc50':       round(acc50, 4),
        'oracle25':    round(ora25, 4),
        'oracle50':    round(ora50, 4),
        'severe_miss': severe,
        'avg_cands':   round(float(np.mean(ncands)), 2),
        'median_cands': float(np.median(ncands)),
        'unique_acc25': round(sum(r['acc25'] for r in unique) / len(unique), 4) if unique else None,
        'multi_acc25':  round(sum(r['acc25'] for r in multi) / len(multi), 4) if multi else None,
        'fallback_empty': sum(1 for r in results if r.get('filter_tag', '').startswith('fallback')),
    }


# ── VLM re-selection ───────────────────────────────────────────

def vlm_reselect(filtered_candidates, record, predictor):
    """Re-run VLM predict() on the filtered candidate image list."""
    images = [c['evidence_path'] for c in filtered_candidates
              if c.get('evidence_path') and Path(c['evidence_path']).exists()]
    if not images:
        return None
    caption = record.get('caption', '')
    try:
        pred_local = predictor.predict(images, caption)
    except Exception as e:
        print(f'  VLM error: {e}')
        return None
    if pred_local is None:
        return None
    return filtered_candidates[pred_local]['instance_index']


# ── Main runner ────────────────────────────────────────────────

def run_variant(variant_name, records, predictor, scene_locs_cache, out_dir: Path):
    filter_fn = FILTER_REGISTRY[variant_name]
    out_path  = out_dir / f'filter_{variant_name}.jsonl'

    results = []
    with open(out_path, 'w') as fout:
        for rec in records:
            scene_id = rec['scene_id']
            if scene_id not in scene_locs_cache:
                scene_locs_cache[scene_id] = _load_ins_locs(scene_id)
            ins_locs = scene_locs_cache[scene_id]

            ge = (rec.get('trace') or {}).get('geo_evidence') or {}
            orig_cands = ge.get('candidates', []) if isinstance(ge, dict) else []

            try:
                filtered, tag = filter_fn(orig_cands, rec, ins_locs)
            except Exception as e:
                print(f'  Filter error ({variant_name}, {scene_id}): {e}')
                filtered, tag = orig_cands, 'error'

            n_orig   = len(orig_cands)
            n_after  = len(filtered)
            tb       = rec['target_box']
            ora25    = int(best_oracle_iou(filtered, ins_locs, tb) >= 0.25)
            ora50    = int(best_oracle_iou(filtered, ins_locs, tb) >= 0.50)

            if filtered != orig_cands and predictor is not None and n_after > 0:
                pred_inst = vlm_reselect(filtered, rec, predictor)
                if pred_inst is not None and 0 <= pred_inst < len(ins_locs):
                    pred_box = ins_locs[pred_inst].tolist()
                    iou = calc_iou(tb, pred_box)
                    acc25 = int(iou >= 0.25)
                    acc50 = int(iou >= 0.50)
                else:
                    iou, acc25, acc50 = rec.get('iou', 0), 0, 0
            else:
                iou    = rec.get('iou', 0)
                acc25  = rec.get('acc25', 0) or 0
                acc50  = rec.get('acc50', 0) or 0
                pred_inst = (rec.get('trace') or {}).get('pred_instance', -1)

            row = {
                **{k: rec[k] for k in ('scene_id', 'obj_id', 'obj_name', 'caption',
                                        'target_box', 'unique')
                   if k in rec},
                'iou':            iou,
                'acc25':          acc25,
                'acc50':          acc50,
                'oracle25':       ora25,
                'oracle50':       ora50,
                'n_cands_before': n_orig,
                'n_cands_after':  n_after,
                'filter_tag':     tag,
                'pred_instance':  pred_inst,
            }
            results.append(row)
            fout.write(json.dumps(row) + '\n')

        metrics = compute_metrics(results, label=variant_name)

    meta_path = out_dir / f'filter_{variant_name}_metrics.json'
    meta_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def print_table(all_metrics):
    hdr = 'Method               N  AvgCand  Med  Oracle@25  Acc@25  Acc@50  SevereMiss  Fallback'
    print(hdr)
    print('-' * len(hdr))
    for m in all_metrics:
        print('%-20s %4d %8.1f %5.0f %10.3f %8.3f %8.3f %11d %9d' % (
            m['label'], m['n'], m['avg_cands'], m['median_cands'],
            m['oracle25'], m['acc25'], m['acc50'],
            m['severe_miss'], m['fallback_empty']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--variants', nargs='+',
                        default=['topk', 'vertical', 'distance', 'direction', 'diversity', 'full'])
    parser.add_argument('--out_dir', default='experiments/full_ablation/outputs/filter_ablation')
    parser.add_argument('--no_vlm', action='store_true',
                        help='Skip VLM re-run (oracle/candidate stats only)')
    parser.add_argument('--vlm_model', default='local-qwen')
    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--max_batch_size', type=int, default=4)
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(args.input)]
    print(f'Loaded {len(records)} records from {args.input}')

    predictor = None
    if not args.no_vlm:
        from seqvlm.adaptive_predictor import AdpativePredictor
        from seqvlm import adaptive_predictor as ap_module
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('no cuda')
        except Exception:
            from seqvlm.ablation import mock_vlm_answer
            orig = ap_module.invoke_api
            def _mock(model, messages):
                if model == 'local-qwen':
                    return mock_vlm_answer(messages)
                return orig(model, messages)
            ap_module.invoke_api = _mock

        predictor = AdpativePredictor(
            vlm_model=args.vlm_model,
            max_retry=args.max_retry,
            max_batch_size=args.max_batch_size,
            image_path='',
            force_program_first=False,
            use_spatial_filter=False,
            view_source='geo_frame_selection',
            input_format='geo_raw_frame',
        )

    scene_locs_cache = {}
    all_metrics = []

    baseline_results = []
    for rec in records:
        scene_id = rec['scene_id']
        if scene_id not in scene_locs_cache:
            scene_locs_cache[scene_id] = _load_ins_locs(scene_id)
        ins_locs = scene_locs_cache[scene_id]
        ge = (rec.get('trace') or {}).get('geo_evidence') or {}
        cands = ge.get('candidates', []) if isinstance(ge, dict) else []
        tb = rec['target_box']
        baseline_results.append({
            **{k: rec[k] for k in ('scene_id', 'obj_id', 'caption', 'unique') if k in rec},
            'iou':            rec.get('iou', 0),
            'acc25':          rec.get('acc25', 0) or 0,
            'acc50':          rec.get('acc50', 0) or 0,
            'oracle25':       int(best_oracle_iou(cands, ins_locs, tb) >= 0.25),
            'oracle50':       int(best_oracle_iou(cands, ins_locs, tb) >= 0.50),
            'n_cands_before': len(cands),
            'n_cands_after':  len(cands),
            'filter_tag':     'baseline',
        })
    all_metrics.append(compute_metrics(baseline_results, label='E_V (baseline)'))

    for variant in args.variants:
        if variant not in FILTER_REGISTRY:
            print(f'Unknown variant: {variant}, skipping')
            continue
        print(f'\n=== Running filter: {variant} ===')
        m = run_variant(variant, records, predictor, scene_locs_cache, out_dir)
        all_metrics.append(m)
        print(f'  acc25={m["acc25"]:.3f}  oracle25={m["oracle25"]:.3f}  avg_cands={m["avg_cands"]:.1f}')

    print('\n\n=== RESULTS TABLE ===')
    print_table(all_metrics)

    summary_path = out_dir / 'filter_ablation_summary.json'
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f'\nSaved summary -> {summary_path}')


if __name__ == '__main__':
    main()
