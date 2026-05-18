"""
Evidence-aware ablation runner.

Usage (dry run, no VLM):
    conda run -n tsdsr bash -c "cd /home/knuvi/bosung/SeqVLM && python3 -u seqvlm/run_evidence_ablation.py \
      --ev_input experiments/full_ablation/outputs/full_E_V_viewpoint.jsonl \
      --evf_input experiments/full_ablation/outputs/full_E_VF_system.jsonl \
      --no_vlm"

Usage (with VLM):
    conda run -n tsdsr bash -c "cd /home/knuvi/bosung/SeqVLM && python3 -u seqvlm/run_evidence_ablation.py \
      --ev_input experiments/full_ablation/outputs/full_E_V_viewpoint.jsonl \
      --evf_input experiments/full_ablation/outputs/full_E_VF_system.jsonl \
      --vlm_model local-qwen"

Logs are saved to:  experiments/full_ablation/logs/evidence_ablation_YYYYMMDD_HHMMSS.log
Results saved to:   experiments/full_ablation/outputs/evidence_ablation/YYYYMMDD_HHMMSS/
"""
import argparse
import copy
import datetime
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT   = Path(__file__).parent.parent
SCANNET_DIR = Path('/data/knuvi/bosung/scannet')
MASK3D_DIR  = Path('/data/knuvi/bosung/Mask3d/scannet200')
GEO_EV_DIR  = REPO_ROOT / 'experiments/full_ablation/geo_evidence/full_E_V_viewpoint'
GEO_EVF_DIR = REPO_ROOT / 'experiments/full_ablation/geo_evidence/full_E_VF_system'

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'seqvlm'))

# ── Helpers ────────────────────────────────────────────────────────────────────

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


def load_ins_locs(scene_id):
    npz = np.load(MASK3D_DIR / f'{scene_id}.npz', allow_pickle=True)
    locs = []
    for pcd in npz['ins_pcds']:
        p = np.asarray(pcd, np.float32)
        if p.shape[0] == 0:
            locs.append(np.zeros(6, np.float32))
            continue
        mn, mx = p[:, :3].min(0), p[:, :3].max(0)
        locs.append(np.concatenate([(mn + mx) / 2, mx - mn]))
    return np.array(locs, np.float32)


def raw_frame_path(scene_id, frame_id):
    return str(SCANNET_DIR / scene_id / 'color' / f'{frame_id}.jpg')


def ev_img_path(scene_id, local_idx, inst_idx):
    return str(GEO_EV_DIR / scene_id / f'cand_{local_idx:02d}_inst_{inst_idx}_geo_raw_frame.jpg')


def evf_img_path(scene_id, local_idx, inst_idx):
    return str(GEO_EVF_DIR / scene_id / f'cand_{local_idx:02d}_inst_{inst_idx}_geo_bbox_overlay.jpg')


def best_oracle_iou(candidates, ins_locs, target_box):
    best = 0.0
    for c in candidates:
        ii = c.get('instance_index', -1)
        if 0 <= ii < len(ins_locs):
            best = max(best, calc_iou(target_box, ins_locs[ii]))
    return best


def get_cands_and_paths(rec):
    ge = (rec.get('trace') or {}).get('geo_evidence') or {}
    cands = ge.get('candidates', []) if isinstance(ge, dict) else []
    paths = [c.get('evidence_path', '') for c in cands]
    return cands, paths, ge


def get_same_best_frame(candidates):
    fids = []
    for c in candidates:
        frames = c.get('frames') or []
        if frames:
            fids.append(frames[0].get('frame_id', -1))
    fids = [f for f in fids if f >= 0]
    return len(set(fids)) < len(fids) if len(fids) > 1 else False


def get_anchor_vis0(rec, candidates, ins_locs):
    tb = rec['target_box']
    for c in candidates:
        ii = c.get('instance_index', -1)
        if 0 <= ii < len(ins_locs) and calc_iou(tb, ins_locs[ii]) >= 0.25:
            frames = c.get('frames') or [{}]
            return frames[0].get('anchor_visible_points', 0) == 0
    return True


# ── Selectors ──────────────────────────────────────────────────────────────────
# Each returns (image_paths: list[str], candidates: list[dict], tag: str)

def select_baseline(rec, ins_locs, evf_rec=None):
    cands, paths, _ = get_cands_and_paths(rec)
    return paths, cands, 'baseline'


def select_topk(rec, ins_locs, evf_rec=None, k=10):
    cands, paths, _ = get_cands_and_paths(rec)
    if len(cands) <= k:
        return paths, cands, 'pass_through'
    return paths[:k], cands[:k], 'topk'


def select_anchor_vis(rec, ins_locs, evf_rec=None):
    cands, paths, ge = get_cands_and_paths(rec)
    anchor_indices = ge.get('anchor_indices') or []
    if not anchor_indices:
        return paths, cands, 'no_anchor'
    scene_id = rec['scene_id']
    new_paths = list(paths)
    swapped = False
    for k, c in enumerate(cands):
        frames = c.get('frames') or []
        if not frames:
            continue
        if frames[0].get('anchor_visible_points', 0) > 0:
            continue
        for fr in frames[1:]:
            if fr.get('anchor_visible_points', 0) > 0:
                p = raw_frame_path(scene_id, fr['frame_id'])
                if Path(p).exists():
                    new_paths[k] = p
                    swapped = True
                    break
    return new_paths, cands, 'anchor_vis' if swapped else 'no_swap'


def select_frame_div(rec, ins_locs, evf_rec=None):
    cands, paths, _ = get_cands_and_paths(rec)
    scene_id = rec['scene_id']
    new_paths = list(paths)
    seen_ids = set()
    swapped = False
    for k, c in enumerate(cands):
        frames = c.get('frames') or []
        if not frames:
            continue
        best_fid = frames[0].get('frame_id', -1)
        if best_fid not in seen_ids:
            seen_ids.add(best_fid)
            continue
        for fr in frames[1:]:
            fid = fr.get('frame_id', -1)
            if fid not in seen_ids:
                p = raw_frame_path(scene_id, fid)
                if Path(p).exists():
                    new_paths[k] = p
                    seen_ids.add(fid)
                    swapped = True
                    break
    return new_paths, cands, 'frame_div' if swapped else 'no_swap'


def select_cond_overlay(rec, ins_locs, evf_rec=None):
    cands, paths, ge = get_cands_and_paths(rec)
    scene_id = rec['scene_id']
    gp = ge.get('geo_parse') or {}
    rel = (gp.get('relation') or '').upper() if isinstance(gp, dict) else ''
    anchor_indices = ge.get('anchor_indices') or []
    new_paths = []
    for k, c in enumerate(cands):
        ii = c.get('instance_index', -1)
        local_idx = c.get('candidate_local_index', k)
        frames = c.get('frames') or []
        if rel in ('LEFT', 'RIGHT', 'FRONT', 'BEHIND', 'BETWEEN') and anchor_indices:
            p = evf_img_path(scene_id, local_idx, ii)
            if not Path(p).exists():
                p = ev_img_path(scene_id, local_idx, ii)
        elif rel in ('CLOSEST', 'NEAR', 'BESIDE', 'NEXT') and anchor_indices:
            found = None
            for fr in frames:
                if (fr.get('target_visible_points', 0) > 500
                        and fr.get('anchor_visible_points', 0) > 500):
                    rp = raw_frame_path(scene_id, fr['frame_id'])
                    if Path(rp).exists():
                        found = rp
                        break
            p = found if found else ev_img_path(scene_id, local_idx, ii)
        else:
            p = ev_img_path(scene_id, local_idx, ii)
        if not Path(p).exists():
            p = paths[k] if k < len(paths) else ''
        new_paths.append(p)
    return new_paths, cands, 'cond_overlay'


def select_distractor(rec, ins_locs, evf_rec=None, cell_size=1.0, max_k=10, min_n=4):
    cands, paths, _ = get_cands_and_paths(rec)
    if len(cands) <= min_n:
        return paths, cands, 'pass_through'
    cell_counts = {}
    new_cands = []
    for c in cands:
        ii = c.get('instance_index', -1)
        if not (0 <= ii < len(ins_locs)):
            continue
        cx, cy = float(ins_locs[ii, 0]), float(ins_locs[ii, 1])
        cell = (int(cx // cell_size), int(cy // cell_size))
        cnt = cell_counts.get(cell, 0)
        if cnt < 2:
            new_cands.append(c)
            cell_counts[cell] = cnt + 1
        if len(new_cands) >= max_k:
            break
    if not new_cands:
        return paths[:max_k], cands[:max_k], 'fallback_empty'
    new_paths = [c.get('evidence_path', '') for c in new_cands]
    tag = 'distractor' if len(new_cands) < len(cands) else 'pass_through'
    return new_paths, new_cands, tag


def select_full(rec, ins_locs, evf_rec=None):
    # Step 1: distractor pruning
    paths1, cands1, _ = select_distractor(rec, ins_locs, evf_rec)

    # Build synthetic record with pruned candidates for subsequent steps
    rec2 = copy.deepcopy(rec)
    ge2 = (rec2.get('trace') or {}).get('geo_evidence') or {}
    if isinstance(ge2, dict):
        ge2['candidates'] = cands1

    # Step 2: anchor co-visibility swap
    paths2, _, _ = select_anchor_vis(rec2, ins_locs, evf_rec)

    # Update evidence_path in cands for step 3
    cands2 = copy.deepcopy(cands1)
    for i, c in enumerate(cands2):
        if i < len(paths2):
            c['evidence_path'] = paths2[i]

    rec3 = copy.deepcopy(rec2)
    ge3 = (rec3.get('trace') or {}).get('geo_evidence') or {}
    if isinstance(ge3, dict):
        ge3['candidates'] = cands2

    # Step 3: frame diversity
    paths3, _, _ = select_frame_div(rec3, ins_locs, evf_rec)

    # Update evidence_path for step 4
    cands3 = copy.deepcopy(cands2)
    for i, c in enumerate(cands3):
        if i < len(paths3):
            c['evidence_path'] = paths3[i]

    rec4 = copy.deepcopy(rec3)
    ge4 = (rec4.get('trace') or {}).get('geo_evidence') or {}
    if isinstance(ge4, dict):
        ge4['candidates'] = cands3

    # Step 4: conditional overlay
    paths4, _, _ = select_cond_overlay(rec4, ins_locs, evf_rec)

    return paths4, cands1, 'full'


SELECTOR_REGISTRY = {
    'baseline':     select_baseline,
    'topk':         select_topk,
    'anchor_vis':   select_anchor_vis,
    'frame_div':    select_frame_div,
    'cond_overlay': select_cond_overlay,
    'distractor':   select_distractor,
    'full':         select_full,
}

# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(results, label, baseline_oracle25=None):
    n = len(results)
    if n == 0:
        return {'label': label, 'n': 0}
    acc25  = sum(r['acc25']    for r in results) / n
    acc50  = sum(r['acc50']    for r in results) / n
    ora25  = sum(r['oracle25'] for r in results) / n
    ora50  = sum(r['oracle50'] for r in results) / n
    avg_cb = sum(r['n_cands_before'] for r in results) / n
    avg_ca = sum(r['n_cands_after']  for r in results) / n
    sbf    = sum(1 for r in results if r.get('same_best_frame')) / n
    av0    = sum(1 for r in results if r.get('anchor_vis0'))    / n
    chair  = [r for r in results if r.get('obj_name', '') == 'chair']
    multi  = [r for r in results if not r.get('unique')]
    return {
        'label':             label,
        'n':                 n,
        'Acc@25':            round(acc25, 4),
        'Acc@50':            round(acc50, 4),
        'Oracle@25':         round(ora25, 4),
        'Oracle@50':         round(ora50, 4),
        'AvgCandBefore':     round(avg_cb, 2),
        'AvgCandAfter':      round(avg_ca, 2),
        'SameBestFrameRate': round(sbf, 4),
        'AnchorVis0Rate':    round(av0, 4),
        'ChairAcc':          round(sum(r['acc25'] for r in chair) / len(chair), 4) if chair else None,
        'MultipleAcc':       round(sum(r['acc25'] for r in multi) / len(multi), 4) if multi else None,
        'OracleDrop':        round(ora25 - baseline_oracle25, 4) if baseline_oracle25 is not None else None,
        'EmptyFallback':     sum(1 for r in results if (r.get('tag') or '').startswith('fallback')),
    }


def print_table(all_metrics):
    header = (f"{'Method':<22} {'Acc@25':>7} {'Acc@50':>7} {'Ora@25':>7} {'OraDrop':>8} "
              f"{'CandB':>6} {'CandA':>6} {'SameBF':>7} {'AncV0':>6} "
              f"{'ChairA':>7} {'MultiA':>7} {'Fallbk':>7}")
    logging.info('%s', header)
    logging.info('%s', '-' * len(header))
    for m in all_metrics:
        def fmt(v, w=7):
            if v is None: return ' ' * w
            if isinstance(v, float): return f'{v:>{w}.3f}'
            return f'{v:>{w}}'
        line = (f"{m.get('label',''):<22}"
                f" {fmt(m.get('Acc@25'))}"
                f" {fmt(m.get('Acc@50'))}"
                f" {fmt(m.get('Oracle@25'))}"
                f" {fmt(m.get('OracleDrop'), 8)}"
                f" {fmt(m.get('AvgCandBefore'), 6)}"
                f" {fmt(m.get('AvgCandAfter'), 6)}"
                f" {fmt(m.get('SameBestFrameRate'), 7)}"
                f" {fmt(m.get('AnchorVis0Rate'), 6)}"
                f" {fmt(m.get('ChairAcc'), 7)}"
                f" {fmt(m.get('MultipleAcc'), 7)}"
                f" {fmt(m.get('EmptyFallback'), 7)}")
        logging.info('%s', line)


# ── VLM re-selection ────────────────────────────────────────────────────────────

def vlm_reselect(image_paths, candidates, record, predictor):
    valid = [(i, p) for i, p in enumerate(image_paths) if p and Path(p).exists()]
    if not valid:
        return None
    imgs = [p for _, p in valid]
    caption = record.get('caption', '')
    try:
        pred_local = predictor.predict(imgs, caption)
    except Exception as e:
        logging.warning('VLM error: %s', e)
        return None
    if pred_local is None or pred_local >= len(valid):
        return None
    cand_idx = valid[pred_local][0]
    if cand_idx < len(candidates):
        return candidates[cand_idx].get('instance_index', -1)
    return None


# ── Per-variant runner ──────────────────────────────────────────────────────────

def run_variant(variant_name, selector_fn, ev_records, evf_lookup,
                predictor, scene_locs_cache, out_dir, baseline_oracle25=None):
    out_path = out_dir / f'evidence_{variant_name}.jsonl'
    results = []
    with open(out_path, 'w') as fout:
        for rec in ev_records:
            scene_id = rec['scene_id']
            if scene_id not in scene_locs_cache:
                scene_locs_cache[scene_id] = load_ins_locs(scene_id)
            ins_locs = scene_locs_cache[scene_id]

            orig_cands, orig_paths, _ = get_cands_and_paths(rec)
            evf_rec = evf_lookup.get((rec['scene_id'], rec['obj_id'], rec['caption'][:40]))

            try:
                new_paths, new_cands, tag = selector_fn(rec, ins_locs, evf_rec)
            except Exception as e:
                logging.warning('Selector error (%s, %s): %s', variant_name, scene_id, e)
                new_paths, new_cands, tag = orig_paths, orig_cands, 'error'

            tb     = rec['target_box']
            ora25  = int(best_oracle_iou(new_cands, ins_locs, tb) >= 0.25)
            ora50  = int(best_oracle_iou(new_cands, ins_locs, tb) >= 0.50)

            changed = (new_paths != orig_paths or new_cands != orig_cands)
            if changed and predictor is not None and new_paths:
                pred_inst = vlm_reselect(new_paths, new_cands, rec, predictor)
                if pred_inst is not None and 0 <= pred_inst < len(ins_locs):
                    pred_box = ins_locs[pred_inst].tolist()
                    iou   = calc_iou(tb, pred_box)
                    acc25 = int(iou >= 0.25)
                    acc50 = int(iou >= 0.50)
                else:
                    iou, acc25, acc50 = rec.get('iou', 0) or 0, 0, 0
                    pred_inst = -1
            else:
                iou       = rec.get('iou', 0) or 0
                acc25     = rec.get('acc25', 0) or 0
                acc50     = rec.get('acc50', 0) or 0
                pred_inst = (rec.get('trace') or {}).get('pred_instance', -1)

            row = {
                **{k: rec[k] for k in ('scene_id', 'obj_id', 'obj_name', 'caption',
                                        'target_box', 'unique') if k in rec},
                'iou':            iou,
                'acc25':          acc25,
                'acc50':          acc50,
                'oracle25':       ora25,
                'oracle50':       ora50,
                'n_cands_before': len(orig_cands),
                'n_cands_after':  len(new_cands),
                'same_best_frame': get_same_best_frame(orig_cands),
                'anchor_vis0':    get_anchor_vis0(rec, orig_cands, ins_locs),
                'pred_instance':  pred_inst,
                'tag':            tag,
            }
            results.append(row)
            fout.write(json.dumps(row) + '\n')

    metrics = compute_metrics(results, label=variant_name,
                               baseline_oracle25=baseline_oracle25)
    (out_dir / f'evidence_{variant_name}_metrics.json').write_text(
        json.dumps(metrics, indent=2))
    return metrics, results


# ── Logging setup ───────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, run_id: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'evidence_ablation_{run_id}.log'
    fmt = '%(asctime)s %(levelname)-7s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ]
    )
    return log_path


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ev_input', required=True)
    parser.add_argument('--evf_input', default=None)
    parser.add_argument('--variants', nargs='+',
                        default=['baseline', 'topk', 'anchor_vis', 'frame_div',
                                 'cond_overlay', 'distractor', 'full'])
    parser.add_argument('--out_dir',
                        default='experiments/full_ablation/outputs/evidence_ablation')
    parser.add_argument('--log_dir', default='experiments/full_ablation/logs')
    parser.add_argument('--run_id', default=None,
                        help='Run identifier. Defaults to YYYYMMDD_HHMMSS timestamp')
    parser.add_argument('--no_vlm', action='store_true')
    parser.add_argument('--vlm_model', default='local-qwen')
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--max_retry', type=int, default=3)
    args = parser.parse_args()

    run_id  = args.run_id or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = REPO_ROOT / args.log_dir
    log_path = setup_logging(log_dir, run_id)
    logging.info('Run ID   : %s', run_id)
    logging.info('Log file : %s', log_path)

    out_dir = REPO_ROOT / args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info('Out dir  : %s', out_dir)

    ev_records  = [json.loads(l) for l in open(args.ev_input)]
    evf_records = [json.loads(l) for l in open(args.evf_input)] if args.evf_input else []
    evf_lookup  = {(r['scene_id'], r['obj_id'], r['caption'][:40]): r for r in evf_records}
    logging.info('Loaded %d E_V records, %d E_VF records', len(ev_records), len(evf_records))

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
    baseline_oracle25 = None

    for variant in args.variants:
        if variant not in SELECTOR_REGISTRY:
            logging.warning('Unknown variant: %s, skipping', variant)
            continue
        logging.info('=== Running variant: %s ===', variant)
        m, _ = run_variant(
            variant,
            SELECTOR_REGISTRY[variant],
            ev_records,
            evf_lookup,
            predictor,
            scene_locs_cache,
            out_dir,
            baseline_oracle25=baseline_oracle25,
        )
        if variant == 'baseline':
            baseline_oracle25 = m['Oracle@25']
            m['OracleDrop'] = 0.0
        all_metrics.append(m)
        logging.info('  Acc@25=%.3f  Oracle@25=%.3f  AvgCandAfter=%.1f  SameBF=%.3f  AncVis0=%.3f',
                     m['Acc@25'], m['Oracle@25'], m['AvgCandAfter'],
                     m['SameBestFrameRate'], m['AnchorVis0Rate'])

    logging.info('=== RESULTS TABLE ===')
    print_table(all_metrics)

    summary_path = out_dir / 'evidence_ablation_summary.json'
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    logging.info('Saved -> %s', summary_path)


if __name__ == '__main__':
    main()
