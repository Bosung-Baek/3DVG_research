import psutil
import time
import argparse
import json
import os
import sys
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import *
from seqvlm import adaptive_predictor as adaptive_predictor_module
from seqvlm.adaptive_predictor import AdpativePredictor


def _install_local_qwen_fallback(args):
    if args.vlm_model != 'local-qwen':
        return
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if cuda_available:
        return

    from seqvlm.ablation import mock_vlm_answer

    original_invoke_api = adaptive_predictor_module.invoke_api

    def invoke_api(model, messages):
        if model == 'local-qwen':
            return mock_vlm_answer(messages)
        return original_invoke_api(model, messages)

    adaptive_predictor_module.invoke_api = invoke_api
    print('[evaluate] CUDA unavailable; using deterministic local-qwen fallback.')


if __name__ == '__main__':
    # add an argument
    parser = argparse.ArgumentParser(description='seqvlm scanrefer')
    parser.add_argument('--data_path', type=str, default='../data/scanrefer_250.json')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--vlm_model', type=str, default='doubao-vision')
    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--max_vlm_props', type=int, default=40)
    parser.add_argument('--ablation_axis', type=str, default='baseline',
                        choices=['baseline', 'parsing', 'viewpoint', 'input_format'])
    parser.add_argument('--input_format', type=str, default='seqvlm_canvas',
                        choices=['seqvlm_canvas', 'geo_candidate', 'geo_raw_frame', 'geo_bbox_overlay'])
    parser.add_argument('--view_source', type=str, default='seqvlm_canvas',
                        choices=['seqvlm_canvas', 'geo_frame_selection'])
    parser.add_argument('--geo_n_views', type=int, default=3)
    parser.add_argument('--geo_max_frames', type=int, default=80)
    parser.add_argument('--geo_evidence_dir', type=str, default='')
    parser.add_argument('--force_program_first', action='store_true')
    parser.add_argument('--output_path', type=str, default='')
    
    args = parser.parse_args()
    _install_local_qwen_fallback(args)
    
    with open(args.data_path, 'r') as f:
        eval_data = json.load(f)
    
    # load label map
    label_map_file = os.path.join(REPO_ROOT, 'data/scannetv2-labels.combined.tsv')
    labels_pd = pd.read_csv(label_map_file, sep='\t', header=0)
    
    correct_25 = 0
    correct_50 = 0
    unique_25 = 0
    unique_50 = 0
    
    total = 0
    vlm_total = 0
    unique_total = 0
    except_total = 0
    
    eps = 10 ** -6
    
    vlm_configs = {
        'image_path': args.image_path, 
        'vlm_model': args.vlm_model, 
        'max_retry': args.max_retry, 
        'max_batch_size': args.max_batch_size, 
        'max_vlm_props': args.max_vlm_props,
        'ablation_axis': args.ablation_axis,
        'input_format': args.input_format,
        'view_source': args.view_source,
        'force_program_first': args.force_program_first,
        'geo_n_views': args.geo_n_views,
        'geo_max_frames': args.geo_max_frames,
        'geo_evidence_dir': args.geo_evidence_dir or os.path.join(
            REPO_ROOT, 'experiments/ablation_seqvlm_3scenes/outputs/geo_evidence', args.exp_name
        ),
    }        
        
    predictor = AdpativePredictor(**vlm_configs)
    logger = create_logger(args.exp_name)
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        open(args.output_path, 'w').close()


    
    
    for i, task in enumerate(eval_data):
        scene_id, obj_id, caption, prog_str, obj_name = task.values()
        # print(task)
        
        print('Case:', i)
        print('scene_id:', scene_id)
        print('caption:', caption)
        print('obj_id:', obj_id)
        print('obj_name:', obj_name)
        
        # load point cloud data
        obj_ids, obj_labels, obj_locs = load_pc(scene_id)
        mapped_class_ids = []

        for obj_label in obj_labels:
            label_ids = labels_pd[labels_pd['raw_category'] == obj_label]['nyu40id']
            label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
            mapped_class_ids.append(label_id)

        index = obj_ids.index(int(obj_id))
        target_box = obj_locs[index]
        target_class_id = mapped_class_ids[index]
                
        total += 1
        unique = (np.array(mapped_class_ids) == target_class_id).sum() == 1
        if unique:
            unique_total += 1

        pred_box, use_vlm = predictor.execute(scene_id, obj_name, caption, prog_str)
        iou = 0.0
        
        if use_vlm:
            vlm_total += 1
            
        if pred_box is None:
            except_total += 1
        else:
            iou = calc_iou(pred_box, target_box)
            print(f'IoU: {iou:.2f}')
            
            if iou >= 0.25:
                correct_25 += 1
                if unique:
                    unique_25 += 1
            if iou >= 0.5:
                correct_50 += 1
                if unique:
                    unique_50 += 1

        accuracy_msgs = [
            'Overall@25: {:.3f}'.format(correct_25 / total), 
            'Overall@50: {:.3f}'.format(correct_50 / total), 
            'Unique@25: {:.3f}'.format(unique_25 / (unique_total + eps)), 
            'Unique@50: {:.3f}'.format(unique_50 / (unique_total + eps)), 
            'Multiple@25: {:.3f}'.format((correct_25 - unique_25) / (total - unique_total + eps)), 
            'Multiple@50: {:.3f}'.format((correct_50 - unique_50) / (total - unique_total + eps)), 
            'Unique Ratio: {} / {}'.format(unique_25, unique_total), 
            'Multiple Ratio: {} / {}'.format(correct_25 - unique_25, total - unique_total), 
            'Except Ratio: {} / {}'.format(except_total, total), 
            'VLM Usage Ratio: {} / {}'.format(vlm_total, total), 
            '\n'
        ]
        print('\n'.join(accuracy_msgs))
        if args.output_path:
            record = {
                'case': i,
                'scene_id': scene_id,
                'obj_id': int(obj_id),
                'obj_name': obj_name,
                'caption': caption,
                'ablation_axis': args.ablation_axis,
                'input_format': args.input_format,
                'view_source': args.view_source,
                'force_program_first': bool(args.force_program_first),
                'use_vlm': bool(use_vlm),
                'pred_box': pred_box.tolist() if pred_box is not None else None,
                'target_box': target_box.tolist(),
                'iou': float(iou),
                'acc25': bool(iou >= 0.25),
                'acc50': bool(iou >= 0.5),
                'unique': bool(unique),
                'trace': predictor.last_trace,
            }
            with open(args.output_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        
        if (i + 1) % 10 == 0:
            for msg in accuracy_msgs:
                logger.info(msg)

    

    
