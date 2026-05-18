import base64
from collections import defaultdict
import json
import logging
import numpy as np
import os
import torch
from pathlib import Path

from PIL import Image
from io import BytesIO


_SCANNET_ORIGIN_DIR = '/data/knuvi/bosung/scannet_origin'
_MASK3D_DIR         = '/data/knuvi/bosung/Mask3d/scannet200'


def load_seg_inst(scene_id):
    root_dir = _MASK3D_DIR
    data = np.load(os.path.join(root_dir, scene_id + '.npz'), allow_pickle=True)
    ins_labels = list(data['ins_labels'])
    ins_scores = [float(x) for x in data['ins_scores']]
    
    ins_locs = []
    scene_pc = []
    for obj in data['ins_pcds']:
        if obj.shape[0] == 0:
            obj = np.zeros((1, 6))
        obj_pcd = obj[:, :3]
        scene_pc.append(obj_pcd)
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        ins_locs.append(np.concatenate([obj_center, obj_size], 0))

    scene_pc = np.concatenate(scene_pc, 0)
    center = (scene_pc.max(0) + scene_pc.min(0)) / 2
    
    return ins_labels, ins_locs, ins_scores, center


def load_pc(scene_id):
    """
    Load GT instance bounding boxes from ScanNet aggregation + .ply files.
    Replaces the original referit3d .pth dependency.
    Returns: (obj_ids, obj_labels, inst_locs)
      obj_ids   : list[int]      — objectId from aggregation.json
      obj_labels: list[str]      — class label per object
      inst_locs : list[np.array] — [cx, cy, cz, wx, wy, wz] axis-aligned bbox
    """
    from plyfile import PlyData
    scene_dir = os.path.join(_SCANNET_ORIGIN_DIR, scene_id)

    # Load axis alignment
    txt_path = os.path.join(scene_dir, f'{scene_id}.txt')
    axis_align = np.eye(4)
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    vals = [float(x) for x in line.split('=')[1].split()]
                    axis_align = np.array(vals).reshape(4, 4)
                    break

    # Load point cloud
    ply_path = os.path.join(scene_dir, f'{scene_id}_vh_clean_2.ply')
    plydata = PlyData.read(ply_path)
    verts = np.stack([plydata['vertex']['x'],
                      plydata['vertex']['y'],
                      plydata['vertex']['z']], axis=1).astype(np.float32)
    # Apply axis alignment
    ones = np.ones((verts.shape[0], 1), dtype=np.float32)
    verts_h = np.concatenate([verts, ones], axis=1)
    verts = (axis_align @ verts_h.T).T[:, :3]

    # Load segmentation (vertex → segment index)
    segs_path = os.path.join(scene_dir, f'{scene_id}_vh_clean_2.0.010000.segs.json')
    with open(segs_path) as f:
        segs_data = json.load(f)
    seg_indices = np.array(segs_data['segIndices'])  # (N_verts,)

    # Load aggregation (segment → object)
    agg_path = os.path.join(scene_dir, f'{scene_id}.aggregation.json')
    with open(agg_path) as f:
        agg = json.load(f)

    skip_labels = {'wall', 'floor', 'ceiling'}
    obj_ids, obj_labels, inst_locs = [], [], []

    for group in agg['segGroups']:
        obj_id  = int(group['objectId'])
        label   = group['label']
        if label in skip_labels:
            continue
        seg_ids = set(group['segments'])
        mask = np.isin(seg_indices, list(seg_ids))
        if mask.sum() == 0:
            continue
        pts = verts[mask]
        obj_center = (pts.max(0) + pts.min(0)) / 2
        obj_size   = pts.max(0) - pts.min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size]).astype(np.float32))
        obj_ids.append(obj_id)
        obj_labels.append(label)

    return obj_ids, obj_labels, inst_locs


def calc_iou(box_a, box_b):
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    
    return 1.0 * intersection / union


def create_logger(exp_name):
    # Create a custom logger
    logger = logging.getLogger('seq-vlm')

    # Set the log level of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Create handlers for writing to a file and logging to console
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = repo_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f'{exp_name}.log', mode='w')
    console_handler = logging.StreamHandler()

    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger


def encode_image_to_base64(image_path):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            buf = BytesIO()
            image.save(buf, format='JPEG')
            byte_data = buf.getvalue()
            base64_str = base64.b64encode(byte_data).decode('utf-8')
            return base64_str
    except Exception:
        return None
