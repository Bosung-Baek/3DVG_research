import glob
import json
import os
import random
from objprint import op
from colorama import Fore, init
init(autoreset=True)

from prompts.prompt import *
from utils import *
from api import invoke_api
from visprog.program import ProgramInterpreter
from seqvlm.feat_handler import VisualFeatHandler
from seqvlm.ablation import (
    build_user_prompt,
    configure_viewpoint_mode,
    geo_structured_to_program,
)
from seqvlm.geo_evidence import build_geo_evidence_images


class AdpativePredictor:
    
    def __init__(self, **kwargs):        
        self.image_path = kwargs.get('image_path')
        self.max_retry = kwargs.get('max_retry')
        self.vlm_model = kwargs.get('vlm_model')
        self.max_batch_size = kwargs.get('max_batch_size')
        self.max_vlm_props = kwargs.get('max_vlm_props')
        self.ablation_axis = kwargs.get('ablation_axis', 'baseline')
        self.input_format = kwargs.get('input_format', 'seqvlm_canvas')
        self.view_source = kwargs.get('view_source', 'seqvlm_canvas')
        self.viewpoint_mode = kwargs.get('viewpoint_mode', 'seqvlm_ego')
        self.force_program_first = kwargs.get('force_program_first', False)
        self.geo_n_views = kwargs.get('geo_n_views', 3)
        self.geo_max_frames = kwargs.get('geo_max_frames', 80)
        self.geo_evidence_dir = kwargs.get('geo_evidence_dir', '../experiments/ablation_seqvlm_3scenes/outputs/geo_evidence')
        self.last_trace = {}
        
        self.prog_inter = ProgramInterpreter()
        self.handler = VisualFeatHandler.get_instance()

    
    def execute(self, scene_id, obj_name, caption, prog_str):
        self.last_trace = {
            'scene_id': scene_id,
            'obj_name': obj_name,
            'caption': caption,
            'ablation_axis': self.ablation_axis,
            'input_format': self.input_format,
            'view_source': self.view_source,
            'viewpoint_mode': self.viewpoint_mode,
            'used_program': prog_str,
            'geo_parse': None,
            'n_props': 0,
            'pred_source': None,
        }
        if self.ablation_axis == 'parsing':
            prog_str, geo_parse = geo_structured_to_program(caption, obj_name, prog_str)
            self.last_trace['used_program'] = prog_str
            self.last_trace['geo_parse'] = geo_parse

        configure_viewpoint_mode(
            'geo_world' if self.ablation_axis == 'viewpoint' else 'seqvlm_ego'
        )
        ins_labels, ins_locs, ins_scores, _ = load_seg_inst(scene_id)
        pred_cls = self.handler.predict_obj_class(obj_name, ins_labels)
        
        index = []
        prop_images = []
        seg_conf_score = 0.2
        
        for i, label in enumerate(ins_labels):
            if ins_scores[i] > seg_conf_score and label == pred_cls:
            # if label == pred_cls:
                canvas = os.path.join(self.image_path, scene_id, str(i), 'canvas.jpg')
                if os.path.exists(canvas):
                    index.append(i)
                    prop_images.append(canvas)
    
        n_props = len(prop_images)
        self.last_trace['n_props'] = n_props
        self.last_trace['prop_indices'] = index
        print('Prop Images:', n_props, prop_images[:20])

        if self.view_source == 'geo_frame_selection' and index:
            geo_images, geo_trace = build_geo_evidence_images(
                scene_id=scene_id,
                candidate_indices=index,
                caption=caption,
                obj_name=obj_name,
                ins_labels=ins_labels,
                ins_locs=ins_locs,
                ins_scores=ins_scores,
                class_predictor=self.handler.predict_obj_class,
                out_dir=self.geo_evidence_dir,
                input_format=self.input_format,
                n_views=self.geo_n_views,
                max_frames=self.geo_max_frames,
            )
            self.last_trace['geo_evidence'] = geo_trace
            if geo_images:
                prop_images = geo_images
                n_props = len(prop_images)
                self.last_trace['n_geo_props'] = n_props
                self.last_trace['geo_prop_images'] = prop_images
        
        if not self.force_program_first and n_props <= self.max_vlm_props:
            pred = self.predict(prop_images, caption)
            if pred is not None:
                self.last_trace['pred_source'] = 'vlm'
                self.last_trace['pred_index'] = int(pred)
                pred_inst = index[pred] if pred < len(index) else index[0]
                self.last_trace['pred_instance'] = int(pred_inst)
                return ins_locs[pred_inst], True

        try:
            init_state = {'scene_id': scene_id}
            result = self.prog_inter.execute(prog_str, init_state)
            if result:
                self.last_trace['pred_source'] = 'program'
                self.last_trace['pred_instance'] = int(result[0]['obj_id'])
                return result[0]['obj_loc'], False
        except Exception as e:
            print(Fore.RED + f'Except: {e}')
            self.last_trace['error'] = str(e)
            
        self.last_trace['pred_source'] = 'none'
        return None, False

    
    def predict(self, prop_images, caption):
        max_batch_size = self.max_batch_size
        base64Images = [encode_image_to_base64(image) for image in prop_images]
        
        remain_props = [(i, image) for i, image in enumerate(base64Images)]
        while len(remain_props) > 1:        
            prop_image_groups = []
            for i in range(0, len(remain_props), max_batch_size):
                g = remain_props[i:i + max_batch_size]
                prop_image_groups.append(g)
            
            remain_props = []
            for i, images in enumerate(prop_image_groups):
                image_id = self.select_with_retry(images, caption)
                if image_id != -1:
                    remain_props.append(images[image_id])
                    
        return remain_props[0][0] if remain_props else None
    
    
    def select_with_retry(self, images, caption):
        n_images = len(images)
        assert n_images > 0
        # return random.randint(-1, n_images - 1)
        
        user_prompt = build_user_prompt(caption, n_images, self.input_format)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x[1]}",
                                "detail": "high",
                            },
                        },
                        images,
                    ),
                ]
            }
        ]
        
        retry = 0
        while retry < self.max_retry:
            try:
                output = invoke_api(self.vlm_model, messages)
                print('Vision Lang Model Output:')
                op(output)
                answer = json.loads(output['answer'])
                image_id = answer['image_id']
                if isinstance(image_id, int) and -1 <= image_id < n_images:
                    return image_id
                
                guide_prompt = IMAGE_ID_INVALID_PROMPT.format(image_id=image_id)
            except Exception as e:
                print(Fore.RED + f'Except: {e}')
                guide_prompt = WRONG_FORMAT_PROMPT
                
            vlm_message = {
                'role': 'assistant', 
                'content': output['answer']
            }
            messages.append(vlm_message)
            messages.append(
                {"role": "user", "content": guide_prompt}
            )
            retry += 1
    
        return -1
