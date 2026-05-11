
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoTokenizer, 
    CLIPModel,
    Blip2Processor, 
    Blip2ForConditionalGeneration
)

class VisualFeatHandler:

    __instance = None
    
    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
    
    def __init__(self):
        # Use ViT-B/32 which is cached locally; patch16 is not available offline
        tokenizer_path = 'openai/clip-vit-base-patch32'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       model_max_length=512,
                                                       use_fast=True,
                                                       clean_up_tokenization_spaces=True)
        self.clip = CLIPModel.from_pretrained(tokenizer_path).cuda()
        # BLIP2 loaded lazily only when judge_consistency is called
        self._blip2_loaded = False
        self.processor = None
        self.model = None
        
    
    def predict_obj_class(self, obj_name, ins_labels) -> str:
        class_list = list(set(ins_labels))
        class_tokens = self.tokenizer([f'a {class_name} in a scene' for class_name in class_list],
                                      padding=True,
                                      return_tensors='pt')
        for name in class_tokens.data:
            class_tokens.data[name] = class_tokens.data[name].cuda()
        
        label_feats = self.clip.get_text_features(**class_tokens)
        # Newer transformers may return a ModelOutput — extract the tensor
        if not isinstance(label_feats, torch.Tensor):
            label_feats = label_feats.pooler_output
        label_feats = label_feats / label_feats.norm(p=2, dim=-1, keepdim=True)

        query_tokens = self.tokenizer([f'a {obj_name} in a scene'], padding=True, return_tensors='pt')
        for name in query_tokens.data:
            query_tokens.data[name] = query_tokens.data[name].cuda()

        query_feats = self.clip.get_text_features(**query_tokens)
        if not isinstance(query_feats, torch.Tensor):
            query_feats = query_feats.pooler_output
        query_feats = query_feats / query_feats.norm(p=2, dim=-1, keepdim=True)

        pred_scores = torch.matmul(query_feats, label_feats.t())
        pred_cls_idx = pred_scores.argmax(dim=-1)[0]
        pred_cls = class_list[pred_cls_idx]
        return pred_cls
    
    
    def _load_blip2(self):
        if not self._blip2_loaded:
            model_path = "Salesforce/blip2-flan-t5-xl"
            self.processor = Blip2Processor.from_pretrained(model_path, clean_up_tokenization_spaces=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
            self._blip2_loaded = True

    def judge_consistency(self, obj_name, images, ratio=0.25) -> bool:
        if len(images) == 0:
            return False
        self._load_blip2()
        prompt = [f"Question: Is there a {obj_name}? Answer:"] * len(images)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return answer.count('yes') / len(answer) >= ratio
        

if __name__ == '__main__':
    pass