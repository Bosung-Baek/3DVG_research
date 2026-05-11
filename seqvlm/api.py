import io
import os
import yaml
from pathlib import Path
from openai import OpenAI
from mmengine.utils.dl_utils import TimeCounter


_REPO_ROOT = Path(__file__).resolve().parents[1]

with open(_REPO_ROOT / "config.yaml", "r") as f:
    configs = yaml.safe_load(f)

# ── Local Qwen2-VL-7B inference (no API server needed) ───────────────────────
_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None

def _get_local_model():
    global _LOCAL_MODEL, _LOCAL_PROCESSOR
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL, _LOCAL_PROCESSOR
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    LOCAL_QWEN_PATH = "/home/knuvi/bosung/models/Qwen2-VL-7B-Instruct"
    model_id = LOCAL_QWEN_PATH if os.path.isdir(LOCAL_QWEN_PATH) else "Qwen/Qwen2-VL-7B-Instruct"
    print("[api] Loading local Qwen2-VL-7B …")
    _LOCAL_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(model_id)
    print("[api] Local model loaded.")
    return _LOCAL_MODEL, _LOCAL_PROCESSOR


def _invoke_local_qwen(messages):
    import base64, re, torch
    from qwen_vl_utils import process_vision_info

    model, processor = _get_local_model()

    # Convert OpenAI-format messages to Qwen format
    qwen_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            qwen_content = []
            for part in content:
                if part["type"] == "text":
                    qwen_content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    qwen_content.append({"type": "image", "image": url})
            qwen_messages.append({"role": role, "content": qwen_content})

    text_input = processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(qwen_messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    return {
        "answer": answer,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "completion_tokens": len(trimmed[0]),
    }


def invoke_api(model, messages):
    if model == "mock-first":
        from seqvlm.ablation import mock_vlm_answer
        return mock_vlm_answer(messages)

    if model == "local-qwen":
        return _invoke_local_qwen(messages)

    base_url, model, api_key = configs[model].values()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    with TimeCounter(tag="Invoke"):
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        result = {
            'answer': response.choices[0].message.content,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens
        }
        return result
    
    
if __name__ == '__main__':
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Who are you?'},
            ],
        }
    ]
    print(invoke_api('gpt-proxy', messages))
