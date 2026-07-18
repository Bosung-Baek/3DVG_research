"""
Baseline E0 — SeqVLM official pipeline (exact reproduction).

Source: SeqVLM (ACM Multimedia 2025)
Ref:  seqvlm/adaptive_predictor.py  +  prompts/prompt.py

Key implementation details matched to original:
1. SYSTEM_PROMPT: exact copy from prompts/prompt.py
2. USER_PROMPT: "Here are the images of {n_images} possible objects."
   where n_images = BATCH SIZE (not total candidates)
3. Tournament: group into max_batch_size, select winner per group, repeat
4. Retry loop: on parse failure, send IMAGE_ID_INVALID_PROMPT / WRONG_FORMAT_PROMPT
   in a multi-turn conversation (same as select_with_retry)
5. Candidate pool: use prop_indices from E0 trace (already class-filtered with seg_conf)
6. Canvas: full canvas.jpg (all frames stacked, red bbox pre-rendered)
"""
from __future__ import annotations
import base64, json, re
from pathlib import Path

from .base import VLMInputFormat, CANVAS_ROOT

# Exact copy from prompts/prompt.py
SYSTEM_PROMPT = """Imagine you are in a room and you are aksed to find one object.

Given a series of images and a query describing a specific object in the room, you need to analyze the images, and find an image that best fits the query.

Please note that each image is composed of sub-images displaying the object from multiple perspectives. In each sub-image, there is a red rectangle box highlighting the object, but the box may also contain other irrelevant objects. You need to make a selection by combining the object in the red rectangle box with surrounding environment from different perspective images.

Return the index of the image where the object is found, and describe the process of selecting this image.

Your response should be in the following format, and it should not include code block markers such as ```json.

{
  "process": "Explain the process of how you identified the room's features and located the target object",
  "image_id": 1 # Replace with the actual index based on the input order of images, starting from 0.
}

Here is an example for you.

```
Input:
Query: Find the black table that is surrounded by four chairs.
Here are the images of 3 possible objects.
[image_0, image_1, image_2]

Output:
{
  "process": "After carefully examining all the input images, I found only the tables in image_1, image_2 are black, but only the tables in image_2 has is surrounded by four chairs. So the correct object is the table in image_2",
  "image_id": 2
}

```

Here are some tips:
# Please follow the format of the example strictly
# If there is no object that fully matches the query, select the most suitable one.
# If the types of all objects are inconsistent with the query, output -1 in the value of image_id.

"""

# Exact copy from prompts/prompt.py  (n_images = batch size, set at call time)
USER_PROMPT = "Query: {query}\nHere are the images of {n_images} possible objects."

IMAGE_ID_INVALID_PROMPT = ("The image_id {image_id} you selected does not exist. "
    "Did you perhaps see it incorrectly? Please reconsider and select another image. "
    "Remember to reply using JSON format with the two keys \"process\", \"image_id\" as required before.")

WRONG_FORMAT_PROMPT = "The answer contains extra characters. Please follow the format of the example strictly."


def _canvas_b64(sid: str, inst_id: int) -> str | None:
    canvas = CANVAS_ROOT / str(sid) / str(inst_id) / "canvas.jpg"
    if canvas.exists():
        return base64.b64encode(canvas.read_bytes()).decode()
    return None


class BaselineE0Format(VLMInputFormat):
    name = "baseline_e0"
    source_paper = "SeqVLM"
    paper_faithful = True

    def build(self, query, scene_id, candidates, anchors,
              relation_info, frame_data, scene_data, config):
        images = []
        for inst_id in candidates:
            b64 = _canvas_b64(scene_id, inst_id)
            images.append(b64)   # None if canvas missing

        meta = self._base_metadata(
            uses_real_rgb=True,
            uses_multiview=True,
            uses_visual_prompt=True,
            candidate_canvas_type="full_canvas_5frame_vertical",
            object_marker="red_rectangle",
            view_selection="all_frames_precomputed",
            num_views_per_candidate=5,
            implementation_difference_from_paper=[],
        )
        # NOTE: prompt does NOT embed n_images here.
        # n_images is injected per-batch in run_tournament (= batch size).
        return {
            "images":   images,
            "system":   SYSTEM_PROMPT,
            "prompt":   USER_PROMPT,   # raw template with {query} and {n_images}
            "query":    query,         # stored separately for batch-time formatting
            "metadata": meta,
        }

    @staticmethod
    def select_with_retry(images_b64: list, query: str, model: str,
                          invoke_fn, max_retry: int = 3) -> int:
        """
        Exact reproduction of AdaptivePredictor.select_with_retry().
        Uses multi-turn conversation on parse failure.
        Returns local index or -1.
        """
        n = len(images_b64)
        user_text = USER_PROMPT.format(query=query, n_images=n)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                *[{"type": "image_url",
                   "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                 "detail": "high"}}
                  for b64 in images_b64],
            ]},
        ]

        guide_prompt = ""
        for retry in range(max_retry):
            if retry > 0 and guide_prompt:
                messages.append({"role": "user", "content": guide_prompt})
            try:
                output = invoke_fn(model, messages)
                raw = output["answer"]
                answer = json.loads(raw)
                image_id = answer.get("image_id", -2)
                if isinstance(image_id, int) and -1 <= image_id < n:
                    return image_id
                guide_prompt = IMAGE_ID_INVALID_PROMPT.format(image_id=image_id)
                messages.append({"role": "assistant", "content": raw})
            except Exception:
                guide_prompt = WRONG_FORMAT_PROMPT
                try:
                    messages.append({"role": "assistant",
                                     "content": output.get("answer","")})
                except Exception:
                    pass
        return -1

    @staticmethod
    def parse_response(raw: str, n_images: int) -> int:
        """Fallback parser (not used when select_with_retry is called directly)."""
        try:
            ans = json.loads(raw)
            iid = ans.get("image_id", -1)
            if isinstance(iid, int) and -1 <= iid < n_images:
                return iid
        except Exception:
            pass
        m = re.search(r'"image_id"\s*:\s*(-?\d+)', raw)
        if m:
            v = int(m.group(1))
            if -1 <= v < n_images:
                return v
        return -1
