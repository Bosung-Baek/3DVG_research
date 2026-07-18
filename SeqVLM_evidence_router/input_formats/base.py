"""Base class for all paper-faithful VLM input formats."""
from __future__ import annotations
import base64, io, os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
CANVAS_ROOT  = REPO / "data/scanrefer_preprocessed"
SCANNET_ROOT = Path("/data/knuvi/bosung/scannet")
MASK3D_DIR   = "/data/knuvi/bosung/Mask3d/scannet200"


def _img_b64(path: Path | str, fmt="JPEG") -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def _pil_b64(img: Image.Image, fmt="JPEG", quality=88) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _best_frame(inst_dir: Path, th: int = 480) -> Image.Image | None:
    """Pick highest-variance frame from frame_*.jpg files."""
    frames = sorted(inst_dir.glob("frame_*.jpg"))
    if not frames:
        canvas = inst_dir / "canvas.jpg"
        if canvas.exists():
            img = Image.open(canvas).convert("RGB")
            w, h = img.size
            n = h // th
            if n == 0:
                return img
            best = None; bs = -1
            for i in range(n):
                t = img.crop((0, i*th, w, (i+1)*th))
                s = float(np.var(np.array(t, dtype=float)))
                if s > bs:
                    bs = s; best = t
            return best
        return None
    best = None; bs = -1
    for fp in frames:
        try:
            img = Image.open(fp).convert("RGB")
            s = float(np.var(np.array(img, dtype=float)))
            if s > bs:
                bs = s; best = img
        except Exception:
            pass
    return best


def parse_selected_key(raw: str, valid_keys: set) -> str | None:
    """
    Unified response parser for all formats.
    All formats use: {"process": "...", "selected_key": "A"}
    Falls back to any valid key found in response.
    """
    import json as _json, re as _re
    try:
        ans = _json.loads(raw)
        key = str(ans.get("selected_key", "")).strip().upper()
        if key in valid_keys:
            return key
    except Exception:
        pass
    # Fallback regex
    m = _re.search(r'"selected_key"\s*:\s*"([A-Z0-9])"', raw, _re.IGNORECASE)
    if m and m.group(1).upper() in valid_keys:
        return m.group(1).upper()
    # Last resort: first valid letter/digit in response
    for ch in raw.upper():
        if ch in valid_keys:
            return ch
    return None


UNIFIED_RESPONSE_FORMAT = '''{
  "process": "brief reasoning",
  "selected_key": "A",
  "confidence": 0.0
}'''


class VLMInputFormat(ABC):
    name: str = "base"
    source_paper: str = "unknown"
    paper_faithful: bool | str = False

    @abstractmethod
    def build(
        self,
        query: str,
        scene_id: str,
        candidates: list[int],         # Mask3D instance indices
        anchors: list[int],             # anchor instance indices
        relation_info: dict,
        frame_data: dict,               # {inst_id: list[frame_path]}
        scene_data: dict,               # locs, labels, etc.
        config: dict,
    ) -> dict:
        """Return {"images": [b64,...], "prompt": str, "metadata": dict}."""

    def _base_metadata(self, **kw) -> dict:
        meta = {
            "input_format": self.name,
            "source_paper": self.source_paper,
            "paper_faithful": self.paper_faithful,
            "faithfulness_note": "",
            "uses_real_rgb": False,
            "uses_rendered_image": False,
            "uses_spatial_text": False,
            "uses_visual_prompt": False,
            "uses_multiview": False,
            "uses_query_decomposition": False,
            "uses_multiscale": False,
            "implementation_difference_from_paper": [],
        }
        meta.update(kw)
        return meta
