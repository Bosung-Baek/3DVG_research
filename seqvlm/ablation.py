import json
import os
import re
from datetime import datetime


ARCHIVE_DIR = "../experiments/ablation_seqvlm_3scenes"


def append_command_record(title, command, purpose, result, cwd=None, archive_dir=ARCHIVE_DIR):
    """Append a compact command record to the ablation archive."""
    path = os.path.join(archive_dir, "COMMANDS.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with open(path, "a") as f:
        f.write(f"\n### {title}\n\n")
        f.write(f"- Time: {stamp}\n")
        if cwd:
            f.write(f"- CWD: `{cwd}`\n")
        f.write("- Command:\n\n")
        f.write("```bash\n")
        f.write(command.rstrip() + "\n")
        f.write("```\n\n")
        f.write(f"- Purpose: {purpose}\n")
        f.write(f"- Result: {result}\n")


def parse_geo_structured(caption, obj_name):
    """Small geo-style structured parser for controlled 3-scene ablations.

    This is intentionally deterministic. It extracts the target, a coarse
    relation, and one anchor phrase, then lets SeqVLM's existing interpreters
    execute the resulting program.
    """
    text = caption.lower()
    target = _clean_phrase(obj_name)
    relation = None
    anchor = ""

    relation_patterns = [
        ("LEFT", [r"to the left of ([^.]+)", r"left of ([^.]+)", r"west of ([^.]+)"]),
        ("RIGHT", [r"to the right of ([^.]+)", r"right of ([^.]+)", r"east of ([^.]+)"]),
        ("HIGHER", [r"above ([^.]+)", r"over ([^.]+)"]),
        ("LOWER", [r"below ([^.]+)", r"under ([^.]+)", r"beneath ([^.]+)"]),
        ("CLOSEST", [r"next to ([^.]+)", r"near ([^.]+)", r"beside ([^.]+)", r"close to ([^.]+)"]),
        ("BEHIND", [r"behind ([^.]+)"]),
        ("FRONT", [r"in front of ([^.]+)"]),
        ("BETWEEN", [r"between ([^.]+)", r"in between ([^.]+)"]),
    ]

    for rel, patterns in relation_patterns:
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                relation = rel
                anchor = _clean_phrase(m.group(1))
                break
        if relation:
            break

    if "middle" in text or "center" in text:
        relation = relation or "MID"
    if any(w in text for w in ["smallest", "smaller"]):
        relation = relation or "MIN_SIZE"
    if any(w in text for w in ["largest", "biggest", "tallest", "taller"]):
        relation = relation or "MAX_SIZE"

    return {
        "target": target,
        "relation": relation or "LOC",
        "anchor": anchor,
        "anchor_type": "object" if anchor else "",
    }


def geo_structured_to_program(caption, obj_name, original_program):
    parsed = parse_geo_structured(caption, obj_name)
    target = parsed["target"]
    relation = parsed["relation"]
    anchor = parsed["anchor"]

    lines = [f"BOX0=LOC(object='{_escape(target)}')"]
    if relation == "LOC":
        return "\n".join(lines), parsed
    if relation == "MID":
        lines.append("TARGET=MID(targets=BOX0)")
        return "\n".join(lines), parsed
    if relation == "MIN_SIZE":
        lines.append("SIZES=CALC_SIZE(targets=BOX0)")
        lines.append("TARGET=MIN(data=SIZES)")
        return "\n".join(lines), parsed
    if relation == "MAX_SIZE":
        lines.append("SIZES=CALC_SIZE(targets=BOX0)")
        lines.append("TARGET=MAX(data=SIZES)")
        return "\n".join(lines), parsed
    if not anchor:
        return "\n".join(lines), parsed

    lines.append(f"BOX1=LOC(object='{_escape(anchor)}')")
    lines.append(f"TARGET={relation}(targets=BOX0, anchors=BOX1)")
    return "\n".join(lines), parsed


def build_user_prompt(caption, n_images, input_format="seqvlm_canvas"):
    if input_format in {"geo_candidate", "geo_raw_frame", "geo_bbox_overlay"}:
        letters = ", ".join(chr(ord("A") + i) for i in range(n_images))
        visual = ""
        if input_format == "geo_bbox_overlay":
            visual = (
                " Colored rectangles labeled A bbox, B bbox, and so on mark "
                "the selectable candidate objects. Yellow Anchor boxes, when "
                "present, are context only and must not be selected."
            )
        elif input_format == "geo_raw_frame":
            visual = " Images are raw selected scene frames without candidate boxes."
        return (
            f"Query: {caption}\n"
            f"There are {n_images} candidate objects: {letters}. Each image is "
            "one candidate evidence bundle, ordered by candidate index from 0. "
            "Compare candidates against the query, including object identity, "
            "spatial context, and visible surroundings."
            f"{visual} Return JSON with keys process and image_id."
        )
    return f"Query: {caption}\nHere are the images of {n_images} possible objects."


def mock_vlm_answer(messages):
    return {
        "answer": json.dumps({
            "process": "mock-first smoke test selection",
            "image_id": 0,
        }),
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def configure_viewpoint_mode(mode):
    if mode:
        os.environ["SEQVLM_VIEWPOINT_MODE"] = mode


def _clean_phrase(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 /_-]+", " ", text)
    stop = {
        "the", "a", "an", "this", "that", "it", "is", "are", "and", "with",
        "on", "at", "in", "of", "room", "wall", "side",
    }
    words = [w for w in text.split() if w not in stop]
    return " ".join(words[:6]).strip() or str(text).strip()


def _escape(text):
    return str(text).replace("\\", "\\\\").replace("'", "\\'")
