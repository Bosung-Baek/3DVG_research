#!/usr/bin/env python3
"""Evaluate replacing the dictionary evidence router with an LLM router.

The LLM applies the same conservative priority order as the dictionary router,
but judges the evidence requirements from the query text and existing
query-type metadata. The selected route is then evaluated by recomposing
completed source outputs, matching the final dictionary-router evaluation style.

Use --mock-dictionary for a no-API smoke run. For actual LLM routing, provide
an existing SeqVLM-style config alias or set OPENROUTER_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from query_type_router import route_for_row_universal  # noqa: E402

DEFAULT_INPUTS = REPO / "inputs"
DEFAULT_OUT = REPO / "experiments/ablation/llm_router"
ROUTES = {"E0", "seeground_ablation_spatial_only", "bev_raw_labeled", "seeground_ablation_3dpos_only"}
DEFAULT_CONFIG = Path("/home/knuvi/bosung/SeqVLM/config.yaml")


PROMPT_POLICY = "priority_decision_tree_v1"


SYSTEM_PROMPT = """You are a routing classifier for 3D visual grounding.
Choose exactly one VLM input route for the query.

Apply the rules in order. Stop at the first matching rule.

Priority rules:
1. If the query is mainly about proximity or distance, choose seeground_ablation_spatial_only.
   This includes near, next to, closest, farthest, by itself, isolated, not near, beside, around, close to, far from.
2. Otherwise, if the query requires visual evidence, choose E0.
   Visual evidence includes color, material, texture, shape, object state, open/closed, dirty/clean, bag/note/towel,
   things on top, clutter, contents, count of visible objects, or any mixed visual+spatial description.
3. Otherwise, if the query is a pure ordinal or global-layout relation, choose bev_raw_labeled.
   This includes middle, leftmost/rightmost, first/second/third, frontmost/backmost, top/bottom order, with no visual attributes.
4. Otherwise, if the query is a pure geometric/topological relation, choose seeground_ablation_3dpos_only.
   This includes between, inside, under, above, below, surrounded by, with no visual attributes.
5. Otherwise, if the query is viewpoint/local-frame dependent, choose E0.
   Current BEV is global XY, so viewer-centered left/right/front/back should stay with E0.
6. Otherwise choose E0.

Routes:
- E0: RGB multiview canvas. Use for visual attributes, object state, mixed evidence, viewpoint/local-frame descriptions, ambiguous queries.
- seeground_ablation_spatial_only: text-only spatial distances/offsets. Use for proximity, near/far/closest/farthest/next-to distance relations.
- bev_raw_labeled: top-down BEV image plus coordinates. Use only for pure ordinal/global layout relations without visual attributes.
- seeground_ablation_3dpos_only: 3D position text. Use only for pure geometric/topological relations such as between/inside/under without visual attributes.

Return compact JSON only:
{"route":"<one route>","reason":"which priority rule matched"}"""

USER_TEMPLATE = """Dataset: {dataset}
Existing query type: {query_type}
Query: {query}

Choose the route."""


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    base_url: str
    model: str


def load_simple_yaml_mapping(path: Path) -> dict:
    """Load the subset of YAML used by SeqVLM config files.

    PyYAML is preferred when available. The fallback handles nested mappings of
    scalar strings so the API key can be loaded without adding a dependency.
    """
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text()) or {}
    except ModuleNotFoundError:
        root: dict = {}
        stack: list[tuple[int, dict]] = [(-1, root)]
        for raw_line in path.read_text().splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, sep, value = line.strip().partition(":")
            if not sep:
                continue
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            value = value.strip()
            if value == "":
                child: dict = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = value.strip("'\"")
        return root


def resolve_api_config(args: argparse.Namespace) -> ApiConfig | None:
    if args.mock_dictionary:
        return None

    config_data = {}
    config_path = args.config
    if config_path and config_path.exists():
        config_data = load_simple_yaml_mapping(config_path)
    elif args.config:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    alias_cfg = {}
    if config_data:
        alias_cfg = config_data.get(args.config_alias, {})
        if not isinstance(alias_cfg, dict):
            raise ValueError(f"Config alias is not a mapping: {args.config_alias}")

    api_key = os.environ.get("OPENROUTER_API_KEY") or alias_cfg.get("api-key") or alias_cfg.get("api_key")
    base_url = (
        os.environ.get("OPENROUTER_BASE_URL")
        or alias_cfg.get("base-url")
        or alias_cfg.get("base_url")
        or "https://openrouter.ai/api/v1"
    )
    model = args.model or os.environ.get("OPENROUTER_MODEL") or alias_cfg.get("model") or "qwen/qwen-2.5-72b-instruct"

    if not api_key:
        raise RuntimeError(
            "No API key found. Provide --config with an alias containing api-key, "
            "set OPENROUTER_API_KEY, or use --mock-dictionary."
        )
    return ApiConfig(api_key=str(api_key), base_url=str(base_url), model=str(model))


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: dict) -> int:
    for key in ("case", "case_id", "query_id"):
        if key in row:
            return int(str(row[key]).lstrip("0") or "0")
    raise KeyError(f"No case id in row: {sorted(row)}")


def success25(row: dict) -> bool:
    if "acc25" in row:
        return bool(row["acc25"])
    if "success_iou25" in row:
        return bool(row["success_iou25"])
    return float(row.get("iou", 0) or 0) >= 0.25


def success50(row: dict) -> bool:
    if "acc50" in row:
        return bool(row["acc50"])
    if "success_iou50" in row:
        return bool(row["success_iou50"])
    return float(row.get("iou", 0) or 0) >= 0.50


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    recover = sum((not r["e0_acc25"]) and r["acc25"] for r in rows)
    regress = sum(r["e0_acc25"] and not r["acc25"] for r in rows)
    return {
        "num_queries": n,
        "acc_iou25": round(sum(r["acc25"] for r in rows) / n, 4),
        "acc_iou50": round(sum(r["acc50"] for r in rows) / n, 4),
        "mean_iou": round(sum(float(r.get("iou", 0) or 0) for r in rows) / n, 4),
        "route_counts": dict(Counter(r["route"] for r in rows)),
        "source_unavailable_fallbacks": sum(bool(r.get("source_unavailable_fallback_e0")) for r in rows),
        "e0_recoveries": recover,
        "e0_regressions": regress,
        "net_vs_e0": recover - regress,
    }


def parse_llm_json(raw: str) -> tuple[str, str]:
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*?\}", raw, flags=re.S)
        obj = json.loads(match.group(0)) if match else {}
    route = str(obj.get("route", "")).strip()
    reason = str(obj.get("reason", "")).strip()
    if route not in ROUTES:
        for candidate in ROUTES:
            if candidate in raw:
                route = candidate
                break
    if route not in ROUTES:
        route = "E0"
        reason = reason or "parse_failed_default_e0"
    return route, reason


def call_openrouter(
    query: str,
    query_type: str,
    dataset: str,
    api_config: ApiConfig,
    max_retry: int = 3,
) -> tuple[str, str, str]:
    payload = {
        "model": api_config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(dataset=dataset, query_type=query_type, query=query)},
        ],
        "temperature": 0,
        "max_tokens": 96,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        chat_completions_url(api_config.base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {api_config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/seqvlm-evidence-router",
            "X-Title": "SeqVLM Evidence Router",
        },
        method="POST",
    )
    last_error = ""
    for attempt in range(max_retry):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            raw = obj["choices"][0]["message"]["content"]
            route, reason = parse_llm_json(raw)
            return route, reason, raw
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            time.sleep(2.0 * (attempt + 1))
    return "E0", f"api_error_default_e0:{last_error[:160]}", ""


def build_items(input_dir: Path) -> list[dict]:
    items = []
    scan_e0 = {row_id(r): r for r in load_jsonl(input_dir / "scanrefer/full_E0_baseline_qwen72b.jsonl")}
    scan_labels = {row_id(r): r for r in load_jsonl(input_dir / "scanrefer/bev_raw_labeled/results.jsonl")}
    for case in sorted(scan_e0):
        label = scan_labels[case]
        row = {
            "dataset": "scanrefer",
            "case": case,
            "query_type": label.get("query_type", "uncategorized"),
            "query": label.get("query") or label.get("caption") or scan_e0[case].get("caption", ""),
        }
        _, dict_route, dict_reason = route_for_row_universal(row)
        row.update({"dictionary_route": dict_route, "dictionary_reason": dict_reason})
        items.append(row)

    nr_e0 = {row_id(r): r for r in load_jsonl(input_dir / "nr3d/official_e0_nr3d_openrouter_qwen_250.jsonl")}
    nr_parse = {row_id(r): r for r in load_jsonl(input_dir / "nr3d/nr3d_dfrc_llm_parse.jsonl")}
    for case in sorted(nr_e0):
        parse = nr_parse[case]
        row = {
            "dataset": "nr3d",
            "case": case,
            "query_type": parse.get("relation_source", "uncategorized"),
            "query": parse.get("caption") or nr_e0[case].get("caption", ""),
        }
        _, dict_route, dict_reason = route_for_row_universal(
            {"relation_source": row["query_type"], "caption": row["query"]}
        )
        row.update({"dictionary_route": dict_route, "dictionary_reason": dict_reason})
        items.append(row)
    return items


def classify_items(args: argparse.Namespace, api_config: ApiConfig | None) -> list[dict]:
    cache_path = args.out_dir / "llm_route_parse.jsonl"
    done = {}
    if args.resume and cache_path.exists():
        done = {(r["dataset"], int(r["case"])): r for r in load_jsonl(cache_path)}
    items = build_items(args.input_dir)
    if args.max_samples:
        items = items[: args.max_samples]

    rows = []
    for idx, item in enumerate(items, 1):
        key = (item["dataset"], item["case"])
        if key in done:
            rows.append(done[key])
            continue
        if args.mock_dictionary:
            route = item["dictionary_route"]
            reason = f"mock_dictionary:{item['dictionary_reason']}"
            raw = json.dumps({"route": route, "reason": reason})
        else:
            if api_config is None:
                raise RuntimeError("Internal error: API config was not resolved.")
            route, reason, raw = call_openrouter(item["query"], item["query_type"], item["dataset"], api_config, args.max_retry)
        row = {
            **item,
            "route": route,
            "reason": reason,
            "raw_answer": raw,
            "route_changed_vs_dictionary": route != item["dictionary_route"],
        }
        rows.append(row)
        if not args.quiet:
            print(f"[{idx}/{len(items)}] {item['dataset']} case={item['case']} qtype={item['query_type']} route={route}")
        write_jsonl(cache_path, rows)
    return rows


def compact(case: int, dataset: str, route_row: dict, source_row: dict, e0_row: dict, fallback: bool = False) -> dict:
    return {
        "case": case,
        "dataset": dataset,
        "query_type": route_row["query_type"],
        "route": route_row["route"],
        "route_reason": "llm_router",
        "llm_reason": route_row.get("reason", ""),
        "dictionary_route": route_row.get("dictionary_route", ""),
        "route_changed_vs_dictionary": bool(route_row.get("route_changed_vs_dictionary")),
        "caption": route_row["query"],
        "iou": float(source_row.get("iou", 0) or 0),
        "acc25": success25(source_row),
        "acc50": success50(source_row),
        "e0_iou": float(e0_row.get("iou", 0) or 0),
        "e0_acc25": success25(e0_row),
        "source_unavailable_fallback_e0": fallback,
    }


def evaluate_routes(input_dir: Path, out_dir: Path, route_rows: list[dict]) -> dict:
    scan_e0 = {row_id(r): r for r in load_jsonl(input_dir / "scanrefer/full_E0_baseline_qwen72b.jsonl")}
    scan_formats = {
        "bev_raw_labeled": {row_id(r): r for r in load_jsonl(input_dir / "scanrefer/bev_raw_labeled/results.jsonl")},
        "seeground_ablation_spatial_only": {
            row_id(r): r for r in load_jsonl(input_dir / "scanrefer/seeground_ablation_spatial_only/results.jsonl")
        },
        "seeground_ablation_3dpos_only": {
            row_id(r): r for r in load_jsonl(input_dir / "scanrefer/seeground_ablation_3dpos_only/results.jsonl")
        },
    }
    nr_e0 = {row_id(r): r for r in load_jsonl(input_dir / "nr3d/official_e0_nr3d_openrouter_qwen_250.jsonl")}
    nr_route_path = input_dir / "nr3d/nr3d_query_type_routed_vlm_bev_exact_scanrefer_format/nr3d_query_type_routed_vlm_results.jsonl"
    nr_routed = {row_id(r): r for r in load_jsonl(nr_route_path)}

    scan_rows, nr_rows = [], []
    for row in route_rows:
        case = int(row["case"])
        route = row["route"]
        if row["dataset"] == "scanrefer":
            e0 = scan_e0[case]
            source = e0 if route == "E0" else scan_formats[route][case]
            scan_rows.append(compact(case, "scanrefer_250", row, source, e0))
        else:
            e0 = nr_e0[case]
            fallback = False
            if route == "E0":
                source = e0
            elif nr_routed[case].get("route") == route:
                source = nr_routed[case]
            else:
                source = e0
                fallback = True
            nr_rows.append(compact(case, "nr3d_250", row, source, e0, fallback=fallback))

    write_jsonl(out_dir / "scanrefer_llm_router_results.jsonl", scan_rows)
    write_jsonl(out_dir / "nr3d_llm_router_results.jsonl", nr_rows)
    summary = {
        "router": "llm" if not route_rows or not str(route_rows[0].get("reason", "")).startswith("mock_dictionary") else "mock_dictionary",
        "prompt_policy": PROMPT_POLICY,
        "scanrefer": metrics(scan_rows),
        "nr3d": metrics(nr_rows),
        "route_diff_vs_dictionary": {
            "num_items": len(route_rows),
            "num_changed": sum(bool(r.get("route_changed_vs_dictionary")) for r in route_rows),
            "changed_by_dataset": dict(Counter(r["dataset"] for r in route_rows if r.get("route_changed_vs_dictionary"))),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    ap.add_argument("--config-alias", default="openrouter-qwen")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-retry", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mock-dictionary", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api_config = resolve_api_config(args)
    route_rows = classify_items(args, api_config)
    summary = evaluate_routes(args.input_dir, args.out_dir, route_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
