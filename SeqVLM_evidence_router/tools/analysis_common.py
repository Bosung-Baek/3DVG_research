"""Shared helpers for evidence-router ablation analyses."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from query_type_router import (
    E0_ROUTE,
    has_visual_attribute,
    is_pure_geometric_query,
    is_pure_ordinal_query,
    query_type_from_row,
    route_for_row_universal,
)

ROUTE_SPATIAL = "seeground_ablation_spatial_only"
ROUTE_BEV = "bev_raw_labeled"
ROUTE_3DPOS = "seeground_ablation_3dpos_only"
ROUTES = (E0_ROUTE, ROUTE_SPATIAL, ROUTE_BEV, ROUTE_3DPOS)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def row_id(row: dict) -> int:
    for key in ("case", "case_id", "query_id"):
        if key in row:
            return int(str(row[key]).lstrip("0") or "0")
    raise KeyError(f"No case id in row keys: {sorted(row)}")


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


def query_text(row: dict) -> str:
    return str(row.get("query") or row.get("caption") or row.get("description") or "")


def summarize(rows: list[dict]) -> dict:
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
        "route_reason_counts": dict(Counter(r.get("route_reason", "") for r in rows)),
        "source_unavailable_fallbacks": sum(bool(r.get("source_unavailable_fallback_e0")) for r in rows),
        "e0_recoveries": recover,
        "e0_regressions": regress,
        "net_vs_e0": recover - regress,
    }


def grouped_summary(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "NA"))].append(row)
    return {name: summarize(vals) for name, vals in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))}


class SourcePack:
    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir
        scan_dir = input_dir / "scanrefer"
        nr_dir = input_dir / "nr3d"
        self.scan_e0 = {row_id(r): r for r in load_jsonl(scan_dir / "full_E0_baseline_qwen72b.jsonl")}
        self.scan_formats = {
            ROUTE_BEV: {row_id(r): r for r in load_jsonl(scan_dir / "bev_raw_labeled/results.jsonl")},
            ROUTE_SPATIAL: {row_id(r): r for r in load_jsonl(scan_dir / "seeground_ablation_spatial_only/results.jsonl")},
            ROUTE_3DPOS: {row_id(r): r for r in load_jsonl(scan_dir / "seeground_ablation_3dpos_only/results.jsonl")},
        }
        self.scan_labels = self.scan_formats[ROUTE_BEV]

        nr_route_path = nr_dir / "nr3d_query_type_routed_vlm_bev_exact_scanrefer_format/nr3d_query_type_routed_vlm_results.jsonl"
        self.nr_e0 = {row_id(r): r for r in load_jsonl(nr_dir / "official_e0_nr3d_openrouter_qwen_250.jsonl")}
        self.nr_parse = {row_id(r): r for r in load_jsonl(nr_dir / "nr3d_dfrc_llm_parse.jsonl")}
        self.nr_routed = {row_id(r): r for r in load_jsonl(nr_route_path)}

    def iter_items(self, dataset: str) -> list[dict]:
        if dataset == "scanrefer":
            items = []
            for case in sorted(self.scan_e0):
                label = self.scan_labels[case]
                text = label.get("query") or label.get("caption") or self.scan_e0[case].get("caption", "")
                items.append(
                    {
                        "dataset": "scanrefer",
                        "case": case,
                        "query_type": label.get("query_type", "uncategorized"),
                        "query": text,
                        "caption": text,
                    }
                )
            return items
        if dataset == "nr3d":
            items = []
            for case in sorted(self.nr_e0):
                parse = self.nr_parse[case]
                text = parse.get("caption") or self.nr_e0[case].get("caption", "")
                items.append(
                    {
                        "dataset": "nr3d",
                        "case": case,
                        "relation_source": parse.get("relation_source", "uncategorized"),
                        "query_type": parse.get("relation_source", "uncategorized"),
                        "query": text,
                        "caption": text,
                    }
                )
            return items
        raise ValueError(f"Unknown dataset: {dataset}")

    def e0_row(self, dataset: str, case: int) -> dict:
        return self.scan_e0[case] if dataset == "scanrefer" else self.nr_e0[case]

    def source_row(self, dataset: str, case: int, route: str) -> tuple[dict, bool]:
        if route == E0_ROUTE:
            return self.e0_row(dataset, case), False
        if dataset == "scanrefer":
            return self.scan_formats[route][case], False
        available = self.nr_routed[case]
        if available.get("route") == route:
            return available, False
        return self.nr_e0[case], True


def compact_result(item: dict, route: str, reason: str, source_row: dict, e0_row: dict, fallback: bool = False) -> dict:
    return {
        "case": int(item["case"]),
        "dataset": item["dataset"],
        "query_type": query_type_from_row(item),
        "route": route,
        "route_reason": reason,
        "caption": query_text(item),
        "iou": float(source_row.get("iou", 0) or 0),
        "acc25": success25(source_row),
        "acc50": success50(source_row),
        "e0_iou": float(e0_row.get("iou", 0) or 0),
        "e0_acc25": success25(e0_row),
        "e0_acc50": success50(e0_row),
        "source_unavailable_fallback_e0": fallback,
    }


def route_final(item: dict) -> tuple[str, str]:
    _, route, reason = route_for_row_universal(item)
    return route, reason


def route_e0(_: dict) -> tuple[str, str]:
    return E0_ROUTE, "e0_only"


def route_proximity_only(item: dict) -> tuple[str, str]:
    if query_type_from_row(item) == "proximity_derived":
        return ROUTE_SPATIAL, "proximity_only_spatial"
    return E0_ROUTE, "proximity_only_default_e0"


def route_proximity_ordinal(item: dict) -> tuple[str, str]:
    if query_type_from_row(item) == "proximity_derived":
        return ROUTE_SPATIAL, "proximity_spatial"
    if query_type_from_row(item) == "ordinal" and is_pure_ordinal_query(query_text(item)):
        return ROUTE_BEV, "pure_ordinal_bev"
    return E0_ROUTE, "proximity_ordinal_default_e0"


def route_proximity_geometric(item: dict) -> tuple[str, str]:
    if query_type_from_row(item) == "proximity_derived":
        return ROUTE_SPATIAL, "proximity_spatial"
    if query_type_from_row(item) == "geometric" and is_pure_geometric_query(query_text(item)):
        return ROUTE_3DPOS, "pure_geometric_3dpos"
    return E0_ROUTE, "proximity_geometric_default_e0"


def route_proximity_ordinal_geometric(item: dict) -> tuple[str, str]:
    if query_type_from_row(item) == "proximity_derived":
        return ROUTE_SPATIAL, "proximity_spatial"
    if query_type_from_row(item) == "ordinal" and is_pure_ordinal_query(query_text(item)):
        return ROUTE_BEV, "pure_ordinal_bev"
    if query_type_from_row(item) == "geometric" and is_pure_geometric_query(query_text(item)):
        return ROUTE_3DPOS, "pure_geometric_3dpos"
    return E0_ROUTE, "proximity_ordinal_geometric_default_e0"


def route_without_visual_fallback(item: dict) -> tuple[str, str]:
    qt = query_type_from_row(item)
    text = query_text(item)
    if qt == "proximity_derived" or "near" in text.lower() or "closest" in text.lower() or "farthest" in text.lower():
        return ROUTE_SPATIAL, "no_visual_fallback_spatial"
    if qt == "ordinal" and is_pure_ordinal_query(text):
        return ROUTE_BEV, "no_visual_fallback_ordinal_bev"
    if qt == "geometric" and is_pure_geometric_query(text):
        return ROUTE_3DPOS, "no_visual_fallback_geometric_3dpos"
    return E0_ROUTE, "no_visual_fallback_default_e0"


def route_without_purity_constraint(item: dict) -> tuple[str, str]:
    qt = query_type_from_row(item)
    if qt == "proximity_derived":
        return ROUTE_SPATIAL, "no_purity_proximity_spatial"
    if qt == "ordinal":
        return ROUTE_BEV, "no_purity_ordinal_bev"
    if qt == "geometric":
        return ROUTE_3DPOS, "no_purity_geometric_3dpos"
    return E0_ROUTE, "no_purity_default_e0"


def route_without_viewpoint_fallback(item: dict) -> tuple[str, str]:
    qt = query_type_from_row(item)
    if qt == "viewpoint_guided":
        return ROUTE_BEV, "no_viewpoint_fallback_bev"
    return route_final(item)


def route_without_priority_ordering(item: dict) -> tuple[str, str]:
    text = query_text(item)
    qt = query_type_from_row(item)
    if has_visual_attribute(text):
        return E0_ROUTE, "unordered_visual_e0"
    if qt == "viewpoint_guided":
        return E0_ROUTE, "unordered_viewpoint_e0"
    if qt == "ordinal" and is_pure_ordinal_query(text):
        return ROUTE_BEV, "unordered_ordinal_bev"
    if qt == "geometric" and is_pure_geometric_query(text):
        return ROUTE_3DPOS, "unordered_geometric_3dpos"
    if qt == "proximity_derived":
        return ROUTE_SPATIAL, "unordered_proximity_spatial"
    return E0_ROUTE, "unordered_default_e0"


POLICY_VARIANTS: dict[str, Callable[[dict], tuple[str, str]]] = {
    "e0_only": route_e0,
    "proximity_only": route_proximity_only,
    "proximity_plus_ordinal": route_proximity_ordinal,
    "proximity_plus_geometric": route_proximity_geometric,
    "proximity_plus_ordinal_geometric": route_proximity_ordinal_geometric,
    "full_router": route_final,
    "without_visual_fallback": route_without_visual_fallback,
    "without_purity_constraint": route_without_purity_constraint,
    "without_viewpoint_fallback": route_without_viewpoint_fallback,
    "without_priority_ordering": route_without_priority_ordering,
}


def evaluate_policy(pack: SourcePack, dataset: str, policy_name: str, router: Callable[[dict], tuple[str, str]]) -> list[dict]:
    rows = []
    for item in pack.iter_items(dataset):
        route, reason = router(item)
        case = int(item["case"])
        e0 = pack.e0_row(dataset, case)
        source, fallback = pack.source_row(dataset, case, route)
        row = compact_result(item, route, reason, source, e0, fallback=fallback)
        row["policy"] = policy_name
        rows.append(row)
    return rows


def bootstrap_ci(values: list[float], *, samples: int = 5000, seed: int = 7, alpha: float = 0.05) -> dict:
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(samples):
        draws.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo = draws[int((alpha / 2) * samples)]
    hi = draws[int((1 - alpha / 2) * samples) - 1]
    return {"mean": sum(values) / n, "ci95": [lo, hi], "num_samples": samples, "seed": seed}
