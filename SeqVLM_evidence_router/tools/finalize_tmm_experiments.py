#!/usr/bin/env python3
"""Finalize result-file based TMM experiment artifacts.

This script does not call a VLM. It reorganizes and analyzes the completed
outputs already bundled in the repository so the experiment folder contains
paper-ready tables for router components, LLM transitions, runtime/cost proxy,
failure examples, and an automatic evidence audit.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from analysis_common import has_visual_attribute, is_pure_geometric_query, is_pure_ordinal_query  # noqa: E402


ROUTES = [
    "E0",
    "seeground_ablation_spatial_only",
    "bev_raw_labeled",
    "seeground_ablation_3dpos_only",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def acc25(row: dict) -> bool:
    return bool(row.get("acc25") if "acc25" in row else row.get("success_iou25", False))


def acc50(row: dict) -> bool:
    return bool(row.get("acc50") if "acc50" in row else row.get("success_iou50", False))


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"num_queries": 0, "acc_iou25": None, "acc_iou50": None, "mean_iou": None}
    return {
        "num_queries": n,
        "acc_iou25": round(sum(acc25(r) for r in rows) / n, 4),
        "acc_iou50": round(sum(acc50(r) for r in rows) / n, 4),
        "mean_iou": round(sum(float(r.get("iou", 0) or 0) for r in rows) / n, 4),
    }


def route_label(route: str) -> str:
    return {
        "E0": "E0 RGB canvas",
        "seeground_ablation_spatial_only": "spatial-only text",
        "bev_raw_labeled": "BEV labeled layout",
        "seeground_ablation_3dpos_only": "3D position text",
    }.get(route, route)


def split_router_components() -> dict:
    src = REPO / "experiments/ablation/policy_ablation/summary.json"
    summary = load_json(src)
    components = summary.get("components", {})
    out = REPO / "experiments/ablation/router_components"
    write_json(out / "summary.json", components)

    rows = []
    for variant, by_dataset in components.items():
        for dataset, vals in by_dataset.items():
            if dataset == "combined":
                continue
            rows.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "acc_iou25": vals.get("acc_iou25"),
                    "acc_iou50": vals.get("acc_iou50"),
                    "mean_iou": vals.get("mean_iou"),
                    "recoveries": vals.get("e0_recoveries"),
                    "regressions": vals.get("e0_regressions"),
                    "net_vs_e0": vals.get("net_vs_e0"),
                    "route_counts": vals.get("route_counts", {}),
                }
            )
    write_json(out / "table.json", rows)
    return {"path": str(out / "summary.json"), "num_variants": len(components)}


def build_llm_transition_matrix() -> dict:
    out_dir = REPO / "experiments/ablation/llm_router_priority_openrouter_qwen"
    rows = []
    for dataset in ("scanrefer", "nr3d"):
        path = out_dir / f"{dataset}_llm_router_results.jsonl"
        for row in load_jsonl(path):
            fixed_dataset = "scanrefer" if dataset == "scanrefer" else "nr3d"
            row = dict(row)
            row["dataset"] = fixed_dataset
            rows.append(row)

    matrix: dict[str, dict[str, int]] = {r: {c: 0 for c in ROUTES} for r in ROUTES}
    by_dataset: dict[str, dict[str, dict[str, int]]] = {
        "scanrefer": {r: {c: 0 for c in ROUTES} for r in ROUTES},
        "nr3d": {r: {c: 0 for c in ROUTES} for r in ROUTES},
    }
    transition_perf = defaultdict(list)
    examples = defaultdict(list)
    changed = 0

    for row in rows:
        d_route = row.get("dictionary_route", "NA")
        l_route = row.get("route", "NA")
        if d_route in matrix and l_route in matrix[d_route]:
            matrix[d_route][l_route] += 1
            by_dataset[row["dataset"]][d_route][l_route] += 1
        key = f"{d_route}->{l_route}"
        transition_perf[key].append(row)
        if d_route != l_route:
            changed += 1
            if len(examples[key]) < 5:
                examples[key].append(
                    {
                        "dataset": row["dataset"],
                        "case": row.get("case"),
                        "query_type": row.get("query_type"),
                        "caption": row.get("caption"),
                        "llm_reason": row.get("llm_reason"),
                        "dictionary_route": d_route,
                        "llm_route": l_route,
                        "e0_acc25": row.get("e0_acc25"),
                        "llm_acc25": row.get("acc25"),
                        "llm_iou": row.get("iou"),
                    }
                )

    transition_metrics = {
        key: {
            **metrics(vals),
            "e0_acc_iou25": round(sum(bool(v.get("e0_acc25")) for v in vals) / len(vals), 4),
            "count": len(vals),
        }
        for key, vals in sorted(transition_perf.items())
    }
    result = {
        "num_queries": len(rows),
        "agreement": len(rows) - changed,
        "changed": changed,
        "agreement_rate": round((len(rows) - changed) / len(rows), 4),
        "matrix_rows_dictionary_cols_llm": matrix,
        "by_dataset": by_dataset,
        "transition_metrics": transition_metrics,
        "changed_examples": dict(examples),
    }
    write_json(out_dir / "transition_matrix.json", result)
    return result


def runtime_proxy() -> dict:
    final_paths = {
        "scanrefer": REPO / "experiments/main_table/final_router_outputs/scanrefer_universal_evidence_routed_results.jsonl",
        "nr3d": REPO / "experiments/main_table/final_router_outputs/nr3d_universal_evidence_routed_results.jsonl",
    }
    llm_parse = REPO / "experiments/ablation/llm_router_priority_openrouter_qwen/llm_route_parse.jsonl"
    parse_rows = load_jsonl(llm_parse) if llm_parse.exists() else []

    datasets = {}
    for dataset, path in final_paths.items():
        rows = load_jsonl(path)
        route_counts = Counter(r["route"] for r in rows)
        avg_query_words = sum(len(str(r.get("caption", "")).split()) for r in rows) / len(rows)
        datasets[dataset] = {
            "num_queries": len(rows),
            "route_counts": dict(route_counts),
            "avg_query_words": round(avg_query_words, 2),
            "dictionary_router_calls": 0,
            "vlm_selection_calls": len(rows),
            "llm_router_extra_calls_if_used": len(rows),
        }

    result = {
        "note": "Proxy accounting from completed result files. It reports call counts, not wall-clock latency.",
        "methods": {
            "E0_baseline": {
                "router": "none",
                "extra_router_vlm_or_llm_calls_per_query": 0,
                "selection_vlm_calls_per_query": 1,
            },
            "dictionary_evidence_router": {
                "router": "deterministic local rules",
                "extra_router_vlm_or_llm_calls_per_query": 0,
                "selection_vlm_calls_per_query": 1,
            },
            "priority_llm_router": {
                "router": "OpenRouter Qwen route classifier",
                "extra_router_vlm_or_llm_calls_per_query": 1,
                "selection_vlm_calls_per_query": 1,
                "observed_parse_rows": len(parse_rows),
            },
        },
        "datasets": datasets,
    }
    write_json(REPO / "experiments/runtime/runtime_proxy.json", result)
    return result


def choose_failure_cases() -> dict:
    transitions = load_jsonl(REPO / "experiments/ablation/route_contribution/transitions.jsonl")
    selected = []

    buckets = {
        "e0_fail_spatial_success": lambda r: r["route"] == "seeground_ablation_spatial_only" and r["transition"] == "recovery",
        "spatial_regression": lambda r: r["route"] == "seeground_ablation_spatial_only" and r["transition"] == "regression",
        "bev_success": lambda r: r["route"] == "bev_raw_labeled" and r["acc25"],
        "router_failure": lambda r: r["route"] != "E0" and not r["acc25"],
        "visual_fallback_kept_correct": lambda r: r["route"] == "E0" and r["route_reason"] == "visual_attribute_default_e0" and r["acc25"],
    }

    for bucket, pred in buckets.items():
        matches = [r for r in transitions if pred(r)]
        for row in matches[:4]:
            selected.append(
                {
                    "bucket": bucket,
                    "dataset": row["dataset"],
                    "case": row["case"],
                    "query_type": row["query_type"],
                    "route": row["route"],
                    "route_name": route_label(row["route"]),
                    "route_reason": row["route_reason"],
                    "caption": row["caption"],
                    "e0_iou": row["e0_iou"],
                    "e0_acc25": row["e0_acc25"],
                    "routed_iou": row["iou"],
                    "routed_acc25": row["acc25"],
                    "transition": row["transition"],
                }
            )

    out = REPO / "experiments/failure_visualization"
    write_jsonl(out / "cases.jsonl", selected)
    write_json(out / "summary.json", {"num_cases": len(selected), "bucket_counts": dict(Counter(c["bucket"] for c in selected))})

    lines = [
        "# Failure and Recovery Cases",
        "",
        "These cases are selected from completed result files. Full RGB-canvas figures require the original rendered canvas assets; this standalone bundle stores query/output evidence and available BEV assets.",
        "",
    ]
    for case in selected:
        lines.extend(
            [
                f"## {case['bucket']} / {case['dataset']} case {case['case']}",
                "",
                f"- Query type: `{case['query_type']}`",
                f"- Route: `{case['route_name']}` (`{case['route_reason']}`)",
                f"- Query: {case['caption']}",
                f"- E0: IoU={case['e0_iou']:.4f}, Acc@0.25={case['e0_acc25']}",
                f"- Routed: IoU={case['routed_iou']:.4f}, Acc@0.25={case['routed_acc25']}",
                f"- Transition: `{case['transition']}`",
                "",
            ]
        )
    (out / "failure_cases.md").write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(out / "cases.jsonl"), "num_cases": len(selected)}


def evidence_audit_proxy() -> dict:
    rows = []
    for path in (
        REPO / "experiments/main_table/final_router_outputs/scanrefer_universal_evidence_routed_results.jsonl",
        REPO / "experiments/main_table/final_router_outputs/nr3d_universal_evidence_routed_results.jsonl",
    ):
        rows.extend(load_jsonl(path))

    audit_rows = []
    for row in rows:
        text = str(row.get("caption", ""))
        audit_rows.append(
            {
                "dataset": row.get("dataset"),
                "case": row.get("case"),
                "query_type": row.get("query_type"),
                "route": row.get("route"),
                "route_reason": row.get("route_reason"),
                "visual_attribute": has_visual_attribute(text),
                "pure_ordinal": is_pure_ordinal_query(text),
                "pure_geometric": is_pure_geometric_query(text),
                "caption": text,
            }
        )

    out = REPO / "experiments/evidence_audit"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "auto_evidence_labels.csv").open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "note": "Automatic evidence audit using the final deterministic evidence rules, not human labels.",
        "num_queries": len(audit_rows),
        "route_reason_counts": dict(Counter(r["route_reason"] for r in audit_rows)),
        "visual_attribute_count": sum(r["visual_attribute"] for r in audit_rows),
        "pure_ordinal_count": sum(r["pure_ordinal"] for r in audit_rows),
        "pure_geometric_count": sum(r["pure_geometric"] for r in audit_rows),
        "by_dataset": {},
    }
    for dataset in sorted(set(r["dataset"] for r in audit_rows)):
        vals = [r for r in audit_rows if r["dataset"] == dataset]
        summary["by_dataset"][dataset] = {
            "num_queries": len(vals),
            "route_reason_counts": dict(Counter(r["route_reason"] for r in vals)),
            "visual_attribute_count": sum(r["visual_attribute"] for r in vals),
            "pure_ordinal_count": sum(r["pure_ordinal"] for r in vals),
            "pure_geometric_count": sum(r["pure_geometric"] for r in vals),
        }
    write_json(out / "summary.json", summary)
    return summary


def update_experiment_summary(extra: dict) -> None:
    path = REPO / "experiments/summary.json"
    summary = load_json(path)
    summary.setdefault("additional_tmm_artifacts", {}).update(extra)
    write_json(path, summary)


def main() -> None:
    artifacts = {
        "router_components": split_router_components(),
        "llm_transition_matrix": {
            "path": "experiments/ablation/llm_router_priority_openrouter_qwen/transition_matrix.json",
            "summary": {
                k: v
                for k, v in build_llm_transition_matrix().items()
                if k in {"num_queries", "agreement", "changed", "agreement_rate"}
            },
        },
        "runtime_proxy": runtime_proxy(),
        "failure_cases": choose_failure_cases(),
        "evidence_audit_proxy": evidence_audit_proxy(),
    }
    update_experiment_summary(artifacts)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
