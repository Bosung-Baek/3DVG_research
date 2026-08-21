#!/usr/bin/env python3
"""Reproduce supported manuscript profiles from a 3DVG_research checkout.

No raw dataset or paid API call is required.  Quantities lacking sufficient
row-level artifacts are explicitly recorded as not reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from epp_toolkit.adapters import base, csvg_compatible
from epp_toolkit.audit import write_report
from epp_toolkit.contract import COMPONENTS


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_pair(profile, name):
    value = profile["metrics"][name]
    return [value["numerator"], value["denominator"]]


def seqvlm_rows(repo, dataset):
    table_path = repo / "experiments/table3_completion/seqvlm_metrics.records.jsonl"
    profile_path = repo / f"experiments/evidence_delivery_profile/{dataset}/records.jsonl"
    table = [r for r in read_jsonl(table_path) if r["dataset"] == dataset]
    profiles = {int(r["query_id"]): r for r in read_jsonl(profile_path) if r["policy"] == "dict"}
    rows = []
    for r in table:
        rec = profiles[int(r["case"])]
        te_applicable = bool(rec["available"] and rec["vlm_mediated"])
        go_applicable = bool(r["decision_evaluable"])
        components = {
            "target_availability": bool(r["available"]),
            "target_exposure": bool(rec["exposed"]) if te_applicable else None,
            "output_validity": bool(r["output_valid"]),
            "grounding_outcome": bool(r["decision_success"]) if go_applicable else None,
        }
        provenance = {k: ("Native" if v is not None else "N/A") for k, v in components.items()}
        rows.append(base(
            f"{dataset}:dict:{r['case']}", "SeqVLM", dataset, rec.get("scene_id"), components,
            bool(r["end_to_end_success"]), provenance,
            metadata={"route": r.get("route"), "query_type": rec.get("query_type")},
        ))
    return rows, [table_path, profile_path]


def profile_pairs(profile):
    actual = {
        "availability": metric_pair(profile, "target_availability_rate"),
        "exposure": metric_pair(profile, "target_exposure_rate"),
        "output_validity": metric_pair(profile, "output_validity_rate"),
        "decision_accuracy": metric_pair(profile, "conditional_grounding_accuracy"),
        "end_to_end": metric_pair(profile, "end_to_end_grounding_accuracy"),
    }
    return actual


def assert_expected(name, profile, expectations):
    actual = profile_pairs(profile)
    wanted = expectations[name]
    if actual != wanted:
        raise RuntimeError(f"{name} aggregate mismatch: actual={actual}, expected={wanted}")
    return actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    repo, out = args.repo_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "1.0", "systems": {}, "source_hashes": {}}
    expectations_path = Path(__file__).with_name("manuscript_expectations.json")
    expectations = json.loads(expectations_path.read_text())
    report["expectations_sha256"] = digest(expectations_path)

    for dataset in ("scanrefer", "nr3d"):
        rows, sources = seqvlm_rows(repo, dataset)
        profile = write_report(rows, sources, out / f"seqvlm_{dataset}")
        name = f"SeqVLM/{dataset}"
        report["systems"][name] = {"status": "exact_reproduction", "metrics": assert_expected(name, profile, expectations)}
        for source in sources:
            report["source_hashes"][str(source.relative_to(repo))] = digest(source)

    csvg_source = repo / "experiments/table3_completion/csvg_compatible_full_profile.records.jsonl"
    csvg_rows = [csvg_compatible(r) for r in read_jsonl(csvg_source)]
    csvg_profile = write_report(csvg_rows, [csvg_source], out / "csvg_compatible")
    csvg_name = "CSVG-compatible/scanrefer"
    report["systems"][csvg_name] = {"status": "sensitivity_only_reproduction", "metrics": assert_expected(csvg_name, csvg_profile, expectations)}
    report["source_hashes"][str(csvg_source.relative_to(repo))] = digest(csvg_source)

    for name, path, reason in (
        ("SeeGround", repo / "experiments/table3_completion/seeground_metrics.json", "aggregate is verified, but the current distributable does not include the third-party proposal archive needed for a clean row-level rerun"),
        ("M3DRef-CLIP", repo / "experiments/table3_completion/final/table3_wacv_main.csv", "no standalone row-level adapter artifact is included in the current distributable"),
    ):
        report["systems"][name] = {"status": "artifact_verified_not_clean_reproduced", "reason": reason, "source_sha256": digest(path)}

    report["status"] = "PASS_WITH_DECLARED_PARTIAL_OBSERVABILITY"
    (out / "reproduction_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
