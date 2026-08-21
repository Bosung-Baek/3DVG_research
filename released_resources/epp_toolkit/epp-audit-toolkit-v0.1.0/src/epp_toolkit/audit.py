from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from . import __version__
from .contract import COMPONENTS, terminal_state, validate


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [mid-half, mid+half]


def metric(k, n):
    return {"numerator": k, "denominator": n, "rate": k/n if n else None, "wilson95": wilson(k, n)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def aggregate(rows):
    rows = [validate(row) for row in rows]
    ta_rows = [r for r in rows if r["components"]["target_availability"] is not None]
    te_rows = [r for r in rows if r["components"]["target_availability"] is True and r["components"]["target_exposure"] is not None]
    ov_rows = [r for r in rows if r["components"]["output_validity"] is not None]
    cg_rows = [r for r in rows if r["components"]["target_availability"] is True and r["components"]["target_exposure"] is True and r["components"]["output_validity"] is True and r["components"]["grounding_outcome"] is not None]
    e2e_rows = [r for r in rows if r.get("endpoint_success") is not None]
    states = Counter(terminal_state(r) for r in rows)
    return {
        "schema_version": "1.0", "toolkit_version": __version__, "n_rows": len(rows),
        "system": sorted({r["system"] for r in rows}), "dataset": sorted({r["dataset"] for r in rows}),
        "metrics": {
            "target_availability_rate": metric(sum(r["components"]["target_availability"] is True for r in ta_rows), len(ta_rows)),
            "target_exposure_rate": metric(sum(r["components"]["target_exposure"] is True for r in te_rows), len(te_rows)),
            "output_validity_rate": metric(sum(r["components"]["output_validity"] is True for r in ov_rows), len(ov_rows)),
            "conditional_grounding_accuracy": metric(sum(r["components"]["grounding_outcome"] is True for r in cg_rows), len(cg_rows)),
            "end_to_end_grounding_accuracy": metric(sum(r["endpoint_success"] is True for r in e2e_rows), len(e2e_rows)),
        },
        "terminal_states": dict(states),
    }


def write_report(rows, input_paths, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    profile = aggregate(rows)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            enriched = dict(row); enriched["terminal_state"] = terminal_state(row)
            f.write(json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    observability = {}
    for component in COMPONENTS:
        counts = Counter(r["provenance"].get(component, "Unobserved") for r in rows)
        observability[component] = dict(counts)
    (output_dir / "observability.json").write_text(json.dumps(observability, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"toolkit_version": __version__, "inputs": [{"path": str(p), "sha256": sha256(p)} for p in input_paths]}
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "profile.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "numerator", "denominator", "rate", "wilson_low", "wilson_high"])
        for name, value in profile["metrics"].items():
            w.writerow([name, value["numerator"], value["denominator"], value["rate"], *value["wilson95"]])
    return profile

