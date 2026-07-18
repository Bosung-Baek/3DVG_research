#!/usr/bin/env python3
"""Paired statistical tests for E0 vs final evidence router."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from analysis_common import SourcePack, evaluate_policy, route_final, write_json  # noqa: E402

DEFAULT_INPUTS = REPO / "inputs"
DEFAULT_OUT = REPO / "experiments/statistics"


def comb(n: int, k: int) -> int:
    return math.comb(n, k)


def exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    observed = comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    prob = 0.0
    for i in range(n + 1):
        pi = comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        if pi <= observed + 1e-15:
            prob += pi
    return min(1.0, prob)


def mcnemar_exact(rows: list[dict], metric: str) -> dict:
    if metric == "acc25":
        r_key, e_key = "acc25", "e0_acc25"
    else:
        r_key, e_key = "acc50", "e0_acc50"
    b = sum((not r[e_key]) and r[r_key] for r in rows)
    c = sum(r[e_key] and not r[r_key] for r in rows)
    return {
        "recoveries_b": b,
        "regressions_c": c,
        "discordant": b + c,
        "exact_two_sided_p": exact_binomial_two_sided(min(b, c), b + c),
    }


def bootstrap_gain(rows: list[dict], metric: str, samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    n = len(rows)
    gains = []
    for _ in range(samples):
        vals = [rows[rng.randrange(n)] for _ in range(n)]
        if metric == "miou":
            gain = sum(v["iou"] - v["e0_iou"] for v in vals) / n
        elif metric == "acc25":
            gain = sum(float(v["acc25"]) - float(v["e0_acc25"]) for v in vals) / n
        else:
            gain = sum(float(v["acc50"]) - float(v["e0_acc50"]) for v in vals) / n
        gains.append(gain)
    gains.sort()
    return {
        "gain": sum(gains) / samples,
        "ci95": [gains[int(0.025 * samples)], gains[int(0.975 * samples) - 1]],
        "num_samples": samples,
        "seed": seed,
    }


def permutation_p(rows: list[dict], samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    diffs = [r["iou"] - r["e0_iou"] for r in rows]
    observed = sum(diffs) / len(diffs)
    more_extreme = 0
    for _ in range(samples):
        val = sum((d if rng.random() < 0.5 else -d) for d in diffs) / len(diffs)
        if abs(val) >= abs(observed):
            more_extreme += 1
    return {"observed_gain": observed, "two_sided_p": (more_extreme + 1) / (samples + 1), "num_samples": samples, "seed": seed}


def analyze_dataset(rows: list[dict], samples: int, seed: int) -> dict:
    return {
        "acc25": {
            "gain": sum(float(r["acc25"]) - float(r["e0_acc25"]) for r in rows) / len(rows),
            "mcnemar": mcnemar_exact(rows, "acc25"),
            "bootstrap": bootstrap_gain(rows, "acc25", samples, seed),
        },
        "acc50": {
            "gain": sum(float(r["acc50"]) - float(r["e0_acc50"]) for r in rows) / len(rows),
            "mcnemar": mcnemar_exact(rows, "acc50"),
            "bootstrap": bootstrap_gain(rows, "acc50", samples, seed + 1),
        },
        "miou": {
            "gain": sum(r["iou"] - r["e0_iou"] for r in rows) / len(rows),
            "bootstrap": bootstrap_gain(rows, "miou", samples, seed + 2),
            "permutation": permutation_p(rows, samples, seed + 3),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pack = SourcePack(args.input_dir)
    out = {}
    for dataset in ("scanrefer", "nr3d"):
        rows = evaluate_policy(pack, dataset, "full_router", route_final)
        out[dataset] = analyze_dataset(rows, args.samples, args.seed)
    write_json(args.out_dir / "paired_tests.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
