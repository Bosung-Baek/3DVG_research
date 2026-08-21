from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .adapters import ADAPTERS
from .audit import write_report


def paths(patterns):
    result = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        result.extend(Path(x) for x in matches)
    unique = []
    for path in result:
        if path not in unique:
            unique.append(path)
    if not unique:
        raise SystemExit("no input files matched")
    return unique


def load(input_paths, adapter):
    rows, exclusions = [], []
    fn = ADAPTERS[adapter]
    seen = set()
    for path in input_paths:
        with path.open(encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = fn(json.loads(line))
                    key = (row["system"], row["dataset"], row["query_id"])
                    if key in seen:
                        raise ValueError(f"duplicate canonical key {key}")
                    seen.add(key); rows.append(row)
                except Exception as exc:
                    exclusions.append({"path": str(path), "line": line_no, "reason": str(exc)})
    return rows, exclusions


def main(argv=None):
    ap = argparse.ArgumentParser(prog="epp-audit")
    sub = ap.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="adapt traces and produce an EPP report")
    audit.add_argument("--adapter", choices=sorted(ADAPTERS), default="canonical")
    audit.add_argument("--input", action="append", required=True, help="JSONL path or glob; repeatable")
    audit.add_argument("--output-dir", required=True)
    adapt = sub.add_parser("adapt", help="convert native traces to canonical JSONL")
    adapt.add_argument("--adapter", choices=sorted(ADAPTERS), required=True)
    adapt.add_argument("--input", action="append", required=True)
    adapt.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    input_paths = paths(args.input)
    rows, exclusions = load(input_paths, args.adapter)
    if args.command == "adapt":
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"status": "PASS" if not exclusions else "PARTIAL", "rows": len(rows), "excluded": len(exclusions), "output": str(out)}))
        return 0 if not exclusions else 2
    out_dir = Path(args.output_dir)
    profile = write_report(rows, input_paths, out_dir)
    with (out_dir / "exclusions.jsonl").open("w", encoding="utf-8") as f:
        for row in exclusions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {"status": "PASS" if not exclusions else "PARTIAL", "rows": len(rows), "excluded": len(exclusions), "profile": profile}
    print(json.dumps(payload, indent=2))
    return 0 if not exclusions else 2


if __name__ == "__main__":
    sys.exit(main())

