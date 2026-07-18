# Reproducibility Guide

This repository is the paper-facing artifact for the evidence-aware routing
experiments. It is designed to verify the reported routing results without
requiring the full original SeqVLM workspace or raw ScanNet/Mask3D assets.

## What Is Reproducible Here

The bundled data pack under `data/source_outputs/` and `inputs/` supports:

- ScanRefer and NR3D main table recomposition.
- Final deterministic evidence-aware routing.
- Input-format ablations from completed source outputs.
- Query-type and route-level analysis.
- LLM-router comparison summaries.
- NR3D end-to-end rerun summaries and additional diagnostic outputs.

## Quick Verification

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the smoke test:

```bash
python tests/smoke_test_pipeline.py
```

The expected locked results are:

| Dataset | Method | Acc@0.25 | Acc@0.50 | mIoU |
|---|---|---:|---:|---:|
| ScanRefer | E0 baseline | 0.504 | 0.452 | 0.4306 |
| ScanRefer | evidence router | 0.520 | 0.468 | 0.4455 |
| NR3D | E0 baseline | 0.612 | 0.604 | 0.6107 |
| NR3D | evidence router | 0.652 | 0.648 | 0.6514 |

## Full Local Regeneration From Bundled Outputs

```bash
python tools/run_full_pipeline.py --clean-inputs
python tools/run_experiment_suite.py --out-dir experiments
```

These commands do not call any external VLM API.

## Scope

This artifact intentionally stores completed source outputs and compact
diagnostic assets. It does not include full ScanNet/Mask3D raw assets or the
complete original SeqVLM training/evaluation workspace. Fresh VLM API runs and
raw canvas/BEV regeneration require the original external assets and private API
configuration.
