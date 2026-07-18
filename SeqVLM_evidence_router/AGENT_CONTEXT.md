# Agent Context

Use this repository as the canonical paper-facing artifact for the
evidence-aware routing project.

## Canonical Repository

Local path:

```text
/home/knuvi/bosung/SeqVLM_evidence_router
```

Primary documents:

- `README.md`: public-facing repository overview and quick reproduction.
- `REPRODUCIBILITY.md`: concise verification guide.
- `experiments/README.md`: full experiment report.
- `experiments/TMM_EXPERIMENT_PLAN.md`: TMM-oriented experiment plan and progress.

## Locked Main Results

| Dataset | E0 Acc@0.25 | Router Acc@0.25 | E0 Acc@0.50 | Router Acc@0.50 | E0 mIoU | Router mIoU |
|---|---:|---:|---:|---:|---:|---:|
| ScanRefer | 0.504 | 0.520 | 0.452 | 0.468 | 0.4306 | 0.4455 |
| NR3D | 0.612 | 0.652 | 0.604 | 0.648 | 0.6107 | 0.6514 |

## Final Router

Implementation:

```text
tools/query_type_router.py
```

Priority:

1. `proximity_derived` -> spatial-only text.
2. visual attribute included -> E0 RGB canvas.
3. pure `ordinal` -> BEV labeled layout.
4. pure `geometric` -> 3D position text.
5. `viewpoint_guided` -> E0 RGB canvas.
6. default/mixed/ambiguous -> E0 RGB canvas.

## Verification

Run:

```bash
python tests/smoke_test_pipeline.py
```

The smoke test must reproduce:

- ScanRefer router: 0.520 / 0.468 / 0.4455
- NR3D router: 0.652 / 0.648 / 0.6514

## Scope

The repository is for paper introduction and verification. It bundles completed
source outputs and compact diagnostics. It does not contain private API keys or
full ScanNet/Mask3D raw assets.
