# Full-Dataset Evaluation Plan

This note schedules validation-scale experiments for ScanRefer and Nr3D. It
separates API-free preprocessing from VLM/API evaluation so that expensive runs
can be launched only after dataset coverage and route counts are known.

Important ScanRefer count note: the official ScanRefer val split is available
at `/data/knuvi/bosung/scanrefer/ScanRefer_filtered_val.json` and has 9,508
queries. The local ZSVG3D-converted file has 9,498 rows, so it is a
protocol-specific converted file rather than the official raw val source.

## 1. Available Full Splits

| Dataset | Available split file | N | Current 250 subset relation |
|---|---|---:|---|
| ScanRefer official raw val | `/data/knuvi/bosung/scanrefer/ScanRefer_filtered_val.json` | 9,508 | `scanrefer_250.json` is a separate sample, not the first 250 rows. |
| ScanRefer ZSVG protocol | `SeqVLM/data/ZSVG3D/data/scanrefer_val.json` | 9,498 | Converted protocol file; do not report as official full val. |
| Nr3D | `SeqVLM/data/nr3d_val_with_obj_name.json` | 7,457 | `nr3d_250.json` is a separate sample, not the first 250 rows. |

There is no hidden label-free test set in the current workspace. The feasible
immediate target is the labeled Nr3D validation split and the official
ScanRefer raw validation split.

## 2. API-Free Preprocessing Status

Completed by the corrected v2 object-ID pipeline:

```bash
/home/knuvi/anaconda3/envs/sam3/bin/python \
  SeqVLM_evidence_router/tools/prepare_full_dataset_preprocessing.py \
  --out SeqVLM_evidence_router/experiments/full_dataset_preprocessing_v2
```

Outputs:

| Output | Path |
|---|---|
| Combined summary | `experiments/full_dataset_preprocessing_v2/summary.json` |
| ScanRefer official route plan | `experiments/full_dataset_preprocessing_v2/scanrefer_official_val_preprocessed.jsonl` |
| ScanRefer official summary | `experiments/full_dataset_preprocessing_v2/scanrefer_official_val_summary.json` |
| Nr3D full route plan | `experiments/full_dataset_preprocessing_v2/nr3d_full_preprocessed.jsonl` |
| Nr3D summary | `experiments/full_dataset_preprocessing_v2/nr3d_full_summary.json` |

No VLM/API calls are made in this stage.

### ScanRefer Official Val Coverage

| Item | Count |
|---|---:|
| Official val queries | 9,508 |
| Scenes | 141 |
| Canvas scenes available | 115 |
| Queries with at least one same-class canvas candidate | 7,232 |
| No same-class Mask3D candidate | 1,194 |
| Same-class candidate exists but no canvas candidate | 1,082 |

ScanRefer target category uses the official `object_name` field. Query
relation_source is derived from caption cues because the official raw annotation
does not include ZSVG symbolic programs.

### Nr3D Coverage

| Item | Count |
|---|---:|
| Full queries | 7,457 |
| Scenes | 130 |
| Object-ID candidate/canvas coverage | 7,457 / 7,457 |
| Average candidates | 3.1251 |

## 3. Full Route Plan

### ScanRefer Official Val

| Route | N |
|---|---:|
| E0 RGB canvas | 7,279 |
| spatial-only text | 1,664 |
| 3D position text | 346 |
| BEV labeled layout | 219 |

### Nr3D

| Route | N |
|---|---:|
| E0 RGB canvas | 3,982 |
| spatial-only text | 3,015 |
| 3D position text | 350 |
| BEV labeled layout | 110 |

## 4. Completed Full-Validation Run

The first completed full-validation API run is NR3D final-router evaluation
using the corrected v2 materialized inputs.

| Dataset | Run | N | Acc@0.25 | Acc@0.50 | mIoU | Status |
|---|---|---:|---:|---:|---:|---|
| Nr3D val | final evidence router, OpenRouter Qwen | 7,457 | 0.5958 | 0.5946 | 0.5985 | complete |

Route breakdown:

| Route | N | Acc@0.25 | Acc@0.50 | mIoU |
|---|---:|---:|---:|---:|
| E0 RGB canvas | 3,982 | 0.5497 | 0.5482 | 0.5532 |
| spatial-only text | 3,015 | 0.6594 | 0.6584 | 0.6609 |
| 3D position text | 350 | 0.6114 | 0.6114 | 0.6137 |
| BEV labeled layout | 110 | 0.4727 | 0.4727 | 0.4796 |

Lightweight summaries are committed under:

```text
experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/
```

The full generated per-query output remains on the data disk:

```text
/data/knuvi/bosung/evidence_router_full_runs_v2/nr3d/final_router/openrouter-qwen/
```

This result should not be directly compared to the 250-query locked NR3D
baseline. A full E0-only baseline is still needed for the full-validation main
table.

ScanRefer counts are based on deterministic relation-source extraction from
caption cues, followed by the final evidence-aware router. Nr3D counts use the
available program/view_dep/caption cues.

## 5. Required Full-Dataset Experiments

The following experiments are needed if the paper moves from 250-query evidence
to full validation-set evidence.

| Priority | Experiment | Dataset | API needed | Purpose |
|---:|---|---|---|---|
| 1 | Full E0 baseline | ScanRefer official, Nr3D | Yes | Establish full-set baseline and source candidate pool. |
| 2 | Full final evidence router | ScanRefer official, Nr3D | Yes | Main table on full validation sets. |
| 3 | Full input-format overall ablation | ScanRefer official, Nr3D | Yes | Run E0, spatial-only, BEV, 3D-position on all queries. |
| 4 | Full query-type/input interaction | ScanRefer official, Nr3D | No extra if #3 exists | Aggregate full ablation by query type/reason. |
| 5 | Full route contribution | ScanRefer official, Nr3D | No extra if #1-#2 exist | Recovery/regression per route vs E0. |
| 6 | Full statistical tests | ScanRefer official, Nr3D | No extra if #1-#2 exist | Paired confidence intervals and significance tests. |
| 7 | LLM-router full comparison | ScanRefer official, Nr3D | Yes | Replace deterministic router with LLM route classifier. |
| 8 | VLM model-change full comparison | ScanRefer official, Nr3D | Yes | Check whether routing gain transfers across VLMs. |
| 9 | GT proposal diagnostic | ScanRefer official, Nr3D | No VLM for oracle, optional VLM if rerun | Separate proposal ceiling from routing effect. |
| 10 | Runtime/cost measurement | ScanRefer official, Nr3D | No extra if logs are captured | Wall-clock, call count, token count, asset generation time. |
| 11 | Failure/recovery visualization | ScanRefer official, Nr3D | No extra if outputs/assets exist | Qualitative evidence for paper and slides. |

## 6. Recommended Execution Order

### Stage A: API-Free Preprocessing

Status: completed for Nr3D and official ScanRefer val.

1. Build validation-scale route plans and coverage summaries.
2. Verify target category fallback examples.
3. Verify candidate/canvas coverage policy:
   - missing same-class candidate: record as proposal miss / Acc 0.
   - same-class candidate but missing canvas: record as canvas miss / Acc 0 for E0.
   - non-E0 routes can still use geometry if candidate IDs are available, but for
     fair comparison with E0 the candidate pool should be frozen from the E0
     source once full E0 is generated.

### Stage B: Full E0 Source Generation

Run ScanRefer and Nr3D E0 on the audited preprocessed datasets. These outputs
are the canonical candidate-pool source for all routed runs.

Expected outputs:

```text
experiments/full_dataset/scanrefer/e0_baseline_openrouter_qwen.jsonl
experiments/full_dataset/nr3d/e0_baseline_openrouter_qwen.jsonl
```

### Stage C: Final Router Route-First Runs

Use the full E0 outputs and full route-plan parse rows to run only the selected
route per query.

Expected outputs:

```text
experiments/full_dataset/scanrefer/final_router_openrouter_qwen/results.jsonl
experiments/full_dataset/nr3d/final_router_openrouter_qwen/results.jsonl
```

### Stage D: Full Ablations

Run forced-route branches for all queries:

```text
E0
spatial-only text
BEV labeled layout
3D position text
```

The full input-format outputs can support overall ablation, query-type/input
interaction, representation oracle, and route contribution tables.

## 7. Implementation Notes

The current `SeqVLM_evidence_router` result aggregation scripts are designed
around bundled 250-query source outputs. Full-set evaluation needs two
additional execution runners:

1. `run_scanrefer_full_e0_eval.py` / generalized ScanRefer E0 runner.
2. `run_full_final_router_vlm_eval.py` that accepts:
   - full dataset path,
   - full route-plan/parse path,
   - full E0 source path,
   - output directory,
   - resume/start/end options.

The existing Nr3D route-first wrapper can be adapted, but ScanRefer needs a
matching route-first runner because the current final-router recomposition only
uses completed 250-query input-format outputs.
