# NR3D Full Validation Final-Router Run

This directory stores lightweight summaries for the NR3D full validation
evidence-router run. The full per-query output is intentionally kept on the data
disk because it is a generated large artifact.

## Source Run

| Item | Value |
|---|---|
| Dataset | NR3D validation split |
| N | 7,457 |
| VLM | OpenRouter Qwen |
| Input root | `/data/knuvi/bosung/evidence_router_vlm_inputs_v2` |
| Full output root | `/data/knuvi/bosung/evidence_router_full_runs_v2/nr3d/final_router/openrouter-qwen/` |
| Local summary copied here | `summary.json` |
| Error analysis copied here | `error_analysis.json` |

## Result

| Metric | Value |
|---|---:|
| Acc@0.25 | 0.5958 |
| Acc@0.50 | 0.5946 |
| mIoU | 0.5985 |
| Processed | 7,457 / 7,457 |
| Status OK | 7,383 |
| Status error | 74 |
| API calls | 8,841 |
| Elapsed time | 19,755.24 seconds |

## Route Breakdown

| Route | N | Acc@0.25 | Acc@0.50 | mIoU |
|---|---:|---:|---:|---:|
| E0 RGB canvas | 3,982 | 0.5497 | 0.5482 | 0.5532 |
| spatial-only text | 3,015 | 0.6594 | 0.6584 | 0.6609 |
| 3D position text | 350 | 0.6114 | 0.6114 | 0.6137 |
| BEV labeled layout | 110 | 0.4727 | 0.4727 | 0.4796 |

## Notes

- This is a full validation-set final-router run, not the locked 250-query main
  table result.
- It should not be compared directly with the 250-query NR3D E0 baseline.
  A full NR3D E0-only baseline is still needed for a full-validation main table.
- The 74 error rows are mostly upstream OpenRouter rate-limit/timeout cases and
  E0 tournament postprocessing errors. See `error_analysis.json`.
