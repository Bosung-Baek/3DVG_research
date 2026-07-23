# Full-Dataset Experiment Checklist

This checklist tracks the experiments needed for full validation-set evaluation
on official ScanRefer val and Nr3D val.

## Dataset Scope

| Dataset | Source | N | Status |
|---|---|---:|---|
| ScanRefer val | `/data/knuvi/bosung/scanrefer/ScanRefer_filtered_val.json` | 9,508 | source found |
| Nr3D val | `SeqVLM/data/nr3d_val_with_obj_name.json` | 7,457 | source found |

Do not use `SeqVLM/data/ZSVG3D/data/scanrefer_val.json` as official ScanRefer
full val. That file has 9,498 rows and is a converted ZSVG protocol file.

## Stage 0: API-Free Preprocessing

- [x] Locate official ScanRefer val source.
- [x] Count official ScanRefer val queries: 9,508.
- [x] Count Nr3D val queries: 7,457.
- [x] Generate full route plans and coverage summaries.
- [x] Save official ScanRefer route plan:
  `experiments/full_dataset_preprocessing_v2/scanrefer_official_val_preprocessed.jsonl`
- [x] Save Nr3D route plan:
  `experiments/full_dataset_preprocessing_v2/nr3d_full_preprocessed.jsonl`
- [x] Save combined preprocessing summary:
  `experiments/full_dataset_preprocessing_v2/summary.json`
- [x] Export coverage-miss lists.
- [x] Audit coverage-miss examples in summary.
- [ ] Decide official policy for no-candidate and no-canvas cases.
- [x] Generate sample VLM input packages with prompts/images.
- [x] Validate sample VLM input packages.
- [x] Materialize full VLM input packages on the data disk.
- [x] Validate full VLM input packages.

Current route counts:

| Dataset | E0 | spatial-only | 3D position | BEV |
|---|---:|---:|---:|---:|
| ScanRefer val | 7,279 | 1,664 | 346 | 219 |
| Nr3D val | 3,982 | 3,015 | 350 | 110 |

Current coverage:

| Dataset | Same-class canvas candidate | No same-class candidate | Candidate exists but no canvas |
|---|---:|---:|---:|
| ScanRefer val | 7,232 | 1,194 | 1,082 |
| Nr3D val | 7,457 | 0 | 0 |

Coverage miss files:

| Dataset | Path | N |
|---|---|---:|
| ScanRefer val | `experiments/full_dataset_preprocessing_v2/scanrefer_official_val_coverage_misses.jsonl` | 2,276 |
| Nr3D val | `experiments/full_dataset_preprocessing_v2/nr3d_full_coverage_misses.jsonl` | 0 |

Each preprocessed row now includes:

- `candidate_coverage_status`
- `pre_e0_same_class_candidate_ids`
- `pre_e0_canvas_candidate_ids`
- `num_same_class_candidates`
- `num_canvas_candidates`

Materialized VLM input packages:

| Dataset | Path | Manifests | OK | No candidates |
|---|---|---:|---:|---:|
| ScanRefer val | `/data/knuvi/bosung/evidence_router_vlm_inputs_v2/scanrefer/` | 9,508 | 7,456 | 2,052 |
| Nr3D val | `/data/knuvi/bosung/evidence_router_vlm_inputs_v2/nr3d/` | 7,457 | 7,457 | 0 |

Validation summary:

`/data/knuvi/bosung/evidence_router_vlm_inputs_v2/materialization_validation_summary.json`

Each case directory contains a `manifest.json`, `system.txt`, and either
`prompt.txt` or `prompt_template.txt`. BEV routes include rendered
`images/bev_0.jpg`; E0 routes include symlinked `images/candidate_*.jpg`
pointing to the existing SeqVLM canvas assets.

## Stage 1: Full E0 Baselines

- [ ] Implement or adapt full ScanRefer E0 runner.
- [ ] Implement or adapt full Nr3D E0 runner.
- [ ] Ensure each output row stores:
  `case`, `dataset`, `scan_id`, `target_id`, `caption`, `obj_name`,
  `candidate_ids`, `selected_id`, `iou`, `acc25`, `acc50`, `miou`.
- [ ] Freeze candidate pools from E0 outputs for routed comparisons.
- [ ] Run ScanRefer full E0 baseline with resume support.
- [ ] Run Nr3D full E0 baseline with resume support.
- [ ] Save outputs:
  `experiments/full_dataset/scanrefer/e0_baseline_openrouter_qwen/results.jsonl`
- [ ] Save outputs:
  `experiments/full_dataset/nr3d/e0_baseline_openrouter_qwen/results.jsonl`
- [ ] Generate E0 summary tables.

## Stage 2: Full Final Evidence Router

- [x] Implement generalized materialized-input route-first runner.
- [x] Load route plan from preprocessing output.
- [x] For spatial-only route, generate structured candidate/anchor text.
- [x] For BEV route, generate BEV labeled layout plus coordinate prompt.
- [x] For 3D-position route, generate structured position text.
- [ ] Run ScanRefer final router.
- [x] Run Nr3D final router.
- [ ] Save outputs:
  `experiments/full_dataset/scanrefer/final_router_openrouter_qwen/results.jsonl`
- [x] Save lightweight Nr3D summary:
  `experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/summary.json`
- [x] Verify Nr3D output row count equals dataset count.
- [ ] Load frozen candidate pool from full E0 output.
- [ ] For E0 route, reuse full E0 result or re-run only if explicitly requested.
- [ ] Verify routed candidate pool matches full E0 candidate pool.
- [ ] Generate main table against full E0 baseline.

Completed Nr3D final-router result:

| Dataset | Run | N | Acc@0.25 | Acc@0.50 | mIoU | Notes |
|---|---|---:|---:|---:|---:|---|
| Nr3D val | final router, OpenRouter Qwen | 7,457 | 0.5958 | 0.5946 | 0.5985 | 74 error rows; see `experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/error_analysis.json` |

## Stage 3: Full Input-Format Ablation

Run each input format on every query, using the same candidate pool.

- [ ] ScanRefer full E0.
- [ ] ScanRefer full spatial-only text.
- [ ] ScanRefer full BEV labeled layout.
- [ ] ScanRefer full 3D-position text.
- [ ] Nr3D full E0.
- [ ] Nr3D full spatial-only text.
- [ ] Nr3D full BEV labeled layout.
- [ ] Nr3D full 3D-position text.
- [ ] Save per-format result files under:
  `experiments/full_dataset/<dataset>/input_format_ablation/`
- [ ] Aggregate overall input-format table.
- [ ] Aggregate query-type by input-format table.
- [ ] Compute representation oracle.

## Stage 4: Router Ablations

- [ ] Full deterministic evidence router.
- [ ] Proximity-only router.
- [ ] Proximity + pure ordinal router.
- [ ] Proximity + pure geometric router.
- [ ] Broad query-type router.
- [ ] No visual fallback.
- [ ] No purity constraint.
- [ ] No viewpoint fallback.
- [ ] No priority ordering.
- [ ] Aggregate Acc@0.25, Acc@0.50, mIoU.
- [ ] Aggregate recovery/regression versus E0.

## Stage 5: LLM Router Ablation

- [ ] Reuse priority prompt matching final dictionary decision order.
- [ ] Run LLM route classification for ScanRefer full val.
- [ ] Run LLM route classification for Nr3D full val.
- [ ] Recompose or route-first evaluate LLM-selected routes.
- [ ] Compute dictionary-vs-LLM transition matrix.
- [ ] Analyze harmful transitions, especially `E0 -> spatial-only`.
- [ ] Save under:
  `experiments/full_dataset/ablation/llm_router_priority_openrouter_qwen/`

## Stage 6: VLM Model Change

- [ ] Choose alternate VLM model list.
- [ ] Run full non-E0 branches with alternate VLM.
- [ ] If budget allows, run full E0 baseline with the same alternate VLM.
- [ ] Compare routing gain under each VLM.
- [ ] Save under:
  `experiments/full_dataset/ablation/vlm_model_change/`

## Stage 7: GT Proposal / Proposal Ceiling

- [ ] Compute proposal oracle Acc@0.25/Acc@0.50/mIoU for ScanRefer.
- [ ] Compute proposal oracle Acc@0.25/Acc@0.50/mIoU for Nr3D.
- [ ] Count zero-candidate and low-IoU candidate cases.
- [ ] If GT-proposal VLM evaluation is needed, run E0 and final router with GT
  proposal candidate pools.
- [ ] Save under:
  `experiments/full_dataset/gt_proposal/`

## Stage 8: Runtime and Cost

- [ ] Log preprocessing time.
- [ ] Log input-generation time per route.
- [ ] Log VLM call count per method.
- [ ] Log token counts for text-only routes if available.
- [ ] Log image count and image resolution per route.
- [ ] Report cost proxy:
  E0 baseline, dictionary router, LLM router, full ablations.
- [ ] Save under:
  `experiments/full_dataset/runtime/`

## Stage 9: Failure / Recovery Visualization

- [ ] Select E0 failure, routed success cases.
- [ ] Select E0 success, routed failure cases.
- [ ] Include at least:
  proximity spatial-only recovery,
  pure ordinal BEV recovery,
  geometric 3D-position recovery,
  visual fallback case.
- [ ] Render actual mesh/BEV scene overlays where available.
- [ ] Generate presentation-ready figures.
- [ ] Save under:
  `experiments/full_dataset/failure_visualization/`

## Stage 10: Final Reporting

- [ ] Main table: ScanRefer full and Nr3D full.
- [ ] Input-format overall ablation.
- [ ] Query-type/input interaction.
- [ ] Router ablation.
- [ ] LLM router comparison.
- [ ] VLM model-change comparison.
- [ ] GT proposal diagnostic.
- [ ] Runtime/cost table.
- [ ] Failure/recovery visualizations.
- [ ] Statistical tests and confidence intervals.
- [ ] Update `experiments/README.md`.
- [ ] Update repository `README.md`.

## Blocking Items

| Item | Why it matters |
|---|---|
| Full E0 runners | Needed to freeze candidate pools and compute full baselines. |
| Full route-first runner | Needed for final full evidence-router evaluation. |
| Full input-format source outputs | Needed for complete ablation and oracle analysis. |
| Canvas coverage policy | Needed to fairly handle cases where candidates exist but rendered canvas is missing. |
| API budget and retry policy | Needed because full datasets require many VLM calls. |
