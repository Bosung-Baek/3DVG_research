# SeqVLM Evidence Router

Standalone repository for the final dictionary evidence-aware routing pipeline.
This is the paper-facing artifact for introduction, inspection, and
reproducibility checks.

The goal is to reproduce the locked ScanRefer and NR3D routing results without
depending on the full SeqVLM experiment workspace. The default evaluator does
not call a VLM or API. It recomposes completed E0 and alternate-input outputs
with a shared dataset-agnostic router.

For a concise verification checklist, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Locked Results

| Dataset | Method | Acc@0.25 | Acc@0.50 | mIoU |
|---|---|---:|---:|---:|
| ScanRefer | E0 baseline | 0.504 | 0.452 | 0.4306 |
| ScanRefer | evidence router | 0.520 | 0.468 | 0.4455 |
| NR3D | E0 baseline | 0.612 | 0.604 | 0.6107 |
| NR3D | evidence router | 0.652 | 0.648 | 0.6514 |

These locked rows are the 250-query paper verification artifacts bundled with
the repository. Full-validation runs are tracked separately under
`experiments/full_dataset_results/`.

## Full-Validation Run Status

The first completed full-validation run is the NR3D final evidence-router run on
7,457 queries. It uses the corrected NR3D object-ID candidate/canvas pipeline
and external OpenRouter Qwen calls.

| Dataset | Run | N | Acc@0.25 | Acc@0.50 | mIoU |
|---|---|---:|---:|---:|---:|
| NR3D val | final evidence router | 7,457 | 0.5958 | 0.5946 | 0.5985 |

Summary files:

- `experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/summary.json`
- `experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/error_analysis.json`

This row is not directly comparable to the 250-query locked baseline above. A
full NR3D E0-only baseline is still required for the full-validation main table.

## Pipeline

```text
query
  -> query type / relation source
  -> dictionary evidence-aware router
  -> selected VLM input result
  -> IoU evaluation
```

Final route policy:

| Priority | Condition | Route |
|---:|---|---|
| 1 | `proximity_derived` | spatial-only text |
| 2 | visual attribute included | E0 RGB canvas |
| 3 | pure `ordinal` | BEV labeled layout |
| 4 | pure `geometric` | 3D position text |
| 5 | `viewpoint_guided` | E0 RGB canvas |
| 6 | default / mixed / ambiguous | E0 RGB canvas |

## Repository Layout

```text
tools/
  query_type_router.py
  evaluate_universal_evidence_router.py
input_formats/
  base.py
  baseline_e0.py
  bev_raw_labeled.py
  seeground_ablation_spatial_only.py
  format_registry.py
inputs/
  scanrefer/
    full_E0_baseline_qwen72b.jsonl
    bev_raw_labeled/results.jsonl
    seeground_ablation_spatial_only/results.jsonl
    seeground_ablation_3dpos_only/results.jsonl
  nr3d/
    official_e0_nr3d_openrouter_qwen_250.jsonl
    nr3d_dfrc_llm_parse.jsonl
    nr3d_query_type_routed_vlm_bev_exact_scanrefer_format/
      nr3d_query_type_routed_vlm_results.jsonl
outputs/
  universal_evidence_router/
data/
  source_outputs/
    scanrefer/
    nr3d/
docs/
  query_type_routing_report_kr.md
```

## Reproduce

Run from this repository root:

```bash
python tools/evaluate_universal_evidence_router.py
```

Or choose an output directory:

```bash
python tools/evaluate_universal_evidence_router.py --out-dir outputs/universal_evidence_router
```

To run the repo-local data generation step first:

```bash
python tools/run_full_pipeline.py --clean-inputs
```

This rebuilds `inputs/` from `data/source_outputs/` and then writes final
metrics to `outputs/universal_evidence_router/`.

Expected summary:

```text
ScanRefer evidence router: Acc@0.25=0.520, Acc@0.50=0.468, mIoU=0.4455
NR3D evidence router:     Acc@0.25=0.652, Acc@0.50=0.648, mIoU=0.6514
```

## Smoke Test

Run the end-to-end smoke test from this repository root:

```bash
python tests/smoke_test_pipeline.py
```

The smoke test checks:

- `data/source_outputs/` can rebuild the canonical `inputs/` layout,
- final input-format modules can build VLM prompts from synthetic scene data,
- the E0 input format can read a generated smoke canvas image,
- representative queries route to the expected input format,
- the evaluator reproduces the locked final JSON/JSONL outputs byte-for-byte.

The GitHub Actions workflow in `.github/workflows/smoke-test.yml` runs the same
test on push and pull request.

## Paper Figures

Paper-facing figures are generated from locked repository-local experiment
artifacts. The figure script does not call a VLM API and does not require private
raw scene assets.

```bash
python tools/create_paper_figures.py
```

Outputs are written to `experiments/figures/` as both `.png` and `.pdf`.
The generated figures cover the pipeline overview, main table, input-format
ablation, query-type/input interaction, route contribution, router ablations,
LLM-router comparison, oracle gap, rerun variance, and representative
failure/recovery cases.

## Paper Experiment Suite

The repo is scoped to the paper experiments below.

```bash
python tools/run_experiment_suite.py --out-dir experiments
```

Generated files:

- `experiments/main_table/table.json`
- `experiments/ablation/input_format_overall_scanrefer.json`
- `experiments/ablation/input_format_by_query_type_scanrefer.json`
- `experiments/ablation/optional_experiments.json`
- `experiments/summary.json`

Immediately runnable with the bundled data pack:

- ScanRefer zero-shot main table row
- NR3D zero-shot main table row
- Overall performance by VLM input format on ScanRefer
- Query-type by input-format performance on ScanRefer

Additional ablations are already summarized under `experiments/`. The suite can
also accept externally completed result files:

```bash
python tools/run_experiment_suite.py \
  --llm-router-result path/to/llm_router_results.jsonl \
  --vlm-model-result path/to/alternate_vlm_results.jsonl \
  --gt-proposal-result path/to/gt_proposal_results.jsonl \
  --runtime-result path/to/runtime_results.jsonl \
  --failure-case-result path/to/failure_cases.jsonl
```

This keeps the repo focused on paper verification without pretending to
regenerate unavailable raw source outputs.

### LLM Router Ablation

The first optional ablation replaces the deterministic dictionary router with an
LLM route classifier.

No-API smoke run:

```bash
python tools/run_llm_router_ablation.py \
  --mock-dictionary \
  --out-dir experiments/ablation/llm_router_mock
```

Actual OpenRouter run with an external API config:

```bash
python tools/run_llm_router_ablation.py \
  --config path/to/private_config.yaml \
  --config-alias openrouter-qwen \
  --out-dir experiments/ablation/llm_router_priority_openrouter_qwen \
  --quiet
```

Then include it in the suite:

```bash
python tools/run_experiment_suite.py \
  --llm-router-result experiments/ablation/llm_router_priority_openrouter_qwen/summary.json
```

The LLM ablation uses the same conservative priority order as the dictionary
router, but lets the LLM judge whether each query is proximity, visual/mixed,
pure ordinal, pure geometric, or viewpoint/local-frame dependent.

Current `openrouter-qwen` priority-router result:

| Dataset | Dictionary router | LLM router |
|---|---:|---:|
| ScanRefer Acc@0.25 | 0.520 | 0.492 |
| NR3D Acc@0.25 | 0.652 | 0.664 |

Notes:

- ScanRefer can evaluate any LLM-selected route because all final input-format
  outputs are bundled.
- NR3D can only evaluate non-E0 LLM routes when a completed source output exists
  for that case/route. If not, the script records
  `source_unavailable_fallback_e0` and falls back to E0 for that case.
- The API key is loaded from the external config at runtime and is not copied
  into this standalone repo or written to logs.

## Included Modules

This repo includes both parts of the proposed pipeline:

1. Routing module
   - `tools/query_type_router.py`
   - Decides whether a query should use E0, spatial-only text, BEV, or 3D
     position text.

2. Input-format generation modules
   - `input_formats/baseline_e0.py`: SeqVLM RGB canvas input.
   - `input_formats/bev_raw_labeled.py`: BEV image + coordinate prompt input.
   - `input_formats/seeground_ablation_spatial_only.py`: text-only 3D spatial
     prompt used for spatial-only / 3D-position routes.
   - `input_formats/format_registry.py`: final registry exposing only the
     formats used by the evidence router.

The locked evaluator uses completed VLM outputs under `inputs/` because the
final numbers are a recomposition benchmark. To regenerate fresh VLM inputs from
raw scenes, the format modules require the same external scene assets used by
the original experiment:

- pre-rendered SeqVLM candidate canvases for E0,
- ScanNet / Mask3D scene assets for BEV rendering,
- candidate `locs`, labels, anchors, and scene metadata for text prompts.

The repo-local `data/source_outputs/` folder is the portable data pack used to
rebuild evaluator inputs and final metrics. It is not a replacement for the full
ScanNet/Mask3D raw asset tree needed to regenerate every VLM image from scratch.

## Notes

- This repo intentionally keeps only the source outputs required for the final
  evidence router.
- Baseline outputs are retained for comparison and for per-query E0 recovery /
  regression analysis.
- Deprecated broad-routing, LLM-router, smoke-run, and unrelated ablation
  artifacts are excluded.
- No API keys or private credentials are stored in the repository.
