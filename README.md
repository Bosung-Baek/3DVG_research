# Evidence-Aware Routing for Zero-Shot 3D Visual Grounding

This repository hosts the code and reproducibility artifact for our current
zero-shot 3D visual grounding study. The project builds on the SeqVLM-style
candidate selection pipeline and studies a simple question:

> Should every referring expression be sent to a VLM with the same visual input,
> or should the input representation change according to the evidence required
> by the query?

Our answer is an **evidence-aware router**: a deterministic routing module that
keeps RGB canvas input for visual or mixed queries, but switches to structured
spatial input when the query mainly asks for distance, proximity, or pure layout
relations.

## What Is New

The original SeqVLM pipeline uses a candidate-centered RGB canvas as the default
VLM input. We keep that baseline and add a routing layer before VLM selection.
The router decides which representation should be used for each query while
keeping the same candidate pool.

The paper-facing artifact is located at:

```text
SeqVLM_evidence_router/
```

It contains:

- the final evidence-aware router,
- input-format modules,
- bundled source outputs for ScanRefer and NR3D verification,
- main-table and ablation summaries,
- smoke tests for reproducing the locked results,
- documentation for the current TMM-oriented experiment plan.

## Key Terms

**3D visual grounding**  
Given a natural-language query and a 3D scene, the task is to identify the target
object instance described by the query.

**Zero-shot setting**  
The pipeline uses a VLM for inference without training a dataset-specific 3D
grounding model on the target benchmark split.

**E0 RGB canvas**  
The baseline SeqVLM-style input. Candidate objects are shown through RGB
multi-view canvas images, and the VLM selects the target candidate.

**Evidence-aware routing**  
A rule-based decision step that chooses the VLM input representation according
to the type of evidence needed by the query. The router does not use the dataset
name as a decision variable.

**Spatial-only text**  
A text prompt containing candidate categories, 3D centers, sizes, anchor-object
offsets, and distances. It intentionally avoids RGB appearance evidence.

**BEV labeled layout**  
A bird's-eye-view layout image with candidate labels, paired with coordinate
text. It is used only for pure ordinal/layout queries in the final router.

**3D position text**  
A structured text representation of candidate and anchor 3D locations. It is
reserved for pure geometric relations such as `between`, `under`, or `inside`.

**Fallback to E0**  
When a query contains visual attributes, viewpoint-local relations, or ambiguous
mixed evidence, the router conservatively keeps the RGB canvas baseline. This
prevents spatial-only inputs from discarding necessary visual information.

**Recomposition evaluation**  
The bundled verification benchmark recomposes already completed VLM source
outputs according to the final router. This isolates the effect of routing and
input representation while keeping the proposal/candidate pool fixed.

## Final Router

Implementation:

```text
SeqVLM_evidence_router/tools/query_type_router.py
```

Routing priority:

| Priority | Condition | Route | Rationale |
|---:|---|---|---|
| 1 | `proximity_derived` | spatial-only text | Distance and proximity cues are directly represented by coordinates and anchor distances. |
| 2 | visual attribute included | E0 RGB canvas | Color, material, shape, object state, and object-on-top cues require visual evidence. |
| 3 | pure `ordinal` | BEV labeled layout | Pure order/layout queries can benefit from a top-down view. |
| 4 | pure `geometric` | 3D position text | Pure geometric relations are naturally expressed by structured 3D coordinates. |
| 5 | `viewpoint_guided` | E0 RGB canvas | Current BEV is global-frame, not viewer-local-frame. |
| 6 | default / mixed / ambiguous | E0 RGB canvas | Conservative fallback reduces routing-induced regressions. |

## Main Results

The locked 250-query verification results are:

| Dataset | Method | Acc@0.25 | Acc@0.50 | mIoU |
|---|---|---:|---:|---:|
| ScanRefer | E0 baseline | 0.504 | 0.452 | 0.4306 |
| ScanRefer | evidence router | **0.520** | **0.468** | **0.4455** |
| NR3D | E0 baseline | 0.612 | 0.604 | 0.6107 |
| NR3D | evidence router | **0.652** | **0.648** | **0.6514** |

The same routing policy improves both datasets without dataset-specific
calibration.

## Reproduce the Locked Results

```bash
cd SeqVLM_evidence_router
pip install -r requirements.txt
python tests/smoke_test_pipeline.py
```

Expected output includes:

```text
SMOKE TEST PASSED
ScanRefer evidence router: Acc@0.25=0.520, Acc@0.50=0.468, mIoU=0.4455
NR3D evidence router:     Acc@0.25=0.652, Acc@0.50=0.648, mIoU=0.6514
```

For details, see:

- `SeqVLM_evidence_router/REPRODUCIBILITY.md`
- `SeqVLM_evidence_router/experiments/README.md`
- `SeqVLM_evidence_router/experiments/TMM_EXPERIMENT_PLAN.md`
- `SeqVLM_evidence_router/AGENT_CONTEXT.md`

## Generate Paper Figures

The paper-facing figures are generated from locked experiment artifacts and are
stored in `SeqVLM_evidence_router/experiments/figures/`.

```bash
cd SeqVLM_evidence_router
python tools/create_paper_figures.py
```

## Repository Layout

```text
.
├── SeqVLM_evidence_router/   # Paper-facing artifact and locked experiments
├── seqvlm/                   # Existing SeqVLM-based research code
├── preprocess/               # Existing preprocessing scripts
├── prompts/                  # Existing prompt templates
├── visprog/                  # Existing visual-programming utilities
└── data/                     # Existing compact metadata/examples
```

## Scope and Limitations

The artifact is intended for paper introduction and result verification. It
bundles completed source outputs and compact diagnostics. It does not include
private API keys, the full ScanNet/Mask3D raw asset tree, or every intermediate
canvas generated during the original development workspace.

Fresh end-to-end VLM runs and raw canvas/BEV regeneration require external scene
assets and private API configuration. The included smoke test and recomposition
scripts are the canonical lightweight verification path for the reported
results.

## Acknowledgments

This work builds on ideas and tooling from SeqVLM, ZSVG3D, SeeGround, and related
zero-shot 3D visual grounding pipelines. The evidence-aware routing artifact is
provided to make the current routing experiments inspectable and reproducible.
