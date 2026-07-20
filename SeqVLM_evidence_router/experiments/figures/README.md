# Paper Figures

This directory contains paper-facing visualizations generated from locked
experiment artifacts. The figures are derived from repository-local JSON/JSONL
files and do not require VLM API calls or private raw scene assets.

## Generated Figures

| File stem | Purpose |
|---|---|
| `fig1_pipeline_overview` | Overview of evidence-aware routing and input-format selection. |
| `fig2_main_results` | Main ScanRefer/NR3D comparison against E0 baseline. |
| `fig3_input_format_ablation_scanrefer` | Shows that blindly applying one non-E0 input to every ScanRefer query is worse than E0. |
| `fig4_query_type_input_heatmap_scanrefer` | Query-type / input-format interaction heatmap. |
| `fig5_route_distribution_contribution` | Final route distribution and recovery/regression contribution. |
| `fig6_policy_component_ablation` | Cumulative policy and router-component ablations. |
| `fig7_llm_router_comparison` | Dictionary router vs priority-prompted LLM router. |
| `fig8_oracle_and_rerun` | Representation oracle gap and NR3D end-to-end rerun variance. |
| `fig9_failure_case_summary` | Lightweight representative recovery/regression/fallback cases. |

Every figure is saved as both `.png` and `.pdf`.

Regenerate all figures:

```bash
python tools/create_paper_figures.py
```
