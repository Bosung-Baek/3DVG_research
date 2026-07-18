# TMM Experiment Plan

이 문서는 TMM 제출을 목표로 현재 evidence-aware routing 결과를 어떻게 보강할지
정리한 실행 계획이다.

현재 확보된 핵심 결과는 다음과 같다.

| Dataset | E0 baseline | Evidence router | Gain |
|---|---:|---:|---:|
| ScanRefer Acc@0.25 | 0.504 | 0.520 | +0.016 |
| NR3D Acc@0.25 | 0.612 | 0.652 | +0.040 |

현재 main result는 유효하지만, reviewer가 제기할 수 있는 가장 큰 질문은 다음이다.

1. 최종 성능 향상이 사실상 `proximity -> spatial-only` 하나로 설명되는가?
2. BEV/3D position branch가 실제로 필요한가?
3. Dictionary/evidence router가 임의적 heuristic 아닌가?
4. 250-query recomposition 결과가 end-to-end pipeline에서도 유지되는가?
5. 통계적으로 얼마나 안정적인가?

따라서 우선순위는 새 input representation 추가가 아니라, 현재 pipeline을 고정한
상태에서 ablation과 검증을 촘촘하게 만드는 것이다.

## Phase 1: No-API Experiments

현재 bundled source outputs만으로 바로 만들 수 있는 실험이다. 가장 먼저 진행한다.

### 1. Policy Cumulative Ablation

목적: full router의 gain이 proximity-only로 전부 설명되는지 확인한다.

| Variant | 설명 |
|---|---|
| E0 only | 모든 query를 E0로 처리 |
| Proximity-only | `proximity_derived`만 spatial-only, 나머지 E0 |
| Proximity + ordinal | proximity + pure ordinal BEV |
| Proximity + geometric | proximity + pure geometric 3D position |
| Proximity + ordinal + geometric | visual fallback 없이/또는 제한적으로 multi-branch 구성 |
| Full router | 현재 최종 evidence-aware router |

각 variant에서 저장할 metric:

- Acc@0.25
- Acc@0.50
- mIoU
- Recovery
- Regression
- Net gain
- Route counts
- Route reason counts

판단 기준:

- Full router와 proximity-only가 거의 같으면, 핵심 contribution을 proximity
  evidence routing으로 좁혀야 한다.
- BEV/3D branch가 추가 net recovery를 만들면 multi-representation router 주장이
  강화된다.

필요 산출물:

- `experiments/ablation/policy_cumulative/summary.json`
- `experiments/ablation/policy_cumulative/scanrefer_results.jsonl`
- `experiments/ablation/policy_cumulative/nr3d_results.jsonl`
- README 표 업데이트

### 2. Route-Level Recovery/Regression

목적: 어떤 route가 몇 건을 살리고 몇 건을 망쳤는지 명확히 제시한다.

| Dataset | Route | Routed N | E0 correct | Route correct | Recovery | Regression | Net |
|---|---|---:|---:|---:|---:|---:|---:|
| ScanRefer | spatial-only | 37 |  |  |  |  |  |
| ScanRefer | BEV | 1 |  |  |  |  |  |
| ScanRefer | 3D position | 4 |  |  |  |  |  |
| NR3D | spatial-only | 95 |  |  |  |  |  |
| NR3D | BEV | 6 |  |  |  |  |  |

query-level transition 파일도 함께 저장한다.

필드:

- query id
- query
- query/evidence type
- selected route
- route reason
- E0 IoU / E0 correct
- route IoU / route correct
- recovery/regression/unchanged

필요 산출물:

- `experiments/ablation/route_contribution/summary.json`
- `experiments/ablation/route_contribution/transitions.jsonl`

### 3. Statistical Significance

목적: 같은 250개 query의 paired comparison으로 gain의 신뢰도를 보고한다.

Acc@0.25 / Acc@0.50:

- McNemar test
- paired bootstrap 95% confidence interval
- exact binomial test as auxiliary

mIoU:

- query-level paired bootstrap 95% confidence interval
- paired permutation test

보고 표:

| Dataset | Metric | Gain | 95% CI | p-value |
|---|---|---:|---:|---:|
| ScanRefer | Acc@0.25 | +0.016 |  |  |
| ScanRefer | mIoU | +0.0149 |  |  |
| NR3D | Acc@0.25 | +0.040 |  |  |
| NR3D | mIoU | +0.0407 |  |  |

주의:

- ScanRefer는 net +4건이라 유의성이 약할 수 있다.
- 숨기기보다 confidence interval과 route-level evidence를 함께 제시한다.

필요 산출물:

- `experiments/statistics/paired_tests.json`
- `experiments/statistics/bootstrap_samples.jsonl` 또는 seed-fixed summary

### 4. Router Component Ablation

목적: router가 임의적인 keyword heuristic이 아니라 evidence-preserving decision
policy임을 보인다.

| Variant | 설명 | 예상 |
|---|---|---|
| Full router | 최종 방식 | baseline for ablation |
| Without visual fallback | visual cue가 있어도 spatial/ordinal/geometric route 허용 | mixed query regression 증가 |
| Without purity constraint | ordinal/geometric label이면 무조건 BEV/3D | broad routing regression 증가 |
| Without viewpoint fallback | viewpoint query를 BEV/3D로 보냄 | local/global frame mismatch 증가 |
| Without priority ordering | evidence condition을 독립 처리 | route instability 증가 |

필요 산출물:

- `experiments/ablation/router_components/summary.json`
- variant별 route distribution과 recovery/regression

### 5. Representation Oracle

목적: 현재 router가 가능한 upper bound 중 얼마나 회수했는지 확인한다.

ScanRefer는 E0, BEV, spatial-only, 3D position 결과가 모두 있으므로 바로 가능하다.
NR3D는 현재 route-first source output이 제한적이므로 available-source oracle로
표기해야 한다.

보고 표:

| Dataset | E0 | Final router | Available oracle |
|---|---:|---:|---:|
| ScanRefer | 0.504 | 0.520 |  |
| NR3D | 0.612 | 0.652 |  |

추가 지표:

- E0 fail, any non-E0 success
- router recovered / oracle-recoverable
- missed oracle recoveries
- oracle에서 어느 representation이 자주 기여하는지

필요 산출물:

- `experiments/ablation/representation_oracle/summary.json`
- `experiments/ablation/representation_oracle/oracle_cases.jsonl`

### 6. LLM Router Detailed Comparison

현재 이미 priority LLM router 결과가 있다.

보강할 분석:

- route transition matrix
- dictionary-LLM agreement rate
- evidence type별 agreement
- mixed query over-routing 비율
- LLM latency/cost, dictionary latency/cost

이미 있는 산출물:

- `experiments/ablation/llm_router_priority_openrouter_qwen/summary.json`
- `experiments/ablation/llm_router_priority_openrouter_qwen/route_diff_analysis.md`

추가 산출물:

- `experiments/ablation/llm_router_priority_openrouter_qwen/transition_matrix.json`

## Phase 2: Small Additional Runs

소량의 API 또는 수동 검토가 필요한 실험이다. Phase 1 이후 진행한다.

### 7. End-to-End Consistency Check

목적: 현재 recomposition 결과가 실제 route-first pipeline과 일치하는지 확인한다.

추천 subset:

- E0 route 20개
- spatial-only route 20개
- BEV/3D route 전부
- recovery 사례
- regression 사례

확인할 사항:

- selected route만 실제로 생성되는지
- VLM call이 branch당 1회인지
- recomposition source output과 end-to-end output의 일치율
- API decoding randomness
- parser/source missing case

필요 산출물:

- `experiments/end_to_end_smoke/summary.json`
- `experiments/end_to_end_smoke/results.jsonl`

### 8. Failure-Case Visualization

목적: 정량 결과의 설득력을 높인다.

최소 네 종류를 준비한다.

1. E0 실패 -> spatial-only 성공
2. Visual fallback이 regression을 막은 사례
3. BEV 또는 3D position 성공 사례
4. Router 실패 사례

각 figure 구성:

- query
- evidence extracted
- E0 RGB canvas
- selected alternative representation
- E0 prediction / routed prediction / GT
- 왜 성공/실패했는지 짧은 설명

필요 산출물:

- `experiments/failure_visualization/cases.jsonl`
- `experiments/failure_visualization/figures/*.png`

### 9. Evidence Extraction Manual Audit

목적: query/evidence type이 임의적이라는 지적을 완화한다.

라벨:

- appearance evidence
- proximity evidence
- ordinal evidence
- geometric evidence
- viewpoint-dependent evidence
- mixed evidence

우선 subset 100개씩만 해도 좋다.

보고 지표:

- precision
- recall
- F1
- false positive examples
- false negative examples

필요 산출물:

- `experiments/evidence_audit/manual_labels.csv`
- `experiments/evidence_audit/summary.json`

## Phase 3: High-Cost Competitiveness Experiments

TMM 경쟁력을 높이는 실험이지만, 비용과 시간이 크므로 Phase 1 결과를 본 뒤 진행한다.

### 10. VLM Model Generalization

목적: routing gain이 특정 VLM에 종속적인지 확인한다.

최소 구성:

- 현재 main VLM
- 작은 동일 계열 모델
- 다른 계열 VLM 하나

전체 input-format ablation은 비용이 크므로 다음만 실행한다.

| VLM | Dataset | E0 | spatial-only overall | final router | Gain |
|---|---|---:|---:|---:|---:|

### 11. Runtime and API Cost

목적: deterministic router의 practical advantage를 보인다.

| Method | Router cost | Representation generation | VLM calls | Input tokens/images | Latency |
|---|---|---|---:|---:|---:|
| E0 | none | RGB canvas | 1 |  |  |
| Dictionary router | negligible | selected branch | 1 |  |  |
| LLM router | extra LLM call | selected branch | 2 |  |  |

### 12. GT Proposal / Oracle Candidate

목적: proposal error와 representation selection effect를 분리한다.

우선순위는 낮다. NR3D가 이미 비교적 candidate-controlled protocol이므로,
Phase 1~2 이후 여력이 있을 때 진행한다.

### 13. Additional Dataset or Expanded Subset

옵션:

- Sr3D 250-query subset
- ScanRefer/NR3D 다른 random seed subset
- proximity query만 full split에서 확장 평가

비용 효율만 보면 proximity subset 확장이 가장 좋다.

## Immediate Next Steps

가장 먼저 해야 할 세 가지는 다음이다.

1. `policy_cumulative` 생성
   - E0 vs proximity-only vs full router를 먼저 확인한다.
   - 이 결과로 방법의 중심을 multi-branch router로 둘지, proximity evidence
     routing으로 좁힐지 결정한다.

2. `route_contribution` 생성
   - route별 recovery/regression/net을 표로 만든다.
   - reviewer가 “어떤 branch가 실제로 기여했는가?”라고 물을 때 직접 답할 수 있다.

3. `paired_statistics` 생성
   - McNemar, bootstrap CI, permutation test를 저장한다.
   - ScanRefer의 작은 gain을 정직하게 보고하고, NR3D의 안정적 gain을 강조한다.

## Implementation Plan

새 스크립트는 다음 순서로 추가한다.

1. `tools/run_policy_ablation.py`
   - 여러 router variant를 source-output recomposition으로 평가
   - `policy_cumulative`, `router_components` 둘 다 지원

2. `tools/analyze_route_contribution.py`
   - final routed results와 E0를 비교해 route별 recovery/regression 계산

3. `tools/run_paired_statistics.py`
   - paired binary metrics와 mIoU bootstrap/permutation 계산

4. `tools/run_representation_oracle.py`
   - available source outputs 기준 oracle upper bound 계산

5. `tools/build_failure_visualizations.py`
   - recovery/regression case를 figure로 저장

각 스크립트는 `experiments/` 아래에 결과를 저장하고,
`tools/run_experiment_suite.py`가 summary에 포함할 수 있도록 JSON schema를 맞춘다.

## Progress Log

현재 완료된 항목:

| Item | Status | Output |
|---|---|---|
| Policy cumulative ablation | done | `experiments/ablation/policy_ablation/summary.json` |
| Router component ablation | done | `experiments/ablation/policy_ablation/summary.json` |
| Route-level recovery/regression | done | `experiments/ablation/route_contribution/summary.json` |
| Paired statistical tests | done | `experiments/statistics/paired_tests.json` |
| Representation oracle | done | `experiments/ablation/representation_oracle/summary.json` |
| LLM router comparison | done | `experiments/ablation/llm_router_priority_openrouter_qwen/summary.json` |
| LLM transition matrix | done | `experiments/ablation/llm_router_priority_openrouter_qwen/transition_matrix.json` |
| NR3D end-to-end final-router rerun | done | `experiments/end_to_end_nr3d_final_router_openrouter_qwen/summary.json` |
| NR3D end-to-end repeated rerun | done | `experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat_summary.json` |
| NR3D missing-branch VLM fill-in | done | `experiments/ablation/nr3d_missing_branches/openrouter_qwen/summary.json` |
| GT proposal / candidate oracle diagnostic | done | `experiments/gt_proposal/summary.json` |
| VLM model-change branch sensitivity | done | `experiments/ablation/vlm_model_change/summary.json` |
| Runtime/cost proxy | done | `experiments/runtime/runtime_proxy.json` |
| Failure/recovery case list | done | `experiments/failure_visualization/cases.jsonl` |
| Evidence audit proxy | done | `experiments/evidence_audit/summary.json` |

새로 추가된 스크립트:

| Script | Purpose |
|---|---|
| `tools/run_policy_ablation.py` | policy cumulative + router component ablation |
| `tools/analyze_route_contribution.py` | final route별 recovery/regression |
| `tools/run_paired_statistics.py` | McNemar, bootstrap CI, permutation test |
| `tools/run_representation_oracle.py` | available representation oracle |
| `tools/run_nr3d_final_router_vlm_eval.py` | original SeqVLM NR3D VLM runner patched with final router |
| `tools/analyze_end_to_end_consistency.py` | compare end-to-end rerun with recomposition result |
| `tools/finalize_tmm_experiments.py` | router components, LLM transitions, runtime proxy, failure cases, evidence audit |
| `tools/run_nr3d_forced_route_vlm_eval.py` | force one NR3D input format over all queries |
| `tools/analyze_nr3d_full_branches.py` | summarize NR3D all-input fill-in and full oracle |
| `tools/run_nr3d_final_router_openrouter_model.py` | run final router with alternate OpenRouter branch model |
| `tools/summarize_gt_proposal_oracle.py` | summarize Mask3D proposal oracle CSV |
| `tools/summarize_additional_experiments.py` | collect high-cost experiment summaries |

현재 결과에서 확인된 핵심:

1. ScanRefer full router는 Acc@0.25 기준 proximity-only와 동일하다.
2. NR3D는 proximity-only 0.648에서 pure ordinal BEV branch가 0.652까지 올린다.
3. ScanRefer route contribution은 spatial-only +4 net이 전부이다.
4. NR3D route contribution은 spatial-only +9 net, BEV +1 net이다.
5. ScanRefer oracle은 0.612로 높아, 현재 router가 놓친 non-E0 recovery가 많다.
6. NR3D available oracle은 기존 route-first source 기준 0.728이고, full branch
   fill-in 후 representation oracle은 0.876이다.
7. NR3D end-to-end rerun은 0.632와 0.656으로 둘 다 E0보다 높다. 평균은 0.644이며,
   non-E0 VLM 재호출에서 약 0.012 수준의 run-to-run variance가 관측된다.
8. Priority LLM router는 dictionary와 77.6% agreement를 보였고, 112/500개 query에서
   다른 route를 선택했다.
9. Dictionary router는 query당 추가 router call이 없지만, LLM router는 query당
   route-classification LLM call 1회가 추가된다.
10. NR3D full input-format fill-in에서 모든 non-E0 input을 전체 query에 일괄 적용하면
    E0보다 낮지만, representation oracle은 0.876까지 오른다. 이는 routing/gating
    문제의 여지를 강하게 보여준다.
11. Mask3D proposal oracle Acc@0.25는 0.672이다. 최종 NR3D router 0.652는 현재
    candidate pool ceiling에 가까운 편이다.
12. Qwen3-VL-8B branch sensitivity run은 0.612로 E0 수준에 머물렀다. Routing gain은
    VLM의 structured spatial prompt 처리 능력에 의존한다.

남은 항목은 실험 실행보다는 논문화 단계이다.

1. failure/recovery case를 실제 canvas/BEV figure로 렌더링한다.
2. evidence audit proxy를 human label audit으로 확장한다.
3. model-change 실험을 full E0 baseline까지 포함하는 별도 프로토콜로 확장한다.
