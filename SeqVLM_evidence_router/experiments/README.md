# Evidence-Aware Routing Experiment Report

이 문서는 현재 레포지토리에서 재현 가능한 최종 실험 결과를 정리한다.
최종 목표는 ScanRefer와 NR3D에 공통으로 적용 가능한 zero-shot 3D visual
grounding pipeline을 만드는 것이다.

Full validation-set 실험 준비 문서는 별도로 관리한다.

- Full-dataset execution plan: `experiments/FULL_DATASET_EVALUATION_PLAN.md`
- Full-dataset checklist: `experiments/FULL_DATASET_CHECKLIST.md`

핵심 아이디어는 쿼리를 새로운 언어학적 taxonomy로 완전히 분류하는 것이
아니라, VLM이 정답 후보를 고르기 위해 필요한 evidence를 보고 입력 표현을
선택하는 것이다.

```text
query + query metadata
        |
        v
evidence-aware router
        |
        +-- E0 RGB canvas
        +-- spatial-only text
        +-- BEV labeled layout
        +-- 3D position text
        |
        v
selected source output recomposition
        |
        v
IoU evaluation
```

현재 실험은 completed VLM source outputs를 재조합하는 방식이다. 즉 final
router의 성능 변화는 proposal pool 차이가 아니라, route policy와 입력 표현
선택에서 온 것으로 해석한다.

## 1. Final Router

최종 라우터는 deterministic evidence-aware router이다. 구현은
`tools/query_type_router.py`에 있다.

라우팅 우선순위는 다음과 같다.

| Priority | 조건 | Route | 의도 |
|---:|---|---|---|
| 1 | `proximity_derived` | spatial-only text | 거리/근접 단서는 candidate-anchor 거리와 offset이 직접적이다. |
| 2 | visual attribute 포함 | E0 RGB canvas | 색, 재질, 형태, 상태, object-on-top 등은 visual evidence가 필요하다. |
| 3 | pure `ordinal` | BEV labeled layout | visual cue가 없는 순수 순서/전역 layout query만 BEV로 보낸다. |
| 4 | pure `geometric` | 3D position text | visual cue가 없는 between/under/inside류만 구조화 좌표를 쓴다. |
| 5 | `viewpoint_guided` | E0 RGB canvas | 현재 BEV는 global frame이라 local viewer-frame을 표현하지 못한다. |
| 6 | 나머지 | E0 RGB canvas | mixed/ambiguous query는 보수적으로 visual baseline을 유지한다. |

중요한 점은 route가 dataset 이름을 사용하지 않는다는 것이다. 같은 query cue는
ScanRefer와 NR3D에서 같은 rule을 탄다.

## 2. Main Table

최종 main table은 `experiments/main_table/table.json` 및
`experiments/summary.json`에 저장되어 있다.

| Dataset | Method | Acc@0.25 | Acc@0.50 | mIoU |
|---|---|---:|---:|---:|
| ScanRefer | E0 baseline | 0.504 | 0.452 | 0.4306 |
| ScanRefer | evidence router | **0.520** | **0.468** | **0.4455** |
| NR3D | E0 baseline | 0.612 | 0.604 | 0.6107 |
| NR3D | evidence router | **0.652** | **0.648** | **0.6514** |

개선폭:

| Dataset | Acc@0.25 gain | Acc@0.50 gain | mIoU gain |
|---|---:|---:|---:|
| ScanRefer | +0.016 | +0.016 | +0.0149 |
| NR3D | +0.040 | +0.044 | +0.0407 |

해석:

- 두 데이터셋 모두 같은 evidence-aware policy로 E0 baseline을 개선한다.
- ScanRefer 개선은 작지만, 같은 policy가 NR3D에서도 상승한다는 점이 중요하다.
- 최종 claim은 dataset-specific calibration이 아니라, query evidence와 VLM
  input representation의 interaction을 활용하는 보수적 routing이다.

## 3. Final Route Distribution

### ScanRefer

| Route | N |
|---|---:|
| E0 RGB canvas | 208 |
| spatial-only text | 37 |
| BEV labeled layout | 1 |
| 3D position text | 4 |

| Route reason | N |
|---|---:|
| visual_attribute_default_e0 | 165 |
| pure_proximity_spatial | 37 |
| default_e0 | 42 |
| pure_geometric_3dpos | 4 |
| pure_ordinal_bev | 1 |
| viewpoint_needs_local_frame_default_e0 | 1 |

### NR3D

| Route | N |
|---|---:|
| E0 RGB canvas | 149 |
| spatial-only text | 95 |
| BEV labeled layout | 6 |
| 3D position text | 0 |

| Route reason | N |
|---|---:|
| pure_proximity_spatial | 95 |
| default_e0 | 76 |
| visual_attribute_default_e0 | 56 |
| viewpoint_needs_local_frame_default_e0 | 17 |
| pure_ordinal_bev | 6 |

해석:

- 실제 성능 향상의 가장 큰 축은 `proximity_derived -> spatial-only text`이다.
- BEV와 3D position은 공격적으로 쓰지 않고, regression을 피하기 위해 매우
  제한적으로 사용한다.
- 이 점 때문에 router는 incremental하게 보일 수 있지만, 의도는 성능 튜닝이
  아니라 visual/mixed evidence를 잘못 버리지 않는 보수적 policy이다.

## 4. Input Format Ablation

ScanRefer 250개에 대해 각 입력 형식을 전체 query에 일괄 적용한 결과는
`experiments/ablation/input_format_overall_scanrefer.json`에 저장되어 있다.

| Input format | Acc@0.25 | Acc@0.50 | mIoU |
|---|---:|---:|---:|
| E0 RGB canvas | **0.504** | **0.452** | **0.4306** |
| BEV labeled layout | 0.444 | 0.412 | 0.3823 |
| spatial-only text | 0.392 | 0.356 | 0.3356 |
| 3D position text | 0.420 | 0.388 | 0.3623 |

해석:

- 어떤 non-E0 input도 전체 query에 일괄 적용하면 E0보다 낮다.
- 따라서 “새 input이 항상 더 좋다”가 아니라, query evidence에 맞춰 제한적으로
  사용해야 한다.
- 이 결과가 routing의 필요성을 뒷받침한다.

## 5. Query-Type/Input Interaction

ScanRefer에서 query type별 입력 형식 성능은
`experiments/ablation/input_format_by_query_type_scanrefer.json`에 저장되어 있다.

주요 결과만 요약하면 다음과 같다.

| Query type | N | E0 | BEV | spatial-only | 3D position | Best |
|---|---:|---:|---:|---:|---:|---|
| `proximity_derived` | 37 | 0.432 | 0.459 | **0.541** | 0.432 | spatial-only |
| `geometric` | 17 | 0.588 | 0.471 | 0.588 | **0.647** | 3D position |
| `ordinal` | 18 | 0.222 | **0.278** | 0.167 | **0.278** | BEV / 3D position |
| `viewpoint_guided` | 4 | 0.250 | **0.750** | 0.500 | **0.750** | BEV / 3D position |
| `none` | 24 | **0.625** | 0.458 | 0.458 | 0.417 | E0 |
| `explicit_direction` | 124 | **0.516** | 0.460 | 0.339 | 0.387 | E0 |

해석:

- Query type에 따라 유리한 input representation이 달라진다.
- `proximity_derived`는 spatial-only가 가장 안정적인 non-E0 route이다.
- `geometric`, `ordinal`, `viewpoint_guided`는 ScanRefer 분석에서는 non-E0
  input에서 이득이 보이지만, NR3D에서 broad routing으로 일반화하면 regression이
  생겼다.
- 따라서 최종 router는 ScanRefer type-level oracle을 그대로 쓰지 않고,
  visual attribute fallback과 pure query 조건을 둔 보수적 policy를 사용한다.

## 6. Dictionary Router vs LLM Router

LLM router ablation은 `openrouter-qwen`으로 수행했다. 결과는
`experiments/ablation/llm_router_priority_openrouter_qwen/summary.json`에
저장되어 있다.

LLM에는 dictionary router와 같은 priority decision tree를 prompt로 제공했다.
즉 단순 free-choice가 아니라 다음 순서를 따르도록 했다.

1. proximity/distance이면 spatial-only
2. visual/mixed evidence이면 E0
3. pure ordinal이면 BEV
4. pure geometric이면 3D position
5. viewpoint/local-frame이면 E0
6. 나머지 E0

| Dataset | Dictionary router | Priority LLM router |
|---|---:|---:|
| ScanRefer Acc@0.25 | **0.520** | 0.492 |
| ScanRefer Acc@0.50 | **0.468** | 0.448 |
| ScanRefer mIoU | **0.4455** | 0.4235 |
| NR3D Acc@0.25 | 0.652 | **0.664** |
| NR3D Acc@0.50 | 0.648 | **0.660** |
| NR3D mIoU | 0.6514 | **0.6636** |

Route difference:

| Item | Count |
|---|---:|
| Total queries | 500 |
| Changed routes vs dictionary | 112 |
| Changed in ScanRefer | 40 |
| Changed in NR3D | 72 |

해석:

- LLM router는 NR3D에서는 더 높은 결과를 냈지만, ScanRefer에서는 성능이 하락했다.
- ScanRefer 하락의 주된 원인은 `E0 -> spatial-only text`로 과도하게 보내는
  경우였다. 이 transition 19개에서 LLM Acc@0.25는 0.316, E0 Acc@0.25는 0.632였다.
- LLM은 `by`, `near`, `left of`, `corner`, `under`, `above` 같은 spatial cue를
  semantic하게 강하게 해석해 mixed visual-spatial query를 non-E0 route로 보내는
  경향이 있다.
- 따라서 LLM router가 나쁘다는 결론보다는, unconstrained 또는 semantic LLM
  routing이 deterministic evidence policy보다 안정적이지 않다는 결론이 적절하다.

자세한 차이 분석은
`experiments/ablation/llm_router_priority_openrouter_qwen/route_diff_analysis.md`에
저장되어 있다.

## 7. Policy Cumulative Ablation

Policy 누적 ablation은
`experiments/ablation/policy_ablation/summary.json`에 저장되어 있다.

| Variant | ScanRefer Acc@0.25 | ScanRefer Net | NR3D Acc@0.25 | NR3D Net |
|---|---:|---:|---:|---:|
| E0 only | 0.504 | 0 | 0.612 | 0 |
| Proximity-only | **0.520** | +4 | 0.648 | +9 |
| Proximity + ordinal | **0.520** | +4 | **0.652** | +10 |
| Proximity + geometric | **0.520** | +4 | 0.648 | +9 |
| Proximity + ordinal + geometric | **0.520** | +4 | **0.652** | +10 |
| Full router | **0.520** | +4 | **0.652** | +10 |

해석:

- ScanRefer에서는 full router의 Acc@0.25 개선이 사실상 proximity-only와 동일하다.
- NR3D에서는 proximity-only가 0.648까지 올리고, pure ordinal BEV branch가
  0.652까지 +0.004를 추가한다.
- 3D position branch는 현재 최종 policy에서는 ScanRefer에서 N=4, NR3D에서 N=0이라
  전체 성능에 거의 기여하지 않는다.
- 따라서 현재 main claim은 “모든 branch가 큰 폭으로 기여한다”가 아니라,
  `proximity_derived -> spatial-only`를 중심으로 한 conservative evidence routing이다.

## 8. Route Contribution

Route별 recovery/regression 분석은
`experiments/ablation/route_contribution/summary.json`에 저장되어 있다.

| Dataset | Route | Routed N | E0 correct | Route correct | Recovery | Regression | Net |
|---|---|---:|---:|---:|---:|---:|---:|
| ScanRefer | spatial-only | 37 | 16 | 20 | 5 | 1 | +4 |
| ScanRefer | BEV | 1 | 0 | 0 | 0 | 0 | 0 |
| ScanRefer | 3D position | 4 | 2 | 2 | 0 | 0 | 0 |
| NR3D | spatial-only | 95 | 59 | 68 | 25 | 16 | +9 |
| NR3D | BEV | 6 | 4 | 5 | 2 | 1 | +1 |

해석:

- ScanRefer 향상은 전부 spatial-only route의 +4 net에서 나온다.
- NR3D 향상은 spatial-only +9 net과 BEV +1 net의 합이다.
- BEV/3D branch는 현재 final router에서 보조적이며, 잘못 사용하면 regression이
  커질 수 있으므로 conservative routing이 필요하다.

## 9. Statistical Tests

Paired statistical tests는 `experiments/statistics/paired_tests.json`에 저장되어
있다. 같은 250개 query에서 E0와 final router를 비교했다.

| Dataset | Metric | Gain | 95% CI | p-value |
|---|---|---:|---:|---:|
| ScanRefer | Acc@0.25 | +0.016 | [0.000, 0.036] | 0.2188 |
| ScanRefer | Acc@0.50 | +0.016 | [0.004, 0.032] | 0.1250 |
| ScanRefer | mIoU | +0.0150 | [0.0020, 0.0309] | 0.0414 |
| NR3D | Acc@0.25 | +0.040 | [-0.012, 0.092] | 0.1742 |
| NR3D | Acc@0.50 | +0.044 | [-0.008, 0.096] | 0.1352 |
| NR3D | mIoU | +0.0407 | [-0.0100, 0.0905] | 0.1332 |

p-value는 Acc에 대해서 exact McNemar/binomial test, mIoU에 대해서 paired
permutation test를 사용했다. CI는 paired bootstrap 5000 samples이다.

해석:

- ScanRefer Acc gain은 net +4로 작아 Acc 기준 유의성은 약하다.
- ScanRefer mIoU gain은 bootstrap CI와 permutation test에서 더 안정적으로 보인다.
- NR3D는 평균 gain은 크지만 recovery/regression이 모두 많아 250-query setting에서
  CI가 넓다.
- 따라서 main text에서는 유의성만 과장하기보다, route-level contribution과
  oracle 분석을 함께 제시하는 것이 안전하다.

## 10. Representation Oracle

Representation oracle 분석은
`experiments/ablation/representation_oracle/summary.json`에 저장되어 있다.

| Dataset | E0 | Final router | Available oracle |
|---|---:|---:|---:|
| ScanRefer | 0.504 | 0.520 | 0.612 |
| NR3D available-source | 0.612 | 0.652 | 0.728 |

| Dataset | E0 fail, oracle success | Router recovered | Missed oracle recoveries |
|---|---:|---:|---:|
| ScanRefer | 27 | 5 | 22 |
| NR3D available-source | 29 | 27 | 2 |

해석:

- ScanRefer에는 아직 회수하지 못한 non-E0 성공 사례가 많다. 즉 future work로
  learnable gate나 better evidence extractor의 여지가 크다.
- NR3D에서는 available non-E0 source 기준으로 router가 oracle recoverable cases의
  대부분을 회수한다.
- ScanRefer oracle은 모든 final input format source가 있으므로 비교적 온전하다.
  NR3D oracle은 현재 available route-first source 기준이므로 full oracle은 아니다.

## 11. Router Component Ablation

Router component ablation도
`experiments/ablation/router_components/summary.json`에 저장되어 있다.

| Variant | ScanRefer Acc@0.25 | ScanRefer Net | NR3D Acc@0.25 | NR3D Net |
|---|---:|---:|---:|---:|
| Full router | 0.520 | +4 | 0.652 | +10 |
| Without visual fallback | 0.512 | +2 | 0.652 | +10 |
| Without purity constraint | 0.528 | +6 | 0.628 | +4 |
| Without viewpoint fallback | 0.528 | +6 | 0.636 | +6 |
| Without priority ordering | 0.500 | -1 | 0.652 | +10 |

해석:

- Visual fallback은 ScanRefer regression을 줄인다.
- Purity/viewpoint fallback을 제거하면 ScanRefer에서는 우연히 올라가지만 NR3D에서
  크게 하락한다. 이는 broad ordinal/viewpoint routing이 dataset-general하지 않음을
  보여준다.
- Priority ordering을 제거하면 ScanRefer가 E0보다 낮아진다. proximity를 visual
  fallback보다 먼저 처리하는 현재 ordering이 중요하다.

## 12. End-to-End NR3D Re-runs

Final evidence router를 원본 SeqVLM NR3D route-first runner에 patch하여
`openrouter-qwen`으로 다시 실행했다. E0 route는 공식 E0 결과를 재사용했고,
spatial-only/BEV route는 실제 VLM API를 다시 호출했다.

결과는 다음 위치에 저장되어 있다.

- `experiments/end_to_end_nr3d_final_router_openrouter_qwen/`
- `experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat2/`
- `experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat_summary.json`

| Method | Acc@0.25 | Acc@0.50 | mIoU |
|---|---:|---:|---:|
| E0 baseline | 0.612 | 0.604 | 0.6107 |
| Final router recomposition | 0.652 | 0.648 | 0.6514 |
| Final router end-to-end rerun 1 | 0.632 | 0.628 | 0.6316 |
| Final router end-to-end rerun 2 | 0.656 | 0.652 | 0.6556 |
| End-to-end rerun mean | 0.644 | 0.640 | 0.6436 |

Run-to-run 표준편차는 Acc@0.25/Acc@0.50/mIoU 모두 약 0.012이다.

| Route | N | Recomposition Acc@0.25 | Rerun 1 | Rerun 2 | E0 Acc@0.25 |
|---|---:|---:|---:|---:|---:|
| E0 | 149 | 0.604 | 0.604 | 0.604 | 0.604 |
| spatial-only | 95 | 0.716 | 0.674 | 0.737 | 0.621 |
| BEV | 6 | 0.833 | 0.667 | 0.667 | 0.667 |

해석:

- 두 end-to-end rerun 모두 E0 baseline보다 높다.
- Recomposition과 rerun의 차이는 non-E0 VLM 재호출에서 발생한다. E0 route는
  재사용이므로 항상 동일하다.
- Rerun 1은 recomposition보다 낮고, rerun 2는 recomposition보다 약간 높다.
- 따라서 VLM API 응답/decoding variance가 존재하며, 최종 논문에서는 temperature 0,
  fixed prompt, repeated run 또는 cached outputs를 명시해야 한다.

## 13. NR3D Full Input-Format Fill-in

NR3D에 대해 E0, spatial-only text, 3D position text, BEV labeled layout을 모두
250개 query 전체에 실행했다. 결과는
`experiments/ablation/nr3d_missing_branches/openrouter_qwen/summary.json`에
저장되어 있다.

| Input format | Acc@0.25 | Acc@0.50 | mIoU |
|---|---:|---:|---:|
| E0 RGB canvas | **0.612** | **0.604** | **0.6107** |
| spatial-only text | 0.580 | 0.580 | 0.5838 |
| 3D position text | 0.560 | 0.560 | 0.5638 |
| BEV labeled layout | 0.584 | 0.584 | 0.5871 |
| Full representation oracle | **0.876** | **0.876** | **0.8774** |

Oracle에서 best source로 선택된 횟수:

| Best source | Count |
|---|---:|
| E0 | 138 |
| spatial-only text | 90 |
| BEV labeled layout | 17 |
| 3D position text | 5 |

해석:

- 각 non-E0 input을 전체 query에 일괄 적용하면 E0보다 낮다.
- 그러나 representation oracle은 0.876까지 올라간다.
- 즉 핵심은 input format 자체가 아니라, 어떤 query에 어떤 representation을
  줄지 결정하는 routing/gating 문제이다.

## 14. GT Proposal / Candidate Oracle Diagnostic

Mask3D candidate pool의 upper bound를 확인했다. 결과는
`experiments/gt_proposal/summary.json`에 저장되어 있다.

| Diagnostic | Value |
|---|---:|
| NR3D Mask3D proposal oracle Acc@0.25 | 0.672 |
| NR3D Mask3D proposal oracle Acc@0.50 | 0.568 |
| Best IoU mean | 0.5576 |
| Avg candidates | 4.036 |
| Avg candidates with canvas | 3.452 |
| Zero-candidate cases | 24 |
| Zero-canvas-candidate cases | 54 |

해석:

- 최종 NR3D router recomposition Acc@0.25 0.652는 Mask3D proposal oracle
  0.672에 가까운 수준이다.
- 현재 protocol에서는 proposal/candidate coverage가 중요한 ceiling으로 작용한다.

## 15. VLM Model Change

유효한 OpenRouter 대체 모델 `qwen/qwen3-vl-8b-instruct`로 non-E0 branch를 다시
호출했다. 결과는 `experiments/ablation/vlm_model_change/summary.json`에 저장했다.

| Setting | Acc@0.25 | Acc@0.50 | mIoU |
|---|---:|---:|---:|
| E0 baseline, openrouter-qwen source | 0.612 | 0.604 | 0.6107 |
| Final router, openrouter-qwen recomposition | **0.652** | **0.648** | **0.6514** |
| Final router, Qwen3-VL-8B non-E0 branches | 0.612 | 0.608 | 0.6129 |

주의:

- 이 실험은 full alternate-VLM baseline이 아니다.
- E0 route 149개는 공식 openrouter-qwen E0 source를 재사용하고, non-E0 branch만
  Qwen3-VL-8B로 바꾼 branch sensitivity test이다.
- 결과적으로 작은 모델에서는 spatial/BEV branch의 이득이 사라져 E0 baseline
  수준으로 돌아간다. Routing gain은 VLM의 structured spatial prompt 처리 능력에
  영향을 받는다.

## 16. Runtime / Cost Proxy

Runtime proxy는 `experiments/runtime/runtime_proxy.json`에 저장되어 있다. 현재
산출물 기반 proxy는 wall-clock latency가 아니라 call-count accounting이다.

| Method | Router cost | Selection VLM calls/query | Extra router calls/query |
|---|---|---:|---:|
| E0 baseline | none | 1 | 0 |
| Dictionary evidence router | local deterministic rules | 1 | 0 |
| Priority LLM router | OpenRouter Qwen classifier | 1 | 1 |

해석:

- Dictionary router는 E0와 동일하게 query당 selection VLM call 1회만 필요하다.
- LLM router는 route 결정을 위해 query당 LLM call 1회가 추가된다.
- LLM router는 NR3D에서는 더 높지만 ScanRefer에서 하락하므로, 현재 최종 pipeline은
  cost와 stability 측면에서 dictionary router가 더 적합하다.

## 17. Failure / Recovery Cases

Failure/recovery case 목록은 `experiments/failure_visualization/`에 저장되어 있다.

| Output | Path |
|---|---|
| Case list | `experiments/failure_visualization/cases.jsonl` |
| Markdown report | `experiments/failure_visualization/failure_cases.md` |
| Summary | `experiments/failure_visualization/summary.json` |

현재 standalone bundle에는 전체 E0 RGB canvas 원본 asset이 포함되어 있지 않으므로,
이 산출물은 query, route, E0 IoU, routed IoU, transition 중심의 lightweight
failure report이다. Full visual figure는 원본 rendered canvas asset을 연결하면
같은 case id로 확장할 수 있다.

## 18. Evidence Audit Proxy

Evidence audit proxy는 `experiments/evidence_audit/`에 저장되어 있다.

| Output | Path |
|---|---|
| Automatic evidence labels | `experiments/evidence_audit/auto_evidence_labels.csv` |
| Summary | `experiments/evidence_audit/summary.json` |

이 audit은 human manual label이 아니라 최종 deterministic evidence rule을 전체
500개 query에 적용한 자동 점검이다. Human audit이 필요하면 이 CSV를 template으로
사용해 appearance/proximity/ordinal/geometric/viewpoint/mixed label을 추가하면 된다.

## 19. LLM Transition Matrix

Priority LLM router와 dictionary router의 transition matrix는
`experiments/ablation/llm_router_priority_openrouter_qwen/transition_matrix.json`에
저장되어 있다.

| Item | Count |
|---|---:|
| Total queries | 500 |
| Same route | 388 |
| Changed route | 112 |
| Agreement rate | 0.776 |

가장 중요한 transition은 dictionary `E0`를 LLM이 `spatial-only`로 바꾼 33건이다.
이 subset에서 LLM Acc@0.25는 0.424, E0 Acc@0.25는 0.606이었다. 즉 LLM은 mixed
visual-spatial query를 과도하게 spatial route로 보내는 경향이 있다.

## 20. Full NR3D Validation Run

NR3D full validation split 7,457개에 대해 corrected v2 object-ID materialized
input pipeline으로 final evidence router를 실행했다. 이 실험은 250-query locked
main table과 별개이며, full-validation main table을 만들기 위한 첫 번째
route-first run이다.

결과는 다음 위치에 저장했다.

- Summary: `experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/summary.json`
- Error analysis: `experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/error_analysis.json`
- Full generated output: `/data/knuvi/bosung/evidence_router_full_runs_v2/nr3d/final_router/openrouter-qwen/`

| Dataset | Run | N | Acc@0.25 | Acc@0.50 | mIoU |
|---|---|---:|---:|---:|---:|
| NR3D val | final evidence router, OpenRouter Qwen | 7,457 | 0.5958 | 0.5946 | 0.5985 |

Route별 결과:

| Route | N | Acc@0.25 | Acc@0.50 | mIoU |
|---|---:|---:|---:|---:|
| E0 RGB canvas | 3,982 | 0.5497 | 0.5482 | 0.5532 |
| spatial-only text | 3,015 | 0.6594 | 0.6584 | 0.6609 |
| 3D position text | 350 | 0.6114 | 0.6114 | 0.6137 |
| BEV labeled layout | 110 | 0.4727 | 0.4727 | 0.4796 |

Run diagnostics:

| Item | Value |
|---|---:|
| Processed | 7,457 / 7,457 |
| OK rows | 7,383 |
| Error rows | 74 |
| API calls | 8,841 |
| Prompt tokens | 54,323,043 |
| Completion tokens | 578,323 |
| Runtime | 19,755.24 sec |

오류 74건은 대부분 OpenRouter upstream rate-limit/timeout 또는 E0 tournament
postprocessing issue였다. 전체 score는 오류 row까지 포함한 conservative metric이고,
OK row만 계산하면 Acc@0.25 0.5991, Acc@0.50 0.5979, mIoU 0.6018이다.

주의할 점은 이 결과를 250-query NR3D E0 baseline과 직접 비교하면 안 된다는
것이다. Full-validation main table을 위해서는 동일한 7,457개 query에 대한
full E0-only baseline이 필요하다.

## 21. Current Interpretation

현재까지의 가장 안전한 주장은 다음과 같다.

1. Query evidence와 input representation 사이에는 interaction이 있다.
   전체 query에 하나의 input을 일괄 적용하면 E0보다 낮지만, 특정 evidence type에는
   non-E0 input이 더 유리하다.

2. 최종 evidence-aware router는 dataset-specific calibration 없이 ScanRefer와
   NR3D 모두에서 E0 baseline을 개선한다.

3. 가장 신뢰할 수 있는 route는 `proximity_derived -> spatial-only text`이다.
   두 데이터셋에서 주요 gain을 만든다.

4. BEV와 3D position은 potential은 있지만 broad routing에는 위험하다.
   Visual/mixed query나 viewpoint-local query를 잘못 보내면 regression이 생긴다.

5. LLM router는 유연하지만 더 안정적이지는 않다.
   NR3D에서는 개선되지만 ScanRefer에서는 mixed query over-routing 때문에 하락한다.
   따라서 최종 pipeline은 deterministic evidence-aware router를 채택한다.

## 22. Output Files

| Output | Path |
|---|---|
| Main table | `experiments/main_table/table.json` |
| Final summary | `experiments/summary.json` |
| Final ScanRefer routed output | `experiments/main_table/final_router_outputs/scanrefer_universal_evidence_routed_results.jsonl` |
| Final NR3D routed output | `experiments/main_table/final_router_outputs/nr3d_universal_evidence_routed_results.jsonl` |
| ScanRefer input-format overall ablation | `experiments/ablation/input_format_overall_scanrefer.json` |
| ScanRefer query-type/input ablation | `experiments/ablation/input_format_by_query_type_scanrefer.json` |
| Policy cumulative ablation | `experiments/ablation/policy_ablation/summary.json` |
| Router component ablation | `experiments/ablation/router_components/summary.json` |
| Route contribution | `experiments/ablation/route_contribution/summary.json` |
| Paired statistical tests | `experiments/statistics/paired_tests.json` |
| Representation oracle | `experiments/ablation/representation_oracle/summary.json` |
| NR3D end-to-end final-router rerun | `experiments/end_to_end_nr3d_final_router_openrouter_qwen/summary.json` |
| NR3D end-to-end repeat summary | `experiments/end_to_end_nr3d_final_router_openrouter_qwen_repeat_summary.json` |
| NR3D full branch fill-in | `experiments/ablation/nr3d_missing_branches/openrouter_qwen/summary.json` |
| GT proposal diagnostic | `experiments/gt_proposal/summary.json` |
| VLM model-change branch sensitivity | `experiments/ablation/vlm_model_change/summary.json` |
| NR3D end-to-end consistency | `experiments/end_to_end_nr3d_final_router_openrouter_qwen/consistency_summary.json` |
| Priority LLM router summary | `experiments/ablation/llm_router_priority_openrouter_qwen/summary.json` |
| Priority LLM transition matrix | `experiments/ablation/llm_router_priority_openrouter_qwen/transition_matrix.json` |
| LLM-vs-dictionary route diff analysis | `experiments/ablation/llm_router_priority_openrouter_qwen/route_diff_analysis.md` |
| Runtime/cost proxy | `experiments/runtime/runtime_proxy.json` |
| Failure/recovery cases | `experiments/failure_visualization/cases.jsonl` |
| Evidence audit proxy | `experiments/evidence_audit/summary.json` |
| Full NR3D final-router summary | `experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/summary.json` |
| Full NR3D final-router error analysis | `experiments/full_dataset_results/nr3d_final_router_openrouter_qwen_v2/error_analysis.json` |

## 23. Paper Figures

논문 본문/보충자료에 바로 사용할 수 있는 figure는
`experiments/figures/`에 저장했다. 모든 figure는 PNG와 PDF를 함께 제공한다.

재생성 명령:

```bash
python tools/create_paper_figures.py
```

| Figure | Path stem | 내용 |
|---|---|---|
| Fig. 1 | `experiments/figures/fig1_pipeline_overview` | Evidence-aware routing pipeline과 route priority |
| Fig. 2 | `experiments/figures/fig2_main_results` | ScanRefer/NR3D main result 비교 |
| Fig. 3 | `experiments/figures/fig3_input_format_ablation_scanrefer` | ScanRefer 전체 query에 입력 형식을 일괄 적용한 ablation |
| Fig. 4 | `experiments/figures/fig4_query_type_input_heatmap_scanrefer` | Query type별 입력 형식 성능 heatmap |
| Fig. 5 | `experiments/figures/fig5_route_distribution_contribution` | 최종 route 분포와 recovery/regression contribution |
| Fig. 6 | `experiments/figures/fig6_policy_component_ablation` | Policy cumulative ablation과 router component ablation |
| Fig. 7 | `experiments/figures/fig7_llm_router_comparison` | Dictionary router와 priority LLM router 비교 |
| Fig. 8 | `experiments/figures/fig8_oracle_and_rerun` | Representation oracle gap과 NR3D rerun variance |
| Fig. 9 | `experiments/figures/fig9_failure_case_summary` | 대표 recovery/regression/fallback case 요약 |
