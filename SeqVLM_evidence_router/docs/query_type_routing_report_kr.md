# Dictionary Evidence-Aware Routing 기반 VLM Grounding 보고서

## 1. 목표

최종 목표는 dataset-specific calibration이 아니라, ScanRefer와 NR3D에 공통으로
적용 가능한 query-aware VLM grounding pipeline을 만드는 것이다.

현재 보고용 최종 기준은 LLM router가 아니라 deterministic dictionary/rule
기반 evidence-aware router이다. 즉 쿼리의 surface cue와 기존 query type을
공통 규칙으로 해석해서 입력 형식을 선택한다.

```text
Query
  -> query type + dictionary/rule evidence 판단
  -> 필요한 입력 형식 선택
  -> 동일 candidate pool에서 VLM selection
  -> IoU 평가
```

이 방식은 데이터셋 이름을 route 결정에 사용하지 않는다. 같은 문장 단서라면
ScanRefer와 NR3D에서 같은 rule이 적용된다.

LLM 기반 route-type classifier는 후속 비교 실험으로 분리한다. 최종 성능표와
보고 기준은 아래 dictionary router 결과만 사용한다.

## 2. 왜 기존 broad routing을 수정했는가

기존 policy는 단일 query type을 바로 route로 매핑했다.

| Query type | 기존 route |
|---|---|
| `proximity_derived` | spatial-only text |
| `ordinal` | BEV |
| `viewpoint_guided` | BEV |
| `geometric` | 3D position text |
| others | E0 |

이 방식은 ScanRefer에서는 Acc@0.25 0.504에서 0.536으로 올랐지만, NR3D에서는
BEV route가 많은 regression을 만들었다. 원인은 `ordinal` 또는
`viewpoint_guided`라는 단일 label 안에 visual attribute, object state,
local viewpoint frame이 섞여 있었기 때문이다.

예:

```text
"the trash can in the corner with the blue bag"
  = ordinal/room cue + visual attribute

"looking straight at the orange towel the desk on the right"
  = viewpoint cue + color anchor + local frame

"the round table with the most objects on top of it"
  = superlative cue + object-on-top visual/semantic evidence
```

따라서 보편적인 routing은 query type보다 evidence requirement를 우선해야 한다.

## 3. Dictionary Evidence-Aware Policy v2

현재 구현은 [tools/query_type_router.py](/home/knuvi/bosung/SeqVLM/tools/query_type_router.py)에
있다. 정책은 다음 순서로 적용된다.

| Priority | Dictionary/rule 조건 | Route | 이유 |
|---:|---|---|---|
| 1 | `proximity_derived` | spatial-only text | 거리/근접 단서는 좌표와 anchor distance가 직접적이다. |
| 2 | visual attribute 포함 | E0 | 색, 재질, 상태, 물체가 위에 있음 등은 RGB evidence가 필요하다. |
| 3 | pure `ordinal` | BEV | visual attribute가 없는 순수 순서/위치 비교만 top-down layout에 보낸다. |
| 4 | pure `geometric` | 3D position text | visual attribute가 없는 between/under/inside류만 구조화 좌표를 쓴다. |
| 5 | `viewpoint_guided` | E0 | local-frame spatial 입력이 아직 없으므로 BEV로 보내지 않는다. |
| 6 | 나머지 | E0 | mixed/ambiguous query는 visual baseline이 가장 안전하다. |

이를 query type별로 풀면 다음과 같다.

| Query type | Dictionary/rule 조건 | 최종 route | 이유 |
|---|---|---|---|
| `proximity_derived` | query type이 `proximity_derived`이면 visual attribute 여부보다 먼저 적용 | spatial-only text | 거리/근접 단서는 candidate-anchor 거리와 offset이 직접적이다. |
| `ordinal` | visual attribute가 없고 순수 순서/위치 비교일 때만 `pure ordinal`로 인정 | BEV labeled layout | `middle`, `leftmost`, `second`, `corner` 같은 전역/국소 순서는 top-down layout이 유리하다. |
| `ordinal` + visual attribute | 색, 재질, 상태, object-on-top 등 visual cue가 포함됨 | E0 RGB canvas | 순서 단서만으로는 부족하고 RGB evidence가 필요하다. |
| `geometric` | visual attribute가 없고 `between`, `under`, `inside`, `above/below` 등이 핵심일 때만 `pure geometric`으로 인정 | 3D position text | 구조화된 3D 좌표/크기/anchor 위치가 기하 관계 판단에 직접적이다. |
| `geometric` + visual attribute | `tan`, `round`, material/state 등 visual cue가 포함됨 | E0 RGB canvas | geometry만으로 target을 고르면 attribute를 놓칠 수 있다. |
| `viewpoint_guided` | viewpoint/local-frame cue 포함 | E0 RGB canvas | 현재 BEV는 global XY frame이라 viewer-centered left/right를 안정적으로 표현하지 못한다. |
| `explicit_direction` | 방향 관계가 있으나 broad direction type은 기본적으로 mixed evidence로 처리 | E0 RGB canvas | left/right/front/back 단서에 visual attribute가 자주 섞이고, E0가 가장 안정적이었다. |
| `none` | 명시적 spatial relation이 약하거나 일반 object description | E0 RGB canvas | 색, 형태, category, local visual context가 중요하다. |
| `object_orientation` | facing/oriented/back side 등 객체 방향 단서 | E0 RGB canvas | 객체의 시각적 방향/상태는 RGB view에서 확인하는 것이 안전하다. |
| `room_side` | wall/corner/room side 단서 | E0 RGB canvas | 현재 policy에서는 room geometry 전용 입력이 없고, E0가 더 안정적이었다. |
| `opposite_derived` | opposite/across 관계 | E0 RGB canvas | N이 작고 E0 성능이 안정적이어서 보수적으로 유지한다. |
| `uncategorized` | parser가 명확한 type을 주지 못함 | E0 RGB canvas | ambiguous query는 잘못된 non-E0 routing보다 visual baseline이 안전하다. |

중요한 점은 `proximity_derived`를 visual attribute rule보다 먼저 처리한다는
것이다. visual-first로 두면 ScanRefer proximity query 대부분이 E0로 돌아가
성능이 떨어진다. 거리/근접 유형은 색 단서가 조금 섞여 있어도 spatial evidence가
핵심인 경우가 많았다.

## 4. 입력 형식

### E0: SeqVLM RGB Canvas

E0는 기본 route이다. 후보별 RGB canvas를 VLM에 제공하고 기존 SeqVLM 방식으로
target instance를 선택한다.

E0가 필요한 경우:

- color/material/shape 같은 appearance clue
- object state: open/closed, seat up, bag/note/towel 등
- object-on-top, clutter, visible object count
- local viewpoint frame을 아직 계산할 수 없는 `viewpoint_guided`
- mixed or ambiguous query

### Spatial-Only Text

이미지를 사용하지 않고 후보와 anchor의 3D 위치, 크기, 상대 offset, 거리 정보를
텍스트로 제공한다.

예:

```text
Query: "it is the chair by itself not near the desk"

Use only relative spatial relationships, distances, ordering, and object categories.
Ignore color, material, texture, and other visual appearance words.

Anchors:
  id12 desk: center=(-0.02,-0.20,+0.31) size=(2.03x2.84x0.61)

Candidates:
  A. id0 chair: center=(+0.90,-2.27,+0.32) size=(0.67x0.51x0.44)
     anchor_delta=(+0.92,-2.07,+0.02) xy_dist=2.26m
  B. id1 chair: center=(+1.31,-0.48,+0.57) size=(0.69x0.74x0.59)
     anchor_delta=(+1.33,-0.27,+0.26) xy_dist=1.36m

Return only JSON: {"letter": "<one of A, B, ...>"}
```

주 route는 `proximity_derived`이다.

### BEV Labeled Layout

BEV는 top-down layout image에 후보를 A/B/C/D로 표시한다. 현재 dictionary
policy에서는 visual attribute가 없는 pure ordinal query에만 제한적으로 사용한다.

예:

```text
Query: "the cup in the middle"

3D Spatial Information:
  A (solid box) = candidate: cup center=(-0.04,0.91,0.87) size=(...)
  B (solid box) = candidate: cup center=(+0.17,0.88,0.87) size=(...)
  C (solid box) = candidate: cup center=(+0.29,0.95,0.89) size=(...)

The VLM also receives a BEV image with A/B/C labels.
```

NR3D에서는 `viewpoint_guided -> BEV`가 regression을 만들었다. 이유는
NR3D의 view-dependent 표현이 global XY frame이 아니라 viewer-centered local
frame을 요구하기 때문이다. 따라서 local-frame spatial input이 구현되기 전까지
viewpoint query는 E0로 보낸다.

### 3D Position Text

후보의 center/size/height와 anchor 정보를 구조화해서 제공한다. 현재 dictionary
policy에서는 visual attribute가 없는 pure geometric query에만 사용한다.

## 5. 입력 형식 시각화

보고용 시각화는
`experiments/full_ablation/outputs/input_format_visualizations/`에 저장했다.

### 입력 형식 overview

![input_format_overview](input_format_visualizations/input_format_overview.png)

### 실제 route 예시: Query type, VLM input, VLM output

한 장에 모든 route를 넣은 합본은 축소 시 글자가 깨질 수 있어, 보고용으로는
아래 고해상도 route별 상세 그림을 사용한다. 각 그림은 왼쪽부터 query type과
예시 query, 실제 VLM input, VLM output을 같은 행에 배치했다.

#### E0 RGB canvas route

![detail_e0_rgb_canvas](input_format_visualizations/detail_e0_rgb_canvas.png)

#### Spatial-only text route

![detail_spatial_only_text](input_format_visualizations/detail_spatial_only_text.png)

#### BEV labeled layout route

![detail_bev_labeled_layout](input_format_visualizations/detail_bev_labeled_layout.png)

#### 3D position text route

![detail_3d_position_text](input_format_visualizations/detail_3d_position_text.png)

### Dictionary routing flow

![dictionary_routing_flow](input_format_visualizations/dictionary_routing_flow.png)

### Route distribution

![route_distribution](input_format_visualizations/route_distribution.png)

그림 생성 스크립트:
`tools/create_input_format_visualizations.py`

## 6. 실험 방식

이번 결과는 새 VLM 호출이 아니라 completed outputs를 재조합한 policy
ablation이다.

- ScanRefer E0:
  `experiments/full_ablation/outputs/full_E0_baseline_qwen72b.jsonl`
- ScanRefer alternate inputs:
  `results/input_format_ablation_paper_faithful/*/results.jsonl`
- NR3D E0:
  `experiments/full_ablation/outputs/official_e0_nr3d_openrouter_qwen_250.jsonl`
- NR3D routed inputs:
  `experiments/full_ablation/outputs/nr3d_query_type_routed_vlm_bev_exact_scanrefer_format/`
- Dictionary universal evaluation output:
  `experiments/full_ablation/outputs/universal_evidence_router/`

Candidate pool은 baseline과 동일하게 유지한다. 따라서 변화는 proposal recall이
아니라 VLM 입력 표현과 routing policy에서 온 것으로 해석한다.

## 7. 전체 결과

| Dataset | Method | Acc@0.25 | Acc@0.50 | mIoU |
|---|---|---:|---:|---:|
| ScanRefer | E0 baseline | 0.504 | 0.452 | 0.4306 |
| ScanRefer | Dictionary evidence router v2 | **0.520** | **0.468** | **0.4455** |
| NR3D | E0 baseline | 0.612 | 0.604 | 0.6107 |
| NR3D | Dictionary evidence router v2 | **0.652** | **0.648** | **0.6514** |

두 데이터셋 모두 baseline보다 상승한다.

- ScanRefer: +1.6pp Acc@0.25
- NR3D: +4.0pp Acc@0.25

## 8. ScanRefer 결과

### Route 분포

| Route | N |
|---|---:|
| E0 | 208 |
| spatial-only text | 37 |
| BEV | 1 |
| 3D position text | 4 |

### Route reason 분포

| Reason | N |
|---|---:|
| visual_attribute_default_e0 | 165 |
| pure_proximity_spatial | 37 |
| default_e0 | 42 |
| pure_geometric_3dpos | 4 |
| pure_ordinal_bev | 1 |
| viewpoint_needs_local_frame_default_e0 | 1 |

### 최종 라우팅 적용 결과

아래 표는 입력 형식 ablation이 아니라, 최종 dictionary router가 실제로 선택한
입력 형식 기준의 결과이다. `E0 Acc@0.25`는 같은 subset을 모두 E0로 처리했을
때의 성능이고, `Recovery/Regression`은 E0 대비 Acc@0.25 기준 변화 개수이다.

| Rule reason | 선택 입력 | N | Routed Acc@0.25 | Routed Acc@0.50 | mIoU | E0 Acc@0.25 | Recovery | Regression | Net |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visual_attribute_default_e0 | E0 RGB canvas | 165 | 0.527 | 0.473 | 0.4484 | 0.527 | 0 | 0 | 0 |
| default_e0 | E0 RGB canvas | 42 | 0.500 | 0.452 | 0.4478 | 0.500 | 0 | 0 | 0 |
| pure_proximity_spatial | spatial-only text | 37 | **0.541** | **0.486** | **0.4482** | 0.432 | 5 | 1 | +4 |
| pure_geometric_3dpos | 3D position text | 4 | 0.500 | 0.500 | 0.5032 | 0.500 | 0 | 0 | 0 |
| pure_ordinal_bev | BEV labeled layout | 1 | 0.000 | 0.000 | 0.0000 | 0.000 | 0 | 0 | 0 |
| viewpoint_needs_local_frame_default_e0 | E0 RGB canvas | 1 | 0.000 | 0.000 | 0.0000 | 0.000 | 0 | 0 | 0 |

ScanRefer에서 실제 성능 향상은 `pure_proximity_spatial`에서 나온다. 나머지
route는 대부분 E0 유지이거나 N이 작아 전체 성능 변화에 거의 영향을 주지 않는다.

### Query type별 결과

| Query type | N | Main route | Acc@0.25 |
|---|---:|---|---:|
| `explicit_direction` | 124 | E0 | 0.5161 |
| `proximity_derived` | 37 | spatial-only | **0.5405** |
| `none` | 24 | E0 | 0.6250 |
| `ordinal` | 18 | mostly E0 | 0.2222 |
| `geometric` | 17 | E0 / 3Dpos | 0.5882 |
| `object_orientation` | 8 | E0 | 0.7500 |
| `room_side` | 7 | E0 | 0.7143 |
| `opposite_derived` | 5 | E0 | 0.8000 |
| `viewpoint_guided` | 4 | E0 | 0.2500 |

ScanRefer에서 기존 broad policy는 0.536까지 올라갔다. 하지만 그 policy는
`ordinal/viewpoint/geometric`을 더 공격적으로 non-E0 route로 보내는 방식이었고,
NR3D에서는 BEV regression을 만들었다. Dictionary evidence policy는 더 보수적이지만,
두 데이터셋에 같은 rule을 적용한다는 점에서 더 객관적인 기준이다.

## 9. NR3D 결과

### Route 분포

| Route | N |
|---|---:|
| spatial-only text | 95 |
| E0 | 149 |
| BEV | 6 |

### Route reason 분포

| Reason | N |
|---|---:|
| pure_proximity_spatial | 95 |
| default_e0 | 76 |
| visual_attribute_default_e0 | 56 |
| viewpoint_needs_local_frame_default_e0 | 17 |
| pure_ordinal_bev | 6 |

### 최종 라우팅 적용 결과

| Rule reason | 선택 입력 | N | Routed Acc@0.25 | Routed Acc@0.50 | mIoU | E0 Acc@0.25 | Recovery | Regression | Net |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_proximity_spatial | spatial-only text | 95 | **0.716** | **0.716** | **0.7175** | 0.621 | 25 | 16 | +9 |
| default_e0 | E0 RGB canvas | 76 | 0.605 | 0.592 | 0.5966 | 0.605 | 0 | 0 | 0 |
| visual_attribute_default_e0 | E0 RGB canvas | 56 | 0.625 | 0.625 | 0.6309 | 0.625 | 0 | 0 | 0 |
| viewpoint_needs_local_frame_default_e0 | E0 RGB canvas | 17 | 0.529 | 0.529 | 0.5313 | 0.529 | 0 | 0 | 0 |
| pure_ordinal_bev | BEV labeled layout | 6 | **0.833** | **0.833** | **0.8333** | 0.667 | 2 | 1 | +1 |

NR3D에서도 주요 개선은 `pure_proximity_spatial -> spatial-only text`에서 나온다.
`pure_ordinal_bev`는 6개에만 적용되어 작은 순이득을 보였고,
`viewpoint_guided`는 local-frame 입력이 없기 때문에 E0로 유지된다.

### Query type별 결과

| Query type | N | Main route | Acc@0.25 |
|---|---:|---|---:|
| `proximity_derived` | 95 | spatial-only | **0.7158** |
| `explicit_direction` | 89 | E0 | 0.5281 |
| `viewpoint_guided` | 18 | E0 | 0.5556 |
| `ordinal` | 16 | E0 / BEV | 0.7500 |
| `none` | 9 | E0 | 0.7778 |
| `object_orientation` | 9 | E0 | 0.8889 |
| `room_side` | 6 | E0 | 0.8333 |
| `geometric` | 4 | E0 | 1.0000 |
| `opposite_derived` | 4 | E0 | 0.5000 |

NR3D의 개선은 대부분 `proximity_derived -> spatial-only`에서 나온다. BEV는
pure ordinal 6개에만 제한적으로 사용되어 net +1 정도의 작은 기여를 한다.
`viewpoint_guided`는 전부 E0로 유지한다.

## 10. BEV + position prompt 공통 기준 검토

BEV + position prompt가 두 데이터셋 모두에서 안정적으로 이득을 주는 routing
기준이 있는지 확인했다. 여기서 BEV + position은 `bev_raw_labeled`처럼
top-down image와 candidate 좌표 텍스트를 함께 제공하는 입력을 의미한다.

### 후보 기준별 결과

| Routing 후보 기준 | ScanRefer BEV+pos vs E0 | NR3D BEV+pos vs E0 | 판단 |
|---|---:|---:|---|
| `viewpoint_guided` / facing류 | ScanRefer는 일부 이득 | NR3D는 net negative 또는 neutral | 공통 기준으로 부적합 |
| `ordinal` 전체 | ScanRefer net negative | NR3D net negative in broad BEV route | 공통 기준으로 부적합 |
| pure ordinal, no visual, no viewpoint | ScanRefer net -1 또는 N 매우 작음 | NR3D는 universal router에서 small gain | 근거 부족 |
| middle/center | 양쪽 모두 net 0 수준 | 양쪽 모두 net 0 수준 | 향상 기준 아님 |
| left/right | 양쪽 모두 net negative | 양쪽 모두 net negative | 부적합 |
| corner/stall | 양쪽 모두 net negative | 양쪽 모두 net negative | 부적합 |

추가로 공통 token/phrase 기준을 탐색했지만, 최소 N=3 이상에서 ScanRefer와
NR3D가 동시에 net positive가 되는 기준은 발견되지 않았다. `facing`은
ScanRefer에서 net +1이지만 NR3D에서는 net 0으로, 향상 기준이라기보다
case-dependent한 보조 단서에 가깝다.

따라서 현 단계에서 BEV + position prompt를 hard replacement route로 사용하는
보편적 기준은 확인되지 않았다. BEV는 최종 pipeline의 핵심 route라기보다,
향후 `E0 + BEV` 형태의 보조 evidence 또는 verifier/reranker로 재설계하는 것이
더 타당하다.

## 11. 기존 broad policy와의 비교

| Dataset | E0 | Broad query-type routing | Dictionary evidence router |
|---|---:|---:|---:|
| ScanRefer | 0.504 | **0.536** | 0.520 |
| NR3D | 0.612 | 0.612 | **0.652** |

Broad policy는 ScanRefer에 유리하지만 NR3D에서 일반화되지 않는다. Dictionary
evidence policy는 최고 성능은 아니지만, dataset-specific calibration 없이 두
데이터셋 모두에서 baseline보다 좋아진다.

## 12. 해석

1. 보편적으로 가장 안정적인 non-E0 route는 spatial-only text이다.
   `proximity_derived`에서 ScanRefer와 NR3D 모두 E0보다 좋아진다.
2. BEV는 pure ordinal에만 제한적으로 써야 한다. visual attribute나 local
   viewpoint가 섞이면 E0가 더 안전하다.
3. NR3D의 `viewpoint_guided`는 global BEV가 아니라 local-frame spatial input이
   필요하다. 현재는 해당 입력이 없으므로 E0가 기본값이다.
4. Query type은 단일 label이 아니라 evidence requirements로 보아야 한다.
   `ordinal + visual attribute`, `viewpoint + color anchor` 같은 mixed query는
   hard routing에서 regression을 만들 수 있다.
5. 향후 개선 방향은 `local-frame spatial text`와 `visual+spatial hybrid input`
   이다. 이 두 입력이 생기면 view-dependent와 mixed query를 더 객관적으로
   처리할 수 있다.

## 13. 주요 산출물

| Artifact | Path |
|---|---|
| Dictionary evidence router | `tools/query_type_router.py` |
| Dictionary policy evaluator | `tools/evaluate_universal_evidence_router.py` |
| Dictionary output summary | `experiments/full_ablation/outputs/universal_evidence_router/summary.json` |
| ScanRefer dictionary results | `experiments/full_ablation/outputs/universal_evidence_router/scanrefer_universal_evidence_routed_results.jsonl` |
| NR3D dictionary results | `experiments/full_ablation/outputs/universal_evidence_router/nr3d_universal_evidence_routed_results.jsonl` |
| Final pipeline README | `experiments/full_ablation/outputs/universal_evidence_router/README.md` |
| Input format visualizations | `experiments/full_ablation/outputs/input_format_visualizations/` |
| Visualization generator | `tools/create_input_format_visualizations.py` |
| NR3D routed-input provenance runner | `tools/run_nr3d_query_type_routed_vlm_eval.py` |

LLM route-type parser 관련 파일은 후속 비교용이며, 이 문서의 최종 성능 기준에는
포함하지 않는다.
