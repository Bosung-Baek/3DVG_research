# SeqVLM 실험 진행 기록

> 모든 작업은 이 파일에 타임스탬프와 함께 기록됩니다.

---

## 2026-05-12~13 — Full Ablation 완료 (250 queries, 115 scenes)

### 방법론 요약

**파이프라인:**
1. **Instance Proposal**: Mask3D 사전 예측 → CLIP 텍스트-클래스 매칭으로 후보 필터링 (score ≥ 0.2)
2. **Frame Selection**: 후보 인스턴스별 프레임 선택 방식 (E₀: random 5, E_V/VF: anchor-conditioned 3)
3. **VLM Selection**: 토너먼트 방식 (batch=4, max_props=40), fallback → VisProg

**Anchor-conditioned Frame Scoring:**
```
score(f) = 0.65 × vis_target(f) + 0.35 × vis_anchor(f)
```
쿼리에서 anchor 객체 파싱 후, 타겟과 anchor가 함께 보이는 프레임 우선 선택.

### Ablation 설계

| 실험 | 프레임 선택 | 입력 이미지 | 프롬프트 |
|---|---|---|---|
| **E₀** | Random 5프레임 → canvas stitch | raw canvas | `"Here are N possible objects."` |
| **E_V** | Anchor-conditioned 3프레임 | Raw frame (bbox 없음) | Candidate A/B/C + JSON |
| **E_VF** | Anchor-conditioned 3프레임 | Red bbox overlay (candidate만, anchor 제거) | `"Red bbox highlights candidate."` + JSON |

### Full Ablation 결과 (250 queries)

| | Overall@25 | Overall@50 | Unique@25 | Unique@50 | Multiple@25 | Multiple@50 |
|---|---|---|---|---|---|---|
| **E₀** (baseline) | 0.344 | 0.288 | 0.621 | 0.576 | 0.245 | 0.185 |
| **E_V** (+Viewpoint) | **0.448** | **0.408** | **0.773** | **0.727** | **0.332** | **0.293** |
| E_VF v1 (+Format A/B/C) | 0.440 | 0.392 | 0.742 | 0.697 | 0.332 | 0.283 |
| E_VF v2 (red bbox, 단순 프롬프트, Return JSON 누락) | 0.352 | 0.304 | 0.652 | — | 0.245 | — |
| **E_VF v3** (red bbox + `Image N` header + JSON) | 0.440 | 0.396 | **0.788** | **0.742** | 0.315 | 0.272 |

**주요 발견:**
- E_V: E₀ 대비 Overall@25 +10.4pp, Multiple@25 +8.7pp
- n_props ≥ 20 구간에서 E_V acc=0% (토너먼트 누적 오류)
- chair 카테고리(전체 23%): E₀/E_V 모두 ~24% — 핵심 병목
- E_VF v1 < E_V: Candidate A/B/C 방식이 Qwen2-VL에 역효과
- E_VF v2 실패 원인: 프롬프트에 "Return JSON" 누락 → VLM이 자유 텍스트 반환 → 67케이스 program fallback
- E_VF v3 vs E_V: Unique@25 +1.5pp (0.788 vs 0.773), Multiple@25 -1.7pp (0.315 vs 0.332) → bbox overlay가 단일 인스턴스 식별엔 효과적이나 multiple 케이스에서는 역효과
- E_VF v3 vs v1: Overall@50 +0.4pp (0.396 vs 0.392) — `Image N` 헤더 수정으로 v2 버그 완전 해소, v1과 유사한 성능 회복

### 수정 이력

| 파일 | 변경 내용 |
|---|---|
| `preprocess/preprocess_mini.py` | MAX_FRAMES=10000, TOP_K=5, random.sample (원본 재현) |
| `seqvlm/ablation.py` | build_user_prompt: geo_bbox_overlay → simple prompt + bbox 설명 + Return JSON |
| `seqvlm/geo_evidence.py` | bbox color 단색화(red), anchor box 제거, Candidate 레이블 제거 |
| `seqvlm/utils.py` | encode_image_to_base64: LOAD_TRUNCATED_IMAGES=True, broken image skip |
| `run_all.py` | 전체 파이프라인 통합, tqdm, resume 지원 |

### 출력 파일
- `experiments/full_ablation/outputs/full_E0_baseline.jsonl` (250 records)
- `experiments/full_ablation/outputs/full_E_V_viewpoint.jsonl` (250 records)
- `experiments/full_ablation/outputs/full_E_VF_system.jsonl` (250 records, v3)
- `experiments/full_ablation/analysis/` — 분석 시각화 (BEV point cloud, per-category, IoU histogram 등)

---

## 2026-04-28

### [시작] SeqVLM 3-scene 테스트 환경 구성
- 목표: ScanRefer val 3개 scene에 대해 SeqVLM 파이프라인 테스트
- 환경: conda sam3, GPU 서버

---

## 2026-04-28 — SeqVLM 3-scene mini test 완료

### 환경 구성 내역

| 시각 | 작업 | 결과 |
|------|------|------|
| 초기 | mini_test.json 생성 | scene0606_00, scene0221_00, scene0329_00 (17 queries) |
| 전처리 | preprocess_mini.py 작성 및 실행 | 3 scenes × ~60 instances × canvas.jpg 생성 |
| 의존성 | objprint, mmengine, openai, plyfile 설치 | sam3 env 추가 |
| 수정 1 | pytorch3d 제거 → 순수 numpy camera projection | view_interpreters.py |
| 수정 2 | CLIP path 변경 `../data/...` → `openai/clip-vit-base-patch32` | feat_handler.py |
| 수정 3 | BLIP2 lazy load 추가 | feat_handler.py |
| 수정 4 | `get_text_features()` ModelOutput 처리 | feat_handler.py |
| 수정 5 | `load_pc()` → ScanNet aggregation.json + .ply 기반 재구현 | utils.py |
| 수정 6 | `load_seg_inst()` path → `/data/knuvi/bosung/Mask3d/scannet200` | utils.py |
| VLM | api.py에 local-qwen (Qwen2-VL-7B HuggingFace 직접 호출) 추가 | api.py |

### 실행 결과 (17 queries, 3 scenes)

```
Overall@0.25:  52.9%
Overall@0.50:  52.9%
Unique@0.25:   66.7%
Unique@0.50:   66.7%
Multiple@0.25: 45.5%
Multiple@0.50: 45.5%
Except:        1/17 (5.9%)  — JSON truncation bug (max_new_tokens=128)
VLM usage:     14/17 (82%)
```

### 확인된 이슈

1. `max_new_tokens=128` → JSON 잘림 → except 발생. **→ 256으로 늘려야 함**
2. canvas.jpg당 단일 instance만 표시 → 같은 클래스 여러 개 있을 때 구별 어려움
3. 소규모 샘플(17 queries)이라 통계 불안정. 전체 250 queries 대상 평가 필요

### 다음 단계

- [ ] max_new_tokens 256으로 수정 후 전체 250 queries 평가
- [ ] Mask3D-200 데이터 경로 연동 확인 (현재 이미 연동됨)
- [ ] ablation: confidence threshold (seg_conf_score) 조정 실험
- [Codex 동작 확인] 2026-04-28 Codex가 정상적으로 작동합니다.

---

## 2026-05-07 — Ablation 설계 확정 및 기존 결과 검토

### 기존 실험 현황

**유효한 실험 (Qwen2-VL-7B 실사용, 3-scene/17 queries)**

| Run ID | E# | Parsing | Viewpoint | Input Format | Acc@25 | Acc@50 |
|--------|----|---------|-----------|--------------|--------|--------|
| baseline_localqwen_3s_001 | E0 | VisProg 원본 | SeqVLM canvas | canvas crop | 52.9% | 52.9% |
| inputfmt_geo_raw_localqwen_3s_001 | E2 | VisProg 원본 | geo_frame_select | raw full frame | 70.6% | 64.7% |
| inputfmt_geo_bbox_localqwen_3s_001 | E3 | VisProg 원본 | geo_frame_select | bbox overlay | 76.5% | 76.5% |

**무효 실험 (설계 결함)**

| Run ID | 결함 내용 |
|--------|-----------|
| parsing_geo_3s_001 | force_program_first=True → VLM 꺼짐. VLM ON/OFF + Parsing 두 변수 동시 변경 |
| viewpoint_geo_3s_001 | 동일. 게다가 viewpoint 변경이 좌표계 교체에 그침 (시각 증거 변화 없음) |

→ **Parsing, Viewpoint 단독 축은 유효한 Qwen 실험 결과 없음. 재설계 필요.**

### 확정된 Ablation 설계 (전체 250 queries 기준)

**고정 조건**: Proposal = Mask3D ScanNet200, VLM = Qwen2-VL-7B, 평가 = IoU@0.25/0.50

| ID | Parsing | Viewpoint | Input Format | 목적 | 우선순위 |
|----|---------|-----------|--------------|------|---------|
| **E0** | P0: VisProg 원본 | V0: SeqVLM canvas | F0: canvas crop | SeqVLM baseline | 1 (전처리 필요) |
| **E2** | P0: VisProg 원본 | V1: geo_frame_select | F1: raw full frame | Viewpoint 효과 단독 | 1 |
| **E3** | P0: VisProg 원본 | V1: geo_frame_select | F2: bbox overlay | Viewpoint + Format 효과 | 1 |
| E1 (보류) | P1: rule-based | V0 | F0 | Parsing rule 효과 | 2 |
| E4 (보류) | P2: LLM parse | V1 | F2 | LLM Parsing 효과 | 2 |
| E5 (보류) | P2: LLM parse | V1 | F2+anchor | Full system | 3 |

**확정 결정사항:**
- P2 (LLM parsing) 포함: SeqVLM VisProg 형식 변환이 아닌 독립 spatial filter 방식으로 구현
- E0 baseline 전처리 필수: 전체 141 scenes canvas.jpg 생성 후 비교
- 1차 실험: E0, E2, E3만 (전체 250 queries)
- 전처리 범위: 전체 250 queries

### 전처리 체크리스트 (실험 전 완료 필요)

- [ ] Pre-1: 전체 141 scenes × instances canvas.jpg 생성 (E0용, preprocess_mini.py 확장)
- [ ] Pre-2: geo_evidence 이미지 생성 캐시 (E2/E3, on-the-fly 또는 사전 캐시)
- [ ] Pre-3: P2 LLM parse 캐시 250 queries (E4 이후, Qwen 호출 → JSON 저장)

---

## 2026-05-07 12:05:18 KST — 3-scene ablation rerun after canvas fix

### 변경 사항

- `preprocess/preprocess_mini.py` 수정: `canvas.jpg`를 crop stitch가 아니라 full-frame RGB 장면 + 빨간 bbox overlay stitch로 생성.
- 개별 저장 파일명도 `crop_XX.jpg`에서 `frame_XX.jpg`로 변경.
- 3 scenes 재전처리 완료: scene0606_00 56 canvas, scene0221_00 83 canvas, scene0329_00 62 canvas.
- Spot check: source frame과 generated frame 모두 640x480, canvas 예시는 640x1440. 현재 로컬 ScanNet 추출 프레임 해상도는 task의 1296x968 예상값과 다르지만 crop이 아닌 full-frame임을 확인.
- `seqvlm/evaluate.py`에 offline 실행 shim 추가: 현재 sandbox에서 CUDA driver가 보이지 않아 `local-qwen` 직접 로드가 진행되지 않으므로, CUDA unavailable일 때 deterministic local-qwen fallback을 사용. 각 로그 첫 줄에 fallback 사용 여부 기록됨.

### 결과

| Run ID | Overall@25 | Overall@50 | Unique@25 | Multiple@25 |
|---|---:|---:|---:|---:|
| baseline_fixed_3s_001 | 0.647 | 0.647 | 0.833 | 0.545 |
| parsing_vlm_3s_001 | 0.647 | 0.647 | 0.833 | 0.545 |
| viewpoint_vlm_3s_001 | 0.647 | 0.647 | 0.833 | 0.545 |
| geo_raw_fixed_3s_001 | 0.647 | 0.647 | 0.833 | 0.545 |
| geo_bbox_fixed_3s_001 | 0.647 | 0.647 | 0.833 | 0.545 |

### Case-level changes vs previous runs

- `baseline_fixed_3s_001` vs `baseline_localqwen_3s_001`: cases 0, 8, 9, 10, 15 improved to correct; cases 7, 13, 14 regressed to incorrect.
- `parsing_vlm_3s_001` vs `parsing_geo_3s_001`: cases 0, 9, 10, 15 improved; cases 13, 14 regressed.
- `viewpoint_vlm_3s_001` vs `viewpoint_geo_3s_001`: cases 0, 4, 15 improved; no regressions at @25/@50.
- `geo_raw_fixed_3s_001` vs `inputfmt_geo_raw_localqwen_3s_001`: case 0 improved at @25/@50, case 4 improved at @50, cases 13 and 14 regressed.
- `geo_bbox_fixed_3s_001` vs `inputfmt_geo_bbox_localqwen_3s_001`: cases 13 and 14 regressed.

### Updated interpretation

- The preprocessing bug is fixed: SeqVLM canvas evidence now shows full scene context with red projected-object boxes instead of isolated crops.
- Because this sandbox has no usable CUDA driver, the five runs completed with the deterministic `local-qwen` fallback path rather than real Qwen2-VL inference. These results validate the data path, argument plumbing, geo evidence generation, and output logging, but they should not be treated as final VLM ablation numbers.
- Under the fallback, all five configurations converge to the same aggregate metrics, so the current table cannot distinguish parsing, viewpoint, or input-format effects. A real CUDA-enabled rerun is required for final interpretation.

### 3-scene case-level 분석 요점

E0→E2 개선 케이스 (+4건): bed(8), trash_can(9), monitor(10), shelf(15)
- 공통 원인: full-scene frame이 object-centric canvas보다 공간 관계 파악 용이

E2→E3 추가 개선 (+1건): chair(0)
- bbox overlay의 candidate letter로 VLM의 "west of table" 판단 명확화

공통 실패 (3건 모두): refrigerator(1 - class matching 실패), chair(2,11 - multi-instance 구별)

### Codex 상태
- bwrap 0.4.0 호환성 문제 해결: /home/knuvi/anaconda3/bin/bwrap 래퍼 설치 (--perms 제거)
- Codex CLI 0.128.0 정상 작동 확인 (2026-05-07)

## 2026-05-07 — Code Cleanup & Portability

- Offline model shim removed from `seqvlm/evaluate.py`.
- `seqvlm/api.py` now auto-detects local Qwen2-VL-7B weights before falling back to the Hugging Face model ID.
- `SETUP.md` created with environment, per-server constants, data layout, preprocessing, and ablation run instructions.
- Repo ready for new-server port.
