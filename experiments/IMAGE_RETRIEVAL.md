# 처음 보는 이미지의 태그 추천 검사 (2026-10-11)

**이미지 파일만 넣어 후보 태그를 추천하는 경로는 작동합니다. 그러나 기존 이미지
캐시에 픽셀 변환 결함이 있었고, 캐릭터·작가·작품을 자동 확정할 품질은 아닙니다.**
태그 관측과 공출현 관계 저장은 이 이미지 특징 문제와 별개입니다.

## 입력과 근거의 경계

- 저장된 test 관측에서 g/s/q/e 각 32장, 총 128장을 ID 해시 순서로 사전 선택했습니다.
- 이미지 픽셀만 인코딩합니다. 정답 태그, 이름, 등급, 파일명 텍스트를 인코더나
  태그 투표에 넣지 않습니다. 등급은 표본 선정과 사후 집계에만 사용했습니다.
- train 이미지 127,256장의 특징만 검색하며 이미지 SHA-256이 같은 자료를 제외합니다.
- 이미지 벡터와 cosine이 높은 관측 32개를 가져와 태그를 가중 투표합니다.
- 작가·캐릭터·작품 후보는 각 3개씩 별도로 돌려줍니다. 후보에는 근거 이미지 ID가 붙습니다.
- VRS 코어의 `A`는 **이웃 관측에 해당 태그가 기록돼 있다**는 판정입니다.
  질의 이미지에 그 태그가 실제로 맞다는 시각 판정은 `U`로 유지합니다.
- 모델을 새로 학습하지 않았습니다. VRS train에서 보지 않은 이미지라는 뜻이며,
  기반 모델의 사전학습 데이터와 겹치는지는 알 수 없습니다. 유사 구도·재압축·크롭 등
  근접 중복까지 제거한 평가는 아닙니다.

## 기존 캐시 결함 재현

당시 로컬 Transformers 소스
`/mnt/storage/Projects/training/.venv/Lib/site-packages/transformers/models/gemma4/modeling_gemma4.py:603`
에는 아래 코드가 있습니다.

```python
pixel_values = 2 * (pixel_values - 0.5)
hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
```

NF4 양자화의 저장 자료형에 따라 픽셀까지 `uint8`로 변환되어 명암·색상 정보가
손실됩니다. `ImageEncoder(legacy_uint8=True)`는 이 과거 동작의 진단용 재현입니다.
정상 추론은 기본값인 `legacy_uint8=False`를 사용하지만, 그 벡터를 과거 캐시와
섞어 검색하면 안 됩니다. 질의와 인덱스의 특징 추출 방식이 같아야 합니다.

캐시의 원래 모델은 `p-e-w/gemma-4-E2B-it-heretic-ara`이며 리비전은
`c9a1d4c031981f14d86eeb0c7d87de7fafd34513`입니다. 언어모델 전체가 아닌
이미지 인코더의 변경하지 않은 텐서 658개, 334,730,112바이트만 분리해 준비했습니다.
processor 판본과 파일별 해시는 인코더 폴더의 `provenance.json`에 있습니다.

고정한 train 이미지 8장에서 과거 동작을 재현한 벡터와 캐시 벡터의 cosine은
약 0.999965~1.0이며, 모두 자기 이미지가 1위였습니다. 이는 **과거 동작 재현**의
증거이며 이미지 의미를 잘 구별한다는 검증은 아닙니다.

## 측정 결과

`P@10`은 상위 10개 추천 중 원본 기록 태그와 일치한 비율입니다. 기록 태그는
완전하지 않으므로 시각적 정확도와 같지 않습니다. 미추천 빈칸도 분모 10에 포함합니다.

| 기존 전체 인덱스: 127,256장 | P@10 | P@20 |
|---|---:|---:|
| 이미지 없이 전역 빈도만 사용 | 36.33% | 29.30% |
| 기존 캐시와 같은 결함을 재현해 검색 | 41.25% | 31.88% |

전체 캐시 수정 전에 g/s/q/e 각 512장, 총 **같은 train 2,048장**으로 비교했습니다.
NF4 설정, processor, 태그 투표 규칙, 질의 이미지 128장은 동일하며 픽셀 변환을
바꿨습니다. 수정 쪽은 train 이미지도 모두 다시 인코딩했습니다.

| 동일한 2,048장 인덱스 | P@10 | P@20 |
|---|---:|---:|
| 이미지 없이 전역 빈도만 사용 | 42.89% | 31.84% |
| 기존 정수 변환 | 43.91% | 30.20% |
| 정상 픽셀 변환으로 재추출 | **53.52%** | **39.65%** |

수정 인덱스 P@10은 g **49.38%**, s **54.38%**, q **52.50%**, e **57.81%**입니다.
128장 모두 같은 장수를 넣은 평균이며 원래 데이터의 등급 비중을 반영한 평균은
아닙니다. 두 표는 인덱스 규모와 등급 비중이 다르므로 직접 성능 향상으로 비교하면
안 됩니다. 수정 비교는 같은 보류 자료를 재사용한 진단이지 별도의 확증 평가가 아닙니다.

| 이름 첫 후보 적중 | 기존 전체 127,256장 | 같은 2,048장: 기존 → 수정 |
|---|---:|---:|
| 작가 | 5/28 (17.86%) | 0/28 → 0/28 |
| 캐릭터 | 8/117 (6.84%) | 1/117 → 11/117 (9.40%) |
| 작품 | 16/125 (12.80%) | 9/125 → 23/125 (18.40%) |

분모는 해당 종류의 정답 태그가 기록된 이미지 수입니다. 작가 정답은 128장 중
28장만 있어 평가 범위가 작습니다. 2,048장 인덱스에 정답 이름이 한 개라도 있는
질의는 작가 4장, 캐릭터 91장, 작품 122장이며, 나머지도 위 분모에 포함했습니다.
관측되지 않은 이름을 거부하는 기능은 검증되지 않았습니다. 빈도를 근거로 생긴
성인 태그 등 잘못된 후보가 일반 이미지에도 섞여, 자동 태깅에 바로 쓰면 안 됩니다.

고정 g 표본 세 번째인 `11108168`(로젤리나)은 수정 후 캐릭터 첫 후보가 `rosalina`로
맞았지만 작품 첫 후보는 틀렸습니다. 이름 한 가지를 맞힌 사례를 전체 인식 성공으로
해석하지 않습니다. 실제 출력은 각 결과 JSONL에 생략 없이 남겼습니다.

## 자원과 보존 상태

- RTX 4070 Ti SUPER를 UUID로 지정했고 CPU affinity 및 연산 스레드는 2개입니다.
- 전체 인덱스 + 이미지 인코더의 프로세스 VRAM 최대 표본은 **1,140MiB (1.11GiB)**,
  RAM 최대 RSS는 **3,159MiB**였습니다. 0.5초 간격 표본은 순간 최대치가 아닙니다.
- 전체 인덱스에서는 로드 후 이미지당 중앙값 **0.095초**, P95 **0.107초**였습니다.
  로드·호환성 검사·128장 처리까지 스크립트 내부 시간은 **19.84초**입니다.
- 2,048장 수정 비교 전체는 **111.42초**, 수정 검색 중앙값 **0.048초**였습니다.
  수정 비교 GPU 프로세스 표본 최대는 662MiB이며, 짧은 초기 할당에 대한 PyTorch
  기록 최대는 746MiB로 표본보다 큽니다. 662MiB를 절대 최대 사용량으로 읽으면 안 됩니다.
- Writer와 이미지 생성 모델은 올리지 않았습니다. 종료 후 4070은 266MiB, 사용률 0%입니다.
- 원본 데이터, 기존 특징 캐시, 기존 전체 VRS 스냅샷은 덮어쓰지 않았습니다.
  수정 벡터는 별도 비교 폴더에만 있고 전체 인덱스에는 아직 반영하지 않았습니다.

오프라인 회귀 테스트 **78개 통과**. 별도 산출물 검사에서 코드·파일 해시 15개,
예측 384건, 근거 ID 참조 106,132회, 원본 이미지 128장의 바이트와 평가 분리,
집계 수치를 확인했습니다. 근거 참조 검사는 후보의 시각적 정답을 보장하지 않습니다.

## 파일과 재실행

- `prepare_image_encoder.py`: 원래 모델의 vision tensor만 HTTP range로 다운로드합니다.
- `vrs_image.py`: GPU 이미지 인코딩·train 한정 검색·태그 및 이름 후보와 근거 반환.
- `run_image_trial.py`: 캐시 재현 검사 후 128장 전체 인덱스 평가.
- `run_image_repair.py`: 같은 2,048장으로 정수 변환과 정상 픽셀 변환 비교.

이번 실행 환경은 torch 2.14.0+cu130, transformers 5.16.1, bitsandbytes 0.50.2,
accelerate 1.15.0, Pillow 12.3.0입니다. bitsandbytes/accelerate는 아래 인코더 폴더의
`deps`에서 불러오며 ComfyUI 상주 프로세스에는 적용하지 않았습니다.

저장 위치:

- `/mnt/storage/Projects/rs_seasonal_vision_e2b_c9a1d4c/provenance.json`
- `/mnt/storage/Projects/rs_seasonal_vrs_image_trial_20261011_v1/report.json`
- `/mnt/storage/Projects/rs_seasonal_vrs_image_trial_20261011_v1/results.jsonl`
- `/mnt/storage/Projects/rs_seasonal_vrs_image_repair_20261011_v1/report.json`
- `/mnt/storage/Projects/rs_seasonal_vrs_image_repair_20261011_v1/corrected_float.jsonl`
- `/mnt/storage/Projects/rs_seasonal_vrs_image_audit_20261011.json`

저장소 루트에서 실행합니다. 기존 출력 디렉터리는 덮어쓰지 않으므로 새 경로를
지정하세요. 다음 명령은 **기존 결함을 재현하는 진단**이며 제품용 정상 추론이 아닙니다.

```bash
nice -n 15 ionice -c 3 taskset -c 0,1 env \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=GPU-c8460706-b04f-d00d-2a88-b036f96fc417 \
  PYTHONPATH=/mnt/storage/Projects/rs_seasonal_vision_e2b_c9a1d4c/deps:. \
  /home/raspie/AI/ComfyUI/.venv/bin/python -m experiments.run_image_trial \
  --memory /mnt/storage/Projects/rs_seasonal_vrs_full_20261011_v1 \
  --dataset /mnt/storage/Projects/danbooru_unified \
  --encoder /mnt/storage/Projects/rs_seasonal_vision_e2b_c9a1d4c \
  --output /mnt/storage/Projects/rs_seasonal_image_trial_next --legacy-uint8
```

수정 비교는 같은 실행 환경에서 모듈을 `experiments.run_image_repair`로 바꾸고
`--legacy-uint8`을 제외한 뒤 새 출력 폴더를 지정합니다. 전체 캐시를 재생성하거나
신규 단보루 자료를 수집하는 기능은 이 진단 명령에 포함되지 않습니다.
