# VRS + Danbooru GPU trial

Gemma 내부 FFN 계산 결과를 저장·재사용하는 200장 실험은
[내부 계산 경험 안내](../vrs_int/ACTIVATION_MEMORY.md)를 참고하세요.
이미지·태그 검색과 별개의 실험이며, 가중치나 층을 제거하지 않습니다.

전체 원본 자료용 구현은 [전체 데이터 구축 문서](FULL_DATASET.md)를 참고하세요.
`vrs_full.py`는 희소 GPU 관계 계산을 사용하고, 작가·캐릭터·작품을 빈도 1까지
보존합니다. 아래의 16,384개 관측·3,072개 태그 제한은 초기 `run_vrs_trial.py`에만
해당합니다.

새 이미지 파일만 입력하는 평가는 [이미지 입력 평가](IMAGE_RETRIEVAL.md)에 있습니다.
기존 이미지 캐시의 픽셀 정수 변환 결함을 재현했으며, 기존 벡터는 진단용입니다.
사용 목적에 맞춘 [이미지 참고 → 새 프롬프트 생성](IMAGE_PROMPT.md)은 수정 이미지
검색 결과와 기존 계절 CSV를 Writer에 연결합니다. 태그 복원 점수를 목표로 삼지 않습니다.

기존 이미지·태그 관측을 보존하고, 태그 공출현과 기존 이미지 특징의 유사도를
CUDA에서 계산하는 **독립적인 로컬 실험**입니다. ComfyUI 노드에는 아직 연결하지
않았습니다. 모델 가중치를 재학습하지 않으며, 외부 다운로드·정기 실행도 하지 않습니다.

## 구현 범위

- `vrs_cuda.py`: 외부 VRS의 `association_scalar.hpp`, `evidence_scalar.hpp`,
  `core_platform.hpp`를 읽고 NVRTC로 CUDA 커널을 컴파일합니다. 판정 규칙을 Python으로
  재작성하지 않습니다. 실제 컴파일 입력과 PTX의 해시를 기록합니다.
- `vrs_memory.py`: 이미지 ID·이미지 SHA-256·태그 기록 SHA-256·특징 파일 SHA-256을
  SQLite에 보존합니다. 동일 관측의 재입력은 횟수를 늘리지 않습니다. 같은 이미지의
  다른 ID도 중복 관측으로 세지 않습니다. 기존 ID의 태그·이미지·특징이 바뀌면 실패하며,
  별도의 명시적 판본 처리가 필요합니다.
- 태그 문서 행렬에서 CUDA 행렬곱으로 공출현 횟수를 계산하고 VRS 판정을 저장합니다.
  후보는 근거 이미지 ID와 함께 반환합니다. 입력 태그 전체를 만족하는 관측만 검색하고,
  기록에 명시된 제외 태그가 있는 관측은 제거합니다. 희귀한 제외 태그도 전체 기록으로
  확인합니다. 알 수 없는 필수 태그는 무시하지 않고 `U`로 반환합니다.
- 기존 768차원 토큰 특징은 GPU에서 평균·정규화해 이미지별 검색용 벡터로 저장합니다.
  이미지 간 이웃 조회의 실행만 확인했습니다. 원래 특징 추출 모델의 provenance와
  이웃의 시각적 적합성은 미검증입니다. 텍스트와 정렬된 특징으로 취급하지 않습니다.
- 별도 프로세스 종료로 CUDA 문맥까지 해제합니다. CUDA 불가 시 CPU로 재시도하지 않습니다.

VRS의 `A`는 **두 태그가 기록에 함께 존재한다는 관측 주장**에 한정됩니다.
태그가 이미지 내용을 정확히 묘사한다거나, 사용자 입력에 해당 태그를 추가해도 된다는
뜻이 아닙니다. 태그 누락은 시각적 부재의 반증으로 세지 않으므로 실제 자료의 refute
열은 0입니다. 반박·충돌 분기는 별도 합성 입력에서 기존 C++ 판정과 비교합니다.
이 실험은 전체 VRS 상주 런타임·시냅스·세대 관리의 통합이 아닙니다.

## 2026-10-11 측정 결과

일반 등급(`g`)의 기존 `processed_v1` train 분할에서 ID 해시 순서로 2,048건을 선택하고 256건을
추가했습니다. 원본 이미지를 새로 인코딩하지 않고 이미 받아둔 특징을 사용했습니다.
최종 관측 2,304건, 어휘 3,072개, 방향을 구분한 자기 자신 제외 공출현 쌍 756,040개입니다.

기존 `processed_v1` test 분할의 일반 등급 106건을 평가했습니다. train/test ID 및 이미지 파일
바이트 해시 중복은 없었습니다. 다른 인코딩·크롭·유사 그림의 중복은 검사하지 않았습니다.
두 스냅샷의 교집합 어휘 2,883개와 동일한 마스킹으로 비교했습니다. 평가 기록의 태그 중
10.59%는 이 어휘 밖입니다. 각 기록의 태그 3개를 입력하고 나머지 기록 태그를
맞히는 **태그 복원 검사**이며, 한국어 이해·프롬프트 자연스러움·그림 품질 평가는 아닙니다.

| 방법 | 2,048건 P@10 | 2,304건 P@10 | 2,304건 R@10 |
| --- | ---: | ---: | ---: |
| 전체 태그 빈도만 사용 | 38.30% | 38.30% | 17.00% |
| 태그별 조건부 공출현 비율 평균 | 48.21% | 48.96% | 21.83% |
| 모든 입력 태그의 동시 관측 요구 | 16.60% | 17.26% | 8.36% |

두 번째 행은 일부 입력과의 관계도 점수에 기여하는 느슨한 탐색 평가입니다.
세 번째 행이 보수적인 query 경로에 해당합니다. 후보당 근거 2건 이상을 요구하며,
응답 가능한 테스트는 **41/106건(38.68%)**입니다. 응답하지 않은 경우와 빈 출력 슬롯도
P@10 분모에 포함합니다. 작은 표본에서는 복합 조건을 만족하는 관측이 부족합니다.
따라서 48.96%를 실제 프롬프트 정확도나 엄격 검색의 정확도로 해석하면 안 됩니다.

`hatsune miku`는 29건을 회수했고 `vocaloid` 29건, `twintails` 24건,
`detached sleeves` 22건의 공출현을 제시했습니다. `white dress`는 202건을 회수했지만
관련 후보에 특정 캐릭터·작품 편향이 섞입니다. `umbrella`, `red umbrella`와
`subway platform` 조합, 존재하지 않는 캐릭터는 이 작은 어휘에서 `U`였습니다.
한국어·동의어 번역, 태그 별칭/포함 관계의 공식 정규화는 아직 없습니다.

| 자원 | 최종 검사 |
| --- | ---: |
| GPU | RTX 4070 Ti SUPER 16GB, UUID로 지정 |
| 프로세스 VRAM 최대 표본 | 572MiB (0.5초 간격, 14표본) |
| PyTorch 최대 allocated / reserved | 251.1 / 318.0MiB |
| CPU 보조 스레드 / CPU affinity | 2 / 논리 CPU 0, 1 |
| 호스트 메모리 최대 RSS | 1,446.4MiB |
| 검사 구간 경과 시간 | 9.77초 |

시간은 Python·PyTorch 초기 로드 후 측정하며, OS 파일 캐시가 가열된 재검사 결과입니다.
최초 실행 시간이나 전체 데이터 구축 시간을 나타내지 않습니다. VRAM은 연속 최대치가
아닌 표본 최대치입니다. Writer LLM을 함께 올리지 않았습니다. PyTorch 할당 한도는
VRAM의 20%이며 CUDA 드라이버 등 모든 GPU 할당을 제한하는 상한은 아닙니다.
실험 종료 후 4070은 기존 사용량 약 266MiB로 돌아왔습니다.

검증: 오프라인 64개 테스트 통과, 네 판정 분기를 포함한 CPU/CUDA 12입력 비교 통과,
빈 GPU 배치·음수 거부 검사 통과, 공출현 24쌍 독립 계수 일치, 반환된 근거 참조
96건 일치, 재입력 32건의 관측 수 불변, 필수/제외 충돌 및 미지 필수 조건 처리 확인.

로컬 전체 보고서와 immutable 스냅샷:

- `/mnt/storage/Projects/rs_seasonal_vrs_trial_20261011_checked/report.json`
- `/mnt/storage/Projects/rs_seasonal_vrs_trial_20261011_checked/ledger.sqlite3`
- `/mnt/storage/Projects/rs_seasonal_vrs_trial_20261011_checked/final/`

## 재실행

Linux, CUDA PyTorch, NumPy, 해당 Python 환경의 NVIDIA NVRTC 라이브러리,
NVIDIA 드라이버, C++17 `g++`, 외부 VRS 헤더가 필요합니다. 자동 설치는 없습니다.
ComfyUI 환경의 **Python 실행 파일만** 사용하며 ComfyUI 서버를 시작하거나 재시작하지 않습니다.
환경·데이터·VRS 경로와 GPU UUID는 해당 시스템에 맞게 지정하세요.

```bash
cd '/home/raspie/문서/ChatGPT/RS-Seasonal-Prompt-Generator'
nice -n 15 taskset -c 0,1 env \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  CUDA_VISIBLE_DEVICES=GPU-c8460706-b04f-d00d-2a88-b036f96fc417 \
  /home/raspie/AI/ComfyUI/.venv/bin/python -m experiments.run_vrs_trial query \
  --memory /mnt/storage/Projects/rs_seasonal_vrs_trial_20261011_checked/final \
  --tags 'hatsune miku' --exclude 'twintails'
```

새로운 실험은 기존에 없는 출력 디렉터리를 지정합니다. 생성물로 기존 스냅샷을
덮어쓰지 않습니다. `--initial`과 `--append`는 각각 최초 관측과 추가 관측 수이며,
추가 관측도 이번 실험에서는 **기존 로컬 자료**에서 가져옵니다.

```bash
nice -n 15 taskset -c 0,1 env \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  CUDA_VISIBLE_DEVICES=GPU-c8460706-b04f-d00d-2a88-b036f96fc417 \
  /home/raspie/AI/ComfyUI/.venv/bin/python -m experiments.run_vrs_trial trial \
  --dataset /mnt/storage/Projects/danbooru_unified \
  --vrs-core /var/home/raspie/vrs9/core \
  --output /mnt/storage/Projects/rs_seasonal_vrs_trial_next \
  --initial 2048 --append 256
```

실험용 상한은 관측 16,384건, 어휘 3,072개입니다. 공출현은 조밀한 행렬로 계산하며
추가 후 재계산합니다. 전체 13만 건의 온라인 증분 인덱스 구현이 아닙니다. 어휘가
많아질 때 빈도 상한으로 희귀 캐릭터가 밀릴 수 있으므로 전체 규모에서는 태그별
희소 인덱스와 신종 캐릭터 별도 평가가 필요합니다.

## 신규 자료를 계속 추가하려면

연결할 흐름은 `신규 post ID 수집 → 원본·태그 판본 기록 → 이미지 중복 확인 →
동일한 추출기로 특징 생성 → 새 관측 추가 → 기존 평가와 신규 캐릭터 평가 →
스냅샷 채택`입니다. 기존 이미지의 수정된 태그를 새 증거로 중복 누적하면 안 됩니다.
지금은 관측 추가·재입력 방지·새 스냅샷 평가까지 검증했으며, 신규 다운로드·태그 수정
판본 정책·동일 추출기의 확인·자동 스케줄은 미구현입니다. 관측이 늘어도 이전 조건의
성능이 떨어지거나 특정 작품 편향이 커질 수 있으므로 자동으로 개선됐다고 판정하지 않습니다.
