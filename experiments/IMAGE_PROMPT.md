# 이미지 참고로 새 프롬프트 만들기

2026-10-11. 사용 목적은 이미지의 원래 태그를 맞히는 것이 아니라,
이미지에서 회수한 태그를 재료로 새 이미지 생성 프롬프트를 자동 작성하는 것이다.
`experiments.image_prompt`는 이 흐름을 수행하는 독립 CLI다.
ComfyUI 노드 배포나 후속 이미지 생성까지 검증한 것은 아니다.

## 동작

1. 이미지 인코더가 입력 파일을 특징 벡터로 바꾼다. 정상 픽셀 변환을 사용한다.
2. 수정 특징 인덱스에서 유사 관측을 검색한다. 입력과 SHA-256이 같은 파일은 제외한다.
   원본 입력의 정답 태그는 읽지 않는다. VRS CUDA 판정 코어는 검색 근거 처리에 사용한다.
3. 검색한 관측에서 이름과 시각 속성을 가져온다. 작가·캐릭터·작품을 각각 독립적으로
   섞지 않고 한 관측에서 함께 선택한다. 시각 속성도 같은 관측을 사용한다.
4. 사용자 방향이 없으면 기존 `seasonal_backgrounds.csv`, `seasonal_times.csv`에서
   시드에 따라 새 계절·장소·시간을 고른다. 한국어/영어 방향을 주면 그것을 우선한다.
5. 이미지 인코더와 검색 벡터를 해제한 뒤 Qwen3-4B-Instruct-2507 Q4_K_M Writer로
   새 태그형 또는 문장형 프롬프트를 작성한다. Writer 인스턴스는 호출마다 닫는다.

`--identity-mode borrow`가 기본이다. 선택한 이름을 창작 재료로 전달하며 입력 이미지의
정체를 인식했다고 주장하지 않는다. `original`은 이름을 Writer 재료에서 제외하고
이름 없는 새 인물을 작성한다. 원본 VRS의 작가·캐릭터·작품 및 전체 검색 후보는 유지한다.
원하는 캐릭터나 작가는 `--direction`으로 명시할 수 있다. 지시 이행은 모델의 능력에
의존하므로 이름 보존이나 장면 무모순을 보장하지 않는다.

기본 검색 범위는 `g,s,q,e` 전체다. `--ratings`는 검색 범위이며 입력 등급 판정이 아니다.
이번 시연은 일반 등급 입력에 대해 `g,s` 범위로 검색했다.

## 현재 로컬 실행

아래는 이미 준비한 로컬 모델과 분리된 비전 의존성을 사용하는 Linux 실행 예다.
새 설치·다운로드는 수행하지 않는다. 출력 폴더는 매번 새로운 경로를 지정한다.

```bash
cd /home/raspie/문서/ChatGPT/RS-Seasonal-Prompt-Generator
nice -n 15 ionice -c 3 taskset -c 0,1 env \
  CUDA_VISIBLE_DEVICES=GPU-c8460706-b04f-d00d-2a88-b036f96fc417 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/mnt/storage/Projects/rs_seasonal_vision_e2b_c9a1d4c/deps:. \
  /home/raspie/AI/ComfyUI/.venv/bin/python -m experiments.image_prompt \
  --image /path/to/reference.png \
  --memory /mnt/storage/Projects/rs_seasonal_vrs_full_20261011_v1 \
  --corrected-index /mnt/storage/Projects/rs_seasonal_vrs_image_repair_20261011_v1 \
  --encoder /mnt/storage/Projects/rs_seasonal_vision_e2b_c9a1d4c \
  --output /mnt/storage/Projects/my_new_image_prompts \
  --format both --variants 2 --seed 42
```

선택 옵션:

- `--direction '겨울 밤의 마을 축제에서 등불을 들고 걷는 장면으로 바꿔줘.'`
- `--identity-mode original`: 이름 없는 새 캐릭터를 구성한다.
- `--format tags|natural_language|both`: 출력 형식.
- `--variants 1..4`, `--seed`: 새 장면 후보. 두 출력 형식은 같은 시드의 장면 힌트를 받지만
  별도 생성이므로 내용이 정확히 일치하는 번역 쌍은 아니다.
- `--ratings g,s,q,e`: 검색 범위를 사용자가 제한할 수 있다.

모든 프롬프트는 `prompt_<번호>_<형식>.txt`에 저장한다. `reference.json`에는 입력
해시, 검색 근거, 선택한 참고 재료가 들어 있다. `messages_*.json`에는 실제 Writer 입력,
`report.json`에는 모델·코드·CSV·산출물 해시와 자원 측정이 들어 있다.

## 실제 확인

입력은 기존 보류 일반 등급 이미지 `11108168.jpg`다. 이름과 정답 태그를 조회 조건에
주지 않고 정상적으로 재추출한 **2,048장** 검색 인덱스만 사용했다. 전체 VRS 관측과
연결하지만 손상된 기존 127,256장 이미지 특징으로 자동 폴백하지 않는다.

- 자동 생성: `/mnt/storage/Projects/rs_seasonal_image_prompt_20261011_auto_final/`
  - 시드 42: `autumn, outdoor, forest, late afternoon`
  - 시드 43: `autumn, outdoor, mountain path, afternoon`
  - 이번 검색에서 선택한 창작용 이름 묶음은 `hoshimachi suisei`, `hololive`다.
    입력 그림의 캐릭터가 스이세이라는 인식 결과가 아니다.
- 방향 지정: `/mnt/storage/Projects/rs_seasonal_image_prompt_20261011_winter_final/`
  - 이름 없는 인물로 겨울 축제·등불·걷는 장면을 지시했다.

측정: RTX 4070 Ti SUPER, CPU 보조 스레드 2개, 논리 CPU 0·1에 제한,
GPU 프로세스 최대 관측 **3,578MiB ≈ 3.49GiB**. 0.5초 간격 측정이므로 순간 최대값은
아니다. 자동 생성 실행의 RAM 최대 RSS는 약 4.90GiB다. Writer와 이미지 생성 모델을
동시 실행하지 않았다. 검색 모델 해제 후 PyTorch 활성 할당은 약 8.13MiB였으며,
CUDA 컨텍스트 및 다른 라이브러리 점유는 이 값에 포함되지 않는다.

오프라인 회귀 **89개 통과**. 이름·시각 속성 묶음의 출처 일관성, 사용자 방향 우선,
같은 시드의 장면 공유, 인덱스·인코더 판본 검사, Writer 예외 발생 시 해제,
CPU 자동 재시도 금지를 검사한다. 이 테스트는 생성문 품질 점수가 아니다.

## 평가 기준과 남은 문제

완성 프롬프트의 참고 특성 활용, 새 장면의 다양성, 지시 이행, 의상·포즈·환경의 일관성,
실제 이미지 생성에서의 유용성이 목적에 맞는 평가다. 태그 복원 정확도는 검색 진단용
보조 지표로만 둔다.

실제 출력에는 새 계절·장소가 반영됐다. 그러나 자동 생성 일부에서 오후 장면에
원래 참고 태그의 별이 남거나, 전달한 작품명이 빠졌다. 겨울 장면에서도 세부 지시의
보존은 완벽하지 않았다. 완성 프롬프트를 자동으로 최종 채택할 수준의 검증은 아직
없으며, 생성 이미지 평가는 하지 않았다. 전체 이미지 인덱스 재추출도 아직 필요하다.
