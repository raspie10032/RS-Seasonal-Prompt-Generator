# VRS-INT → TIPO 생성 상태 실험 / Generation state experiment

이미지 검색으로 얻은 **기존 추천 JSON**을 생성용 상태로 정리하고, 로컬 TIPO
언어모델에 텍스트로 전달하는 독립 실험입니다. 신경망 내부 상태 주입이나 재학습이
아닙니다. ComfyUI 노드에 자동 연결하지 않으며, 저장된 추천을 사용하므로 이 단계에서
이미지 인코더·원본 이미지·전체 기억을 다시 읽지 않습니다.

This experiment exports an existing VRS-INT recommendation into a structured creative
state and serializes it as text for a local TIPO language model. It is not hidden-state
injection, adapter training, or a ComfyUI integration. Image retrieval must already have
produced the recommendation; this stage does not read images, the encoder, or the store.

Optional: [local tag dictionaries and reference selection](TAG_DICTIONARY.md)
extend the built-in vocabulary. Add `--tag-dictionary /path/to/tags.csv` for
category/Korean meaning annotations, and `--resolve` to select a single-subject
recipe before writing. These are opt-in; the default limits below describe the
original dictionary-free state. Resolved states retain the full classified pool
and store a bounded selection separately in `writer_reference`.

선택적으로 [개인 태그 사전과 재료 선택](TAG_DICTIONARY.md)을 사용할 수 있습니다.
`--tag-dictionary`는 의미·분류를 보완하고 `--resolve`는 생성용 재료를 추립니다.
원본 경험을 바꾸지 않으며, 아래의 후보 수 제한은 기본 사전 미사용 경로 기준입니다.

## 상태가 담는 내용 / State contents

- 사용자가 지정한 유지 태그 `keep`, 새 장면 태그 `target`, 제외 태그 `exclude`.
- 작가·캐릭터·작품 **후보** 각각 최대 3개. 검색 투표는 정답 확률이 아닙니다.
- 외형·의상·행동/표정·장소·스타일/구도별 참고 태그 각각 최대 5개.
- 후보별 투표와 근거 관측 키, 같은 관측에서 함께 나온 태그, 정확한 snapshot과
  원본 추천 내용의 SHA-256. 모델 입력에는 키와 해시를 생략하고 짧게 전달합니다.
- 품질·연도 태그는 걸러내고 명시적인 이름 분류는 보호합니다. 시각 판정은 `U`.

The scene vocabulary is intentionally small and deterministic. A tag must have an
explicit `general` category and match the vocabulary to enter a scene group. Unknown
tags are counted as omitted and remain in the original recommendation. This is a
limited selector, not a trained semantic classifier; its omissions constrain results.
Source captions and recommendations are never modified. Reference names remain
candidates, while user-specified names are explicit controls.

## 상태만 내보내기 / Export without a model

저장된 `recommendation.json`은 `vrs_int recommend` 출력 JSON을 파일로 저장한 것입니다.
`--keep`, `--target`, `--exclude`는 태그 하나마다 반복합니다. 출력 파일이 존재하면
덮어쓰지 않습니다. 입력·출력 경로는 코드 저장소 밖의 개인 디렉터리를 쓰세요.

```bash
python -m vrs_int.generation_state \
  --recommendation /private/vrs/recommendation.json \
  --request 'Keep the chosen character and move the scene to a winter market.' \
  --keep 'example character' --keep 'example series' \
  --target 'winter' --target 'night market' --target 'coat' \
  --exclude 'beach' \
  --output /private/vrs/generation-state.json
```

Python: `build_state(recommendation, request, keep=..., target=..., exclude=...)`
returns the export. `messages_for(state, "state")` returns chat messages. Other modes
are `"request"` and `"flat"`; they keep the same request and explicit controls. The
flat and state modes contain **the same selected reference-tag set**. Structured mode
adds categories, identity votes and up to three co-occurrence examples, so this tests
the full structured presentation, not JSON punctuation alone.

`keep`/`target`/`exclude`는 이 실험에서 사람이 명시한 조건입니다. 자연어 요청에서
이를 자동 추출하는 기능은 아직 없습니다. 따라서 조건 태그를 포함한 `request`
기준군은 자연어 문장만 준 시험과 구분해야 합니다.

## GPU 비교 실행 / Paired GPU comparison

먼저 동작하는 CUDA `llama-cpp-python`과 **이미 설치된 언어모델 GGUF**를 준비합니다.
기존 TIPO에서 쓰는 Gemma 모델 파일을 지정할 수 있습니다. `mmproj`는 필요하지 않습니다.
이 실행기는 모델/패키지를 자동 다운로드하거나 CPU 추론으로 폴백하지 않습니다.
GPU UUID는 `nvidia-smi -L`에서 고르고, 실행 중인 다른 작업의 메모리 여유를 확인합니다.
CPU 스레드는 2개, context는 4096, 출력 상한은 512 토큰입니다.

개인 `cases.json` 예시(아래 이름은 설명용 가상 이름):

```json
[
  {
    "name": "winter_market",
    "recommendation": "/private/vrs/recommendation.json",
    "request": "Keep the selected character, wearing a coat at a winter market.",
    "keep": ["example character"],
    "target": ["coat", "winter", "night market"],
    "exclude": ["beach", "swimsuit"]
  }
]
```

프로젝트 루트에서 실행합니다. 아래 GPU UUID와 파일 경로는 실제 로컬 값으로 바꿉니다.
출력 디렉터리는 **새 경로**여야 합니다. 기존 결과를 덮어쓰지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=GPU-your-device-uuid OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m experiments.run_generation_state_trial \
  --cases /private/vrs/cases.json \
  --model /private/models/your-tipo-model.gguf \
  --output /private/vrs/trial-001 \
  --seeds 42 91
```

기존 TIPO의 `Partial tags:` 형식도 비교하려면 새 출력 디렉터리를 지정하고
`--framing tipo-partial`을 추가합니다. 기본 `creative` 형식은 유지·목표·제외를
각각 표기하며, `tipo-partial`은 같은 유지·목표 태그를 초기 태그 줄에 넣고 기존
`SYSTEM_TIPO` 지시문을 사용합니다. 두 형식의 결과는 별도 실험으로 해석합니다.

Use `--framing tipo-partial` with a new output directory to check the existing TIPO
input convention. This changes the system prompt and placement of required tags;
it is a separate framing experiment, not another seed of the default comparison.
All three arms within each framing still share controls and the same reference-tag set.

The runner is for a Linux CUDA workstation (`nvidia-smi` and `resource` are used for
measurements). It uses the existing `tipo_engine.load_tipo` loader, requests all layers
on the selected GPU, closes only its own model, and never calls the normal node's
constraint-fixing postprocessor. Check the runtime logs to confirm actual offload.
For layer logs, set `RS_TIPO_VERBOSE=1`; this also emits large inference diagnostics.
An installed CUDA runtime must be visible to the loader as in your working TIPO setup.
GPU layer offload does not mean all weight bytes reside in VRAM: host-mapped tensors
and system RAM can remain. VRAM here covers only the writer process, not concurrent
applications or the earlier image-encoder/retrieval stage. The first generation may
include kernel initialization, so inspect individual timings before quoting latency.

## 결과 읽기 / Interpretation

`report.html`에서 요청별·시드별 세 출력 원문을 나란히 봅니다. `report.json`에는
원문, 모든 입력 메시지, 모델/코드 해시, 시드/샘플링 설정, 시간, 토큰 사용량,
프로세스 VRAM 샘플의 최대값이 저장됩니다. `state-N.json`은 생성에 사용한 상태입니다.
진행 중에는 `progress.json`이 갱신됩니다.

자동 검사는 정규화한 **정확한 태그 문자열**로 유지/목표 누락, 금지 태그 포함,
품질·연도 잔존, 검색 후보 이름의 혼합을 셉니다. 동의어까지 판정하지 않습니다.
일반 태그의 의미 모순, 새로 지어낸 이름, 그림으로 만들었을 때의 품질을 자동으로
판정하지 않습니다. 생성이 길이 제한으로 끝나면 완료 성공으로 세지 않습니다.

Inspect raw prompts for coherent appearance, clothing, setting changes and useful new
combinations. A literal adherence score is not a prompt-quality or character-recognition
score. Two seeds and a few cases establish feasibility only; they cannot prove a general
benefit from structured state. A state can work while being worse than a flat list.

전체 경험을 다시 구축할 필요는 없습니다. 검색 모델과 언어모델을 동시에 올릴 필요도
없습니다. **실제 추천·상태·결과·이미지·가중치는 개인 자료이며 코드 배포에 넣지 않습니다.**

## Gemma 4 tuning / Gemma 4 조정

For the language-only LoRA pilot, held-out instruction comparison and its limits, see
[Gemma 4 tuning](GEMMA4_TUNING.md). 한국어 지시와 참고 상태를 활용하도록 Gemma 4를
조정하는 방법입니다. 기존 TIPO GGUF에 이어 학습하는 것과는 구분합니다.
