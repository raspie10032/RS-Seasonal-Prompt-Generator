# Gemma 4 TIPO 품질 개선: 평가부터 / Evaluation first

이 단계는 프롬프트 **확장 품질**을 비교할 입력과 채점 도구를 준비합니다.
새 학습이나 품질 개선을 완료했다는 의미가 아닙니다. 모델과 개인 자료는
저장소 밖에 두세요. ComfyUI 실행은 [네이티브 가이드](../NATIVE_TIPO.md)를 따릅니다.

현재의 **자연어 + 태그 이중 출력** 목표와 다음 학습 순서는
[맥락 확장 개발 계획](../PROMPT_EXPANSION.md#다음-개발-계획--2026-10-14)에 정리했습니다.
아래 태그 전용 실험 기록과 새 이중 출력 계획을 구분하세요. 실제 이미지 학습 경로는
이 문서 후반의 `Korean instructions with real image inputs`에 있으며,
새 `gemma4-context-expansion/1` 계약의 학습·평가 연결은 다음 구현 대상입니다.

## 평가 범위

| 유형 | 확인할 동작 |
|---|---|
| `partial_tags` | 지정 태그를 유지하고 어울리는 장면 요소 추가 |
| `korean` | 한국어 요청을 영문 태그로 옮기고 확장 |
| `edit` | 기존 프롬프트의 유지·변경·제외 조건 준수 |
| `image` | 참고 이미지의 특징을 활용해 요청한 새 장면 구성 |
| `vrs` | 출처가 있는 참고 후보에서 요청과 양립하는 요소 선택 |

개인 평가 초안은 유형별 40건, 총 200건입니다. 40개 원본 묶음에 5가지
요청을 연결하고 묶음 단위로 개발용 100건 / 최종 확인용 100건을 나눕니다.
이는 200명의 실제 사용자 요청이나 200개 독립 이미지가 아닙니다.
요청은 실패 유형을 참고해 새로 작성했으며, 원본 메타데이터는 생성 의도입니다.
작가·캐릭터·작품 이름은 요청에 명시한 경우 보존을 평가합니다.
이미지에서 이름을 알아맞히는 시험은 아닙니다.

동일 파일 해시, 정규화한 생성 의도, 원본 묶음, 64비트 dHash의 근접도를
검사합니다. dHash 해밍 거리 6 이하는 분리 전에 검토 대상으로 처리합니다.
이는 모든 크롭·편집·유사 장면의 중복을 배제했다는 보장이 아닙니다.
개인 자료의 캐릭터 편중과 원본 선택 조건은 개인 준비 보고서에 기록합니다.

VRS 문제에는 저장된 추천 결과의 해시와 snapshot을 기록합니다.
다른 요청에서 재활용한 창작 재료이며, 선택 이미지에 대해 새 검색을 수행한
결과인 것으로 표시하지 않습니다. 최종 확인 자료를 학습 예시로 사용하지 마세요.

## 입력 형식

[`tipo_prompt.py`](../tipo_prompt.py)의 `gemma4-tipo-native/1` 계약을
네이티브 노드와 새 텍스트 학습 자료가 공유합니다. 지시를 포함한 사용자
메시지 하나 뒤에 assistant 정답을 붙입니다. 학습 라벨은 사용자 입력 전체를
가리고 assistant 부분에만 둡니다. 토큰 길이 초과는 잘라 버리지 않고 실패합니다.

과거 파일의 system/user/assistant 형식은 재현용으로 보존합니다. 신규 자료에
`prompt_contract`를 명시해야 새 경로를 사용합니다. 예를 들면 다음과 같습니다.

```python
from tipo_prompt import VERSION, text_messages

inputs = {"prompt": "겨울 공원에서 걷는 장면. 목도리 제외.",
          "tag_length": "short", "ban_tags": "masterpiece, best quality, 8k"}
messages = text_messages(inputs, answer="winter, park, walking, coat, boots")
# 학습 자료에는 실제로 검토한 충분한 확장 답안과 원본 출처를 추가합니다.
```

`input.image`가 있는 학습 행은 텍스트 학습기에서 거부합니다. 이미지 processor,
이미지 토큰, 전처리 및 픽셀을 포함한 학습은 다음 단계에서 별도로 구현해야 합니다.
본문 지시를 공유하는 것만으로 HF와 ComfyUI의 비전 전처리나 출력이 같아지는
것은 아닙니다. 실제 토크나이저 및 실행 경로도 비교해야 합니다.

## 평가 자료 고정과 채점

모든 경로는 개인 저장 위치를 사용하세요. 아래 명령은 모델을 로드하지 않습니다.

```bash
python -m experiments.tipo_evaluation freeze \
  --draft /private/tipo/draft.jsonl --output /private/tipo/evaluation
python -m experiments.tipo_evaluation score \
  --cases /private/tipo/evaluation/dev.jsonl \
  --outputs /private/tipo/model-dev.jsonl --report /private/tipo/model-dev-score.json
```

`freeze`는 새 디렉터리에 `dev.jsonl`, `final.jsonl`, `manifest.json`을 만들며
기존 결과를 덮어쓰지 않습니다. JSONL 행 형식은
[`test_tipo_evaluation.py`](../tests/test_tipo_evaluation.py)의 최소 예제를 참고하세요.
평가 행에는 정답 생성 문장(`answer`)이나 학습용 `messages`를 넣지 않습니다.

모델 결과에는 원문을 반드시 기록합니다.

```json
{"id":"one","raw":"winter, coat, scarf","postprocessed":"winter, coat","truncated":false}
```

원문과 후처리 결과는 따로 채점합니다. 중복, 요구/금지 태그, 명시한 충돌 집합,
품질·연도 태그, 형식, 태그 개수와 생성 잘림을 검사합니다. 누락된 결과는
`complete: false`로 표시합니다. 반복 결과 ID와 모르는 ID는 오류입니다.

태그 판정은 정규화된 문자열과 **문제별 명시적 동의어** 기준입니다.
어휘가 다른 자연스러운 표현은 사람이 검토해야 하며, 이 점수는 의미 이해의
완전한 판정이 아닙니다. 새 태그 개수도 창작 품질 점수가 아닙니다.
`creative_quality`와 `visual_fidelity`는 자동으로 채우지 않습니다.

후보 이름을 가린 비교에서 다음을 별도로 검토하세요.

1. 지정한 이름·장면을 살렸는가.
2. 새 행동·의상·조명·구도가 서로 어울리는가.
3. 입력 복사나 비슷한 말 반복보다 유용한 구체성이 늘었는가.
4. 이미지 참고 문제에서 실제 보이는 특징을 적절히 활용했는가.

## 실행 중인 ComfyUI로 기준 출력 수집

네이티브 노드와 모델을 이미 로드할 수 있는 로컬 ComfyUI에서 실행합니다.
`--model-name`은 CLIPLoader에 표시되는 이름이며 `--model-file`은 서버가 그
이름으로 실제 선택하는 파일입니다. 폴더 설정/심볼릭 링크도 확인하세요.

```bash
python -m experiments.run_tipo_native_eval \
  --cases /private/tipo/evaluation/dev.jsonl \
  --output /private/tipo/native-baseline-v1 \
  --comfy-input /path/to/ComfyUI/input \
  --model-name gemma4-e2b-native-bf16.safetensors \
  --model-file /path/to/ComfyUI/models/text_encoders/gemma4-e2b-native-bf16.safetensors \
  --temperature 0 --max-tokens 384 --seed 42
```

이 도구는 개발용 자료만 받으며, 각 요청 전 큐가 비었는지 검사합니다.
다른 큐 작업을 발견하면 다음 요청을 제출하지 않습니다. 서버 재시작·모델 설치·
자동 취소·실패 요청 재제출은 하지 않습니다. 결과가 불명확한 시간 초과는
`*-submitted.json`의 prompt ID와 서버 history로 확인한 후 처리하세요.

원문과 후처리 출력, 개별 서버 history, 모델/입력 파일 해시, 설정, 요청별
경과 시간을 별도 기록합니다. 요청별 시간에는 HTTP 대기가 포함되고 전체
시간에는 이미지 준비와 파일 해시 계산도 포함됩니다. 순수 모델 추론 속도가
아닙니다. 현재 PreviewAny API는 생성 토큰과
종료 사유를 주지 않으므로 `truncated: null`로 기록합니다. 태그 조건 통과를
정상 종료나 창작 품질의 검증으로 해석하지 마세요.

이미지 문제는 메타데이터를 제거한 RGB 사본을 고유 input 하위 폴더에 준비합니다.
원본 파일은 수정하지 않습니다. 사본과 history도 개인 자료이므로 공개하지 마세요.
현재 RGB 변환은 일반적인 불투명 이미지용이며 투명 이미지의 배경 합성 조건은
별도로 고정해야 합니다. 텍스트·이미지 전처리의 학습/추론 비교도 별도입니다.

모든 조건 점수에는 정확 일치의 한계가 있습니다. `outdoor`/`outdoors`처럼
동의어가 문제에 등록되지 않았거나, 후처리가 `holding umbrella`를 남기고
`umbrella`를 제거하면 실패로 계산될 수 있습니다. 점수와 함께 원문을 검토하고,
동의어 정책을 고칠 때는 새 평가 버전으로 모든 비교 모델에 동일하게 적용하세요.
최종 확인 자료의 출력으로 학습 답안이나 동의어를 조정하지 마세요.

## 개인 텍스트 답안 고정

현재 모델의 원문을 그대로 정답으로 삼지 않습니다. 생성 메타데이터에서
사용자가 유지할 이름·외형을 선택하고, 요청에 맞는 행동·의상·소품·조명·구도를
작성합니다. `answer_origin`과 `review`에 작성 주체와 검토 범위를 남기세요.
메타데이터는 생성 의도이며, 이런 답안은 이미지의 실제 관찰 캡션이 아닙니다.
참고 후보를 직접 구성했다면 실제 VRS 검색 결과로 표기하지 마세요.

[`tipo_supervision.py`](tipo_supervision.py)는 답안을 생성하지 않고 작성된
자료를 검증·고정합니다. 최소 행 구조는
[`test_tipo_supervision.py`](../tests/test_tipo_supervision.py)를 참고하세요.
`schema: tipo-authored-supervision/1`, `purpose: training`, `split: train|eval`,
공통 `prompt_contract`, `input`, `answer`, `required`, `excluded`, `protected_names`,
`family`, `group_id`, 원본 provenance를 기록합니다. 학습 도중 검증용 `eval`은
최종 품질 평가의 `final`과 별개입니다.

```bash
python -m experiments.tipo_supervision \
  --draft /private/tipo/authored-draft.jsonl \
  --reserved-evaluation /private/tipo/evaluation/dev.jsonl /private/tipo/evaluation/final.jsonl \
  --output /private/tipo/supervision-v1
```

동일 원본의 변형과 같은 장면 레시피는 먼저 같은 split에 배치합니다.
검사는 train/eval 사이 및 개발·최종 평가와의 원본/의도/근접 dHash 중복을
막으며, 정답의 지정 요소·금지 태그·반복·충돌을 확인합니다. 출력 경로가 이미
있으면 덮어쓰지 않습니다. 학습기는 이 스키마의 파일 해시와 분리를 CUDA
초기화 전에 다시 확인합니다. 이미지 학습 행을 텍스트로 바꿔 읽지 않습니다.

자동 검사를 통과한 자료도 문장 의도·장면의 자연스러움·표현 다양성 검토가
필요합니다. 작은 장면 레시피에서 파생한 400행은 400개의 독립 장면이 아닙니다.
특정 성별·인원·캐릭터에 편중된 소규모 파일은 제한된 학습 실험에 사용하고,
일반적인 TIPO 품질이 입증된 자료로 취급하지 마세요.

## 다음 학습 단계

검토한 답안 300–500건으로 작은 학습을 시작하고, 결과를 확인한 뒤 늘립니다.
학습·학습 중 검증 자료에는 `source.image_sha256`, `intent_sha256`, `family_id`,
`dhash64`와 원본 묶음을 기록합니다. 네이티브 계약 학습은
`--reserved-evaluation /private/tipo/evaluation/dev.jsonl /private/tipo/evaluation/final.jsonl`
을 요구하며, 자료 출처가 없거나 평가와 겹치면 CUDA 초기화 전에 실패합니다.
기존 합성 자료를 이 출처 요건에 맞는 실제 자료로 가장하지 마세요.

개발용 평가와 학습 중 검증만으로 후보를 고른 다음 최종 확인용을 실행합니다.
과거 20건의 합성 조건 통과율 향상은 별도 3건 VRS 평가의 악화를 상쇄하지
않습니다. 학습 손실, 정답 태그 개수, 설치 성공만으로 품질 향상을 주장하지 않습니다.

## 소규모 학습과 네이티브 비교

개인 학습 환경에서 실행합니다. ComfyUI 큐와 GPU 여유를 먼저 확인하고, 실행 중인
다른 작업을 중단하지 마세요. 아래 예시는 320개 학습 행을 160 step × 누적 2회로
한 번씩 사용합니다. 자료 수가 다르면 step 수도 조정해야 합니다. 고정 난수로
매 epoch의 순서를 섞고 실제 사용한 행 인덱스를 run-config에 기록합니다.

```bash
python -m experiments.train_gemma4_tipo \
  --base /private/gemma4-full-hf --data /private/tipo/supervision-v1 \
  --output /private/tipo/pilot-001 --gpu-uuid GPU-YOUR-UUID \
  --steps 160 --grad-accum 2 --rank 8 --lr 0.0001 \
  --reserved-evaluation /private/tipo/evaluation/dev.jsonl /private/tipo/evaluation/final.jsonl
python -m experiments.merge_gemma4_tipo \
  --pilot /private/tipo/pilot-001 --output /private/tipo/merged-hf-001
python scripts/prepare_gemma4_native.py \
  --source /private/tipo/merged-hf-001 \
  --output /private/tipo/gemma4-tipo-pilot-001.safetensors
```

학습기는 언어 LoRA만 학습하며, 학습 프로세스에서 사용하지 않는 비전/오디오를
제거합니다. 원본 체크포인트는 수정하지 않습니다. 병합기는 학습 시점의 원본
해시를 확인한 뒤 **전체 원본**에 언어 LoRA를 합칩니다. PEFT가 저장한 짧은
`q_proj` 등의 이름은 비전에도 일치할 수 있으므로, run-config의 정확한 언어
모듈 경로를 복원합니다. 변경된 모든 텐서를 기록하고, 비전·PLE를 포함한 대상
이외 텐서는 바이트 단위로 같아야 성공합니다. CPU 작업은 2개 스레드로 제한합니다.

학습은 NF4 기반, 병합 결과는 원본 BF16 기반이므로 두 실행의 출력은 같다고
가정하지 않습니다. 새 파일을 ComfyUI의 text_encoders에서 고유 이름으로 선택해
동일한 개발 평가를 실행하세요. 기존 모델은 보존합니다. 학습 검증 20건의 점수와
ComfyUI 개발 100건의 점수는 별개이며, 최종 확인 자료는 후보 선택 전 사용하지
않습니다. 이 단계는 텍스트 추가 학습입니다. 이미지 입력의 출력 변화는 언어
가중치 변경의 효과이고 비전 추가 학습의 증거가 아닙니다.

### 첫 개발 실험의 범위와 한계

320행을 한 번씩 사용한 텍스트 LoRA 후보를 전체 BF16 원본에 병합하고,
동일한 설정과 개발 100건으로 네이티브 실행을 비교했습니다. 기존 BF16 베이스와
후보의 원문 조건 통과는 15→44건, 후처리 후에는 18→52건이었습니다. 원문에서
중복이 검출된 출력은 47→17건이었습니다. 이는 고정한 태그 조건 검사 결과이며,
전반적인 프롬프트 품질이나 사람의 선호 점수가 아닙니다. 기존 GGUF TIPO와의
비교도 아닙니다. 최종 확인용 100건은 실행하지 않았습니다.

아직 기본 모델로 채택하지 않습니다. 일부 수정 요청에서는 이전의 충돌 장면을
지웠지만 `holding cup`을 `warm cup`으로 바꿔 행동 조건을 놓쳤고, 긴 태그 묶음을
반복하는 답도 남았습니다. 자동 검사가 통과시킨 이미지 답에 `kneeling`과
`standing`이 함께 나왔습니다. **등록된 충돌이 0건이라는 결과는 모순이 없다는
뜻이 아닙니다.** 기존 채점 기록은 보존하고, 다음 평가 규칙에는 이런 누락을
반영한 뒤 양쪽 출력을 같은 규칙으로 재채점해야 합니다. 이미지 충실도와
창작 품질은 별도 검토가 필요합니다. 개인 원문과 체크포인트는 공개하지 않습니다.

## 실패 유형을 반영한 애니·게임 그래픽 학습자료 보강

`experiments.tipo_corpus`는 개인 생성 이미지 카탈로그에서 이름·머리색·눈색을
가져와 **새로 작성한 장면 답안**에 결합하는 텍스트 학습자료 준비 도구입니다.
학습이나 다운로드를 실행하지 않습니다. 원본 프롬프트는 생성 의도이며 픽셀
정답이 아닙니다. 실제 이미지 입력 학습과 3D 이미지 캡션 수집은 별도 작업입니다.

원래 positive 프롬프트를 다시 해석해 음수 가중치로 억제한 태그를 제외합니다.
사진·실사 지향 태그가 있으면 제외하며, 스타일 근거가 불분명한 카탈로그 항목도
선택하지 않습니다. NovelAI 생성 정보는 일러스트 의도의 간접 근거로 기록하고
`pixels-unreviewed`를 붙입니다. 이 분류는 픽셀 전수 검사가 아닙니다. 카메라를
들거나 사진을 찍는 **캐릭터의 행동**은 사진풍과 구분합니다. 작가·캐릭터·작품명은
요청에 명시하고 답안에서 보존하며, 원본 의상·행동·신체 묘사는 가져오지 않습니다.

장면 레시피는 개인 JSONL로 작성합니다. 다음은 가상의 형식 예시 한 행입니다.
실제로는 서로 다른 장면을 여러 개 작성해 학습·검증 가족을 분리해야 합니다.

```json
{"family":"example_cup","subject":"person","ko":"실내에 서서 컵을 손에 들게. 앉지 마.","core":["indoors","standing","holding cup","cup"],"actions":["holding cup"],"exclude":["sitting","kneeling"],"old":["beach","summer"],"details":["shirt","pants","coat","window","table","steam","curtains","lamp"],"variants":[["upper body","from side","warm colors"],["full body","looking down","shadow"]]}
```

`actions`는 `core`에 들어가야 합니다. 레시피의 모든 정답 태그를 사용자가
제공한 단보루 사전과 대조하며, 미등록 행동을 조용히 삭제하지 않습니다.
예를 들어 악기 태그와 `playing instrument`를 함께 써서 행동을 남길 수 있습니다.
레시피의 정답과 금지·삭제 요소가 충돌하면 중단합니다. `subject`는 `person`과
`scenery`를 지원하지만, 실제로 확보하지 못한 자료 유형을 있다고 간주하지 않습니다.

```bash
python -m experiments.tipo_corpus \
  --catalog /private/generated/catalog.sqlite3 \
  --reserved-evaluation /private/tipo/evaluation/dev.jsonl /private/tipo/evaluation/final.jsonl \
  --previous-supervision /private/tipo/supervision-v1/train.jsonl /private/tipo/supervision-v1/eval.jsonl \
  --recipes /private/tipo/recipes.jsonl \
  --dictionary /private/tags/danbooru-tags.csv \
  --output /private/tipo/curriculum-v2
```

카탈로그는 읽기 전용으로 열고 원본 바이트·생성 의도·dHash를 검사합니다.
평가 자료와 이전 학습자료를 먼저 제외하고, 원본·장면 가족의 학습/검증 분할을
정한 뒤 부분 태그, 한국어, 장면 수정, 참고 후보 선택, 초안 교정, 행동 유지의
여섯 형식을 만듭니다. 참고 후보는 **작성한 방해 후보**이며 VRS에서 검색한 결과가
아닙니다. 같은 원본에서 파생한 여섯 행을 독립된 여섯 장면으로 세지 마세요.

결과의 `selection.json`은 후보 수집, `vocabulary.json`은 레시피 어휘 검사를
기록하고, `supervision-v2/manifest.json`은 실제 학습/검증 행 수와 SHA-256을
고정합니다. 필요한 원본이 부족하면 후보 기록만 저장하고 학습자료 동결은 중단합니다.
레시피 수나 목표 비율을 수정한 뒤 `--catalog` 대신
`--selected-sources /private/previous/sources.json`으로 후보를 재사용할 수 있습니다.
원본 바이트와 실사 의도 제외 조건을 다시 확인하며 기존 출력 폴더는 덮어쓰지 않습니다.

첫 보강본은 384개 원본의 생성 메타데이터와 64개 작성 장면으로 2,304행을
구성했습니다. 학습 1,872행, 학습 중 검증 432행이며 서로 다른 답안은 384개입니다.
2D 애니 일러스트 요청 1,728행, 셀 셰이딩 3D 게임 그래픽 요청 576행입니다.
3D 행은 **새 프롬프트의 요청 스타일**이고 실제 3D 원본 캡션이라는 뜻이 아닙니다.
원본은 여성 352개·남성 32개로 여전히 여성에 편중되어 있으며 배경 전용 원본은
포함되지 않았습니다. 답안의 일반 태그는 전부 로컬 사전에 있었고, 작가·캐릭터·작품명은
요청에 명시된 것을 모두 보존했습니다. 원본 이름 전체는 출처에 보존하고,
작가 지정 없음·소수·전체 지정을 섞어 특정 작가 조합을 기본값으로 학습하지 않도록 했습니다.
단일 인물 중심의 제한적인 추가 학습자료입니다.

답안 2,304행의 실제 HF/ComfyUI 입력 토큰열 일치와 답변 전용 label 마스킹을
CPU로 검사했습니다. 평가 성능 향상과 이미지 인식 개선은 아직 검증하지 않았습니다.
선호 검토와 다음 학습·개발 평가가 필요하며 최종 평가 원본은 계속 예약합니다.
개인 카탈로그·레시피·JSONL·사전·검토 기록은 저장소 밖에 보관하세요.

## 두 번째 텍스트 LoRA 결과와 채택 제한

보강 자료의 학습 1,872행을 한 번씩 사용해 같은 원본 HF 모델에서 별도 LoRA를
학습했습니다. 첫 후보에 이어서 학습한 결과가 아닙니다. rank 8, learning rate
`1e-4`, gradient accumulation 2, 최대 길이 768, 총 936 optimizer step을
사용했습니다. 저장/재로딩과 소규모 사전·사후 평가를 포함한 스크립트 실행은
약 472초, 학습 중 PyTorch 최대 할당 메모리는 약 7.2GiB였습니다.
언어 가중치 190개만 변경했고 비전·PLE 등 비대상 가중치 1,762개는 원본과
바이트 단위로 동일했습니다. 전체 모델 병합 후 BF16 네이티브로 평가했습니다.

고정 개발 100문제를 같은 `max_tokens=384`, `temperature=0`, `seed=42`로
비교했습니다. 원래 기준의 원문/후처리 통과는 베이스 15/18건, 첫 후보 44/52건,
두 번째 후보 50/54건입니다. 첫 후보에서 발견한 자세 모순을 보완하기 위해
**단일 인물인 이 개발 문제에 한정해서** `standing/sitting/kneeling/lying` 중
둘 이상이 있으면 충돌로 기록하는 `posture-audit/1`을 세 모델의 저장된 출력에
동일하게 적용했습니다. 원본 문제와 과거 점수는 변경하지 않았습니다.

| 자세 검사 보완 후 결과 | 베이스 | 첫 후보 | 두 번째 후보 |
|---|---:|---:|---:|
| 원문 조건 전체 통과 | 15/100 | 42/100 | 50/100 |
| 후처리 후 조건 전체 통과 | 18/100 | 50/100 | 54/100 |
| 이름 요구사항 보존 | 226/245 | 244/245 | 236/245 |
| 원문 반복 포함 | 47/100 | 17/100 | 18/100 |
| 원문 제외 태그 재등장 | 7/100 | 5/100 | 10/100 |

두 번째 후보의 원문 유형별 통과는 부분 태그 17/20, 한국어 5/20, 수정 8/20,
이미지 3/20, VRS 참고 17/20입니다. 첫 후보의 같은 보완 기준은 각각
11/20, 5/20, 5/20, 4/20, 17/20입니다. 첫 후보 대비 통과로 바뀐 문제는
23개, 실패로 바뀐 문제는 15개였습니다. 이름 누락 9건은 출력 5개에 걸친
태그 요구사항 수이며, 인물 식별 정확도가 아닙니다. 괄호가 포함된 이름을
여러 태그로 쪼개거나 작품명을 생략하는 사례를 출력에서 확인했습니다.

**두 번째 후보는 기본 모델로 채택하지 않습니다.** 전체 조건 점수는 올랐지만
사용자가 중시하는 이름 보존과 이미지 입력이 후퇴했고 반복·제외 지시도
해결되지 않았습니다. 두 후보와 원본을 모두 보존합니다. 24건의 학습 중 검증에서
조건 통과가 4→15건으로 늘어난 결과만으로 네이티브 품질을 판단하지 마세요.
다음 실험은 기존 자료 혼합, 이름 원형 보존·한국어 제외 지시 보강, 학습률/학습량
비교가 필요합니다. 이는 검증할 가설이며 원인이 확정된 것은 아닙니다.

이번 학습은 텍스트 전용입니다. 창작 선호·이미지 충실도를 채점한 결과가 아니며,
동의어와 포함 관계를 완전히 처리하지 못하는 조건 검사 한계도 남습니다.
최종 확인용 100문제는 실행하지 않았습니다. 개인 출력·가중치·비교 기록은
저장소 밖에 보관하고, 기존 워크플로와 기본 모델을 자동 교체하지 마세요.

## English

This workflow includes evaluation preparation and a first experimental text LoRA;
it does not establish a quality-approved TIPO model.
The private draft has 200 authored requests derived from 40 source groups, covering
partial tags, Korean, editing, images and VRS references. Split groups before deriving
requests. Generation metadata describes intent, not verified pixels. Identity names
are explicit user requirements; this does not measure recognition accuracy.

Native inference and new text supervision share `tipo_prompt.py`. Historical pilots
retain their original template. Text training rejects image rows; multimodal training
and preprocessing parity still require separate implementation and verification.

Freeze versioned private JSONL with the CLI above. Keep raw and postprocessed model
outputs separate. Scores cover only annotated exact/alias constraints, duplication,
conflicts and format. Novel-tag counts do not establish creativity. Human preference
and reference-image review remain required. Source and dHash checks are bounded
duplicate screens, not proof against every crop or related image. Keep private prompts,
images, dictionaries, retrieval results and model artifacts outside Git.

`experiments.run_tipo_native_eval` records development cases through a local running
ComfyUI server, one request at a time while its queue is empty. It preserves raw and
postprocessed text and prompt histories. It does not restart the server or retry
an unknown outcome. Confirm the loader's name-to-file mapping. PreviewAny does not
expose stop reasons, so termination is unknown; request wall time is not pure GPU
inference time. Staged RGB images and histories remain private artifacts.

`experiments.tipo_supervision` validates and freezes authored text examples, with
source/scene separation, reserved dev/final sources, answer constraints and hashes.
It does not write creative answers or verify their visual fidelity. Name/appearance
metadata is generation intent. Mark constructed reference candidates as constructed,
not actual VRS retrieval. Recipe-derived variants are not independent scenes. Check
token lengths and assistant masks with the actual tokenizer before a training pilot;
image supervision still needs its separate multimodal path.

The pilot shuffles complete epochs reproducibly and records actual row coverage.
Merge only into the hash-matched full HF base, restoring exact language-only
adapter target paths. The merge audit checks all non-target tensors byte for byte,
including vision and PLE. NF4 training and BF16 native deployment are different
runtimes: compare the native development outputs separately. Preserve the original
model and reserve final cases until candidate selection. Text-only tuning does not
establish visual learning or improved human preference.

The first 320-row, one-epoch pilot improved fixed native development constraints
from 15/100 to 44/100 on raw output and from 18/100 to 52/100 after postprocessing.
Outputs with detected duplicates fell from 47 to 17. These are comparisons against
the native BF16 base, not the older TIPO GGUF. Final cases remain unused. Keep the
candidate experimental: a required action can become only an object, repeated tag
blocks remain, and a passing image answer contained both `kneeling` and `standing`.
The frozen conflict list missed that contradiction. Zero registered conflicts does
not prove consistency. Preserve historical scores and rescore both outputs under a
versioned expanded rubric before making further claims. Visual fidelity and creative
quality still need separate review; private source outputs and weights stay outside Git.

`experiments.tipo_corpus` prepares a local, text-only illustration curriculum from
positive generation metadata and privately authored scene recipes. It reparses
negative weights, excludes photographic intent, records unresolved pixel verification,
preserves requested identities, checks recipe vocabulary against a user-supplied
Danbooru dictionary, and rejects conflicting targets. Camera actions are not photo
styles. Split source/scene families before producing six instruction variants, reserve
development/final and previous training sources, and keep all corpus artifacts private.
The CLI above can reuse a private `--selected-sources` cache instead of rescanning a
catalog; source bytes and photographic-intent exclusions are rechecked. Existing outputs
are never overwritten. Source shortages save a selection report without freezing a corpus.

The first supplemental corpus contains 2,304 rows (1,872 train / 432 validation),
384 metadata sources, 64 authored scenes and 384 unique answers. Of these rows, 1,728
request 2D illustration and 576 request cel-shaded 3D game graphics. These are target
prompt styles, not verified 3D image captions. Sources remain female-heavy (352 female,
32 male; no scenery-only sources). All requested artist/character/series tags are
preserved; general answer tags occur in the local dictionary. Full source identities
remain in provenance while instructions vary between no artist, a subset, and all
source artists, to avoid teaching a frequent artist stack as a universal default. Constructed reference
candidates are not actual VRS retrieval. Native/HF input token parity and assistant-only
labels were checked for all rows on CPU; this does not establish improved model quality,
visual learning, or human preference.

A second text LoRA trained once over all 1,872 training rows from the same original
HF base, not from the first adapter (rank 8, learning rate 1e-4, accumulation 2,
length 768, 936 steps). The script including evaluation and adapter reload took about
472 seconds; peak PyTorch training allocation was about 7.2 GiB. Only 190 language
tensors changed; 1,762 non-target tensors, including vision and PLE, remained byte-identical.

On the identical 100 native development inputs and generation settings, historical
raw/postprocessed passes are 15/18 for the base, 44/52 for pilot 001, and 50/54 for
pilot 002. A supplemental posture-audit/1 flags multiple standing/sitting/kneeling/lying
tags only for these single-subject cases. Applied equally to all saved outputs, it gives
15/18, 42/50, and 50/54. Frozen cases and historical scores remain unchanged.
Pilot 002 raw passes by task are partial tags 17/20, Korean 5/20, editing 8/20,
images 3/20, and VRS references 17/20; pilot 001 under the same audit has 11, 5, 5,
4, and 17 respectively. Twenty-three cases improve and fifteen regress versus pilot 001.

Do not promote pilot 002 as the default: required-name preservation drops from
244/245 to 236/245, raw forbidden-tag outputs rise from 5 to 10, and repeated-tag
outputs rise from 17 to 18. The nine missing name requirements occur across five
outputs; these are explicit-name preservation checks, not recognition accuracy.
Some parenthesized names split into separate tags or lose their series qualifier.
Both candidates and the original remain available. Mixing the earlier corpus,
strengthening exact-name and Korean exclusion examples, and comparing learning
rates/training duration are next hypotheses, not established explanations.
The 24-case training-validation increase from 4 to 15 passes does not establish
native quality. This remains text-only tuning with no preference or visual-fidelity
claim. Final evaluation cases remain unused and all private artifacts stay outside Git.

## Korean instructions with real image inputs

Use `--multimodal` to train the language adapter while retaining a frozen image
encoder/projector. The default remains text-only and rejects image rows. Images
must have absolute local paths, SHA-256 provenance and hash-bound visual review
(`visual_review.image_sha256`, `reviewer`, `style` equal to `2d` or `3d`, and
`observed_tags`). `visual_required` lists only reviewed visible properties to
retain, and must also be in `required`. Use an explicitly prepared, hashed RGB
composite for transparent images. Do not label original generation intent as
visible truth. Never cache an image as text-only `messages`.

Freeze these rows with `freeze_supervision(..., allow_images=True)` after splitting
source and scene families. Preserve both reserved development and final inputs.
Korean requests should express inclusion, exclusion and scene changes themselves;
leaving `ban_tags` empty prevents an English ban list from answering the Korean task.
Mix paired include/exclude cases and image requests with and without explicit names.
For image requests, derive appearance requirements from reviewed pixels rather
than placing an appearance caption in the request. An answer may describe the new
requested scene; it is not necessarily a caption of the reference image.

```bash
python -m experiments.train_gemma4_tipo \
  --base "$BASE" --data "$PRIVATE_DATA" --output "$NEW_RUN" \
  --gpu-uuid "$GPU_UUID" --multimodal \
  --initial-adapter "$PREVIOUS_RUN/adapter" \
  --steps 300 --grad-accum 2 --rank 8 --lr 0.00003 \
  --max-length 1024 --eval-count 24 --max-new-tokens 384 \
  --reserved-evaluation "$EVAL/dev.jsonl" "$EVAL/final.jsonl"
```

This example processes 600 rows once when the training split contains 600 rows.
The optional initial adapter must come from a completed, reload-verified pilot
against the same hashed base. This starts a new optimizer, not an exact optimizer
resume. The selected training-validation subset balances image and text examples.
The existing training environment needs matching Gemma 4 Transformers, PEFT,
bitsandbytes, torchvision and Pillow; these are training dependencies, not required
ComfyUI-node installation dependencies.

`tipo_multimodal.py` matches native media placement, including trailing newlines,
and ComfyUI's uint8 bicubic resize before the HF processor patchifies pixels.
Record this input contract, package versions, code hashes and processor settings.
Compare token sequences, actual image patch values/positions and assistant-only
labels against the installed ComfyUI tokenizer before training. Token equality
alone is insufficient. The training run records actual vision-tower calls, input
pixel tensors and processed image-row count; only language projections are
trainable. Vision weights are FP16 and frozen in training; training operations use
BF16 autocast. A merge retains the original checkpoint's vision precision, which
is BF16 for the existing native base. This does not produce an FP16 native vision
release automatically.

The first focused curriculum has 696 rows: 384 Korean text tasks and 312 actual
image tasks; 600 train and 96 validation. Image tasks use 30 previously downloaded
2D game illustrations, split into 24 training sources and six validation sources.
These are repeated authored variations, not 312 independent images. Their prior
VRS retrieval tests cannot serve as independent TIPO validation after reuse.
All answers use dictionary-checked ordinary tags and retain requested names.
Keep image files, visual annotations, prompts, local dictionary and weights private.
Reserve the original final set; evaluate native Korean/image performance and exact
name preservation before considering adoption. Automatic tag constraints do not
measure naturalness, creative usefulness, or general visual accuracy.

Pilot 003 continues pilot 001 (not pilot 002) at learning rate 3e-5 for 300
optimizer steps, accumulation two: 312 Korean text rows and 288 image/Korean
rows, each seen once. Its frozen vision tower was actually called for all 288
image training rows. The script, including before/after checks and adapter reload,
took about 293 seconds; peak PyTorch allocation was 7.19 GiB. Input tokens and
assistant masks match native execution for all 696 rows, preprocessed pixels for
all 312 image rows, and native patch values/positions for all 30 image sources.
Only 190 language tensors change after merging; all 1,762 non-target tensors are
byte-identical. These checks do not make NF4 training numerically identical to
native BF16 inference.

On the original native development set with the same posture audit, pilot 003
has 56/100 raw passes and 60/100 postprocessed passes. Task counts are partial
tags 13/20, Korean 8/20, editing 8/20, images 8/20, and VRS 19/20. Compared with
pilot 002, Korean improves from 5 and images from 3; partial tags regress from 17.
Exact-name hits recover from 236/245 to 244/245, but ten outputs still contain
forbidden tags and nineteen contain duplicates. This is not a general quality score.

A separate focused validation subset has 24 Korean text requests and 24 image
requests over six held-out illustrations; the named and unnamed requests reuse
those same six sources. It is drawn from training validation, not a new final
benchmark. Pilot 001 versus pilot 003 gives 0/48 versus 8/48 complete raw passes
(image 0/24 to 7/24; Korean 0/24 to 1/24). Required visual-property hits rise from
25/72 to 60/72; unnamed image requests rise from 15/36 to 30/36. Do not describe
this as 83% image accuracy: only three annotated coarse properties per request
are counted, repeated sources are correlated, and false added properties are not
included in this recall measure. Exact spelling and required style tags can also
make otherwise useful requests fail. Name preservation on this subset regresses
from 159/159 to 151/159, including incorrectly grouped parenthesized names.

A deterministic image-swap control changes only the image for twelve unnamed
requests, preserving the Korean instructions and seed. Both models change all
twelve outputs. Pilot 003 retains 18/24 source-distinguishing properties with the
correct image and 0/24 after swapping; it follows 18/24 properties of the new image.
Pilot 001 has 7/24, 2/24, and 8/24 respectively. This supports image dependence
and better use of the annotated features on these six sources; it does not prove
broad visual understanding. A horned reference still produces invented cat ears
and a wolf tail. Preserve this failure alongside successful hair-color swaps.

Keep pilot 003 experimental. Korean generalization, exact-name preservation on
new phrasing, visual hallucinations, and creative usefulness remain unresolved.
The original final evaluation remains unused; private outputs, review annotations,
control results, comparison hashes, and weights stay outside the repository.

The focused Korean 1/24 complete-pass result is not a Korean comprehension
score. A follow-up audit found zero exact forbidden-tag occurrences and zero
listed posture conflicts across those 24 outputs. Three fail only because
`lineart` is missing, although the request asks for a 2D anime illustration
without specifying line art. Other misses include tag spelling, unrequested
props, and genuine action/name errors. Preserve the original strict scores;
report explicitly requested behavior separately before using this rubric to
decide what language training is needed.

An exploratory paired-language probe translated the first eight focused Korean
requests into English. It reused saved Korean outputs and ran English requests
through the identical native pilot 003 file hash with temperature zero, seed 42,
and max tokens 384. Before English inference, the diagnostic excluded style-tag
spelling and an unrequested rolling pin from required tags. Remaining non-name
exact-tag hits were Korean 68/75 versus English 63/75; exact name hits were 33/39
versus 37/39; outputs with duplicates were 3/8 versus 1/8. Neither language
produced the listed forbidden tags or posture conflicts; original strict passes
were zero in both. These are not semantic-accuracy rates: synonyms still count
as misses, English translations can resolve Korean ambiguities, and eight
conveniently selected validation cases do not establish general language ability.
This probe does not support treating English translation alone as the solution.
Keep its requests, outputs, model/input hashes and protocol in private artifacts.
