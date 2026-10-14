# Gemma TIPO Native / ComfyUI 네이티브 실행

[한국어 README](README_KOR.md) · [English README](README.md)

태그형·자연어 지시문형 두 출력은 **[Gemma Prompt Expander Native](PROMPT_EXPANSION.md)**를
사용하세요. 같은 네이티브 모델을 쓰며 한국어·영어·일본어 사전 참고 입력을 지원합니다.
For context expansion into two renderings, see the new [Prompt Expander guide](PROMPT_EXPANSION.md).

모델 품질 개선용 평가·학습 입력 형식은
[TIPO 품질 평가 가이드](experiments/TIPO_QUALITY.md)를 참고하세요.
Evaluation and shared training-input instructions: [TIPO quality guide](experiments/TIPO_QUALITY.md).

## 한국어

`Gemma TIPO Native`는 ComfyUI의 `CLIP.tokenize/generate/decode`와 모델 관리로
Gemma 4 E2B를 실행합니다. 한국어·영어 요청, 태그, 선택한 이미지 한 장으로
이미지 생성용 영문 태그를 확장합니다. 노드가 모델이나 패키지를 자동 설치하지
않으며, 별도 llama.cpp 프로세스·GGUF·mmproj가 필요 없습니다.

### 설치와 연결

1. 이 저장소를 `ComfyUI/custom_nodes/RS-Seasonal-Prompt-Generator`에 설치합니다.
   계절 노드와 네이티브 노드는 추가 pip 설치가 없습니다. ComfyUI의 torch를
   교체하지 마세요. Gemma 4와 `CLIP.generate`를 지원하는 ComfyUI가 필요합니다.
2. ComfyUI 형식의 **Gemma 4 E2B 전체 safetensors**를
   `ComfyUI/models/text_encoders/`에 넣습니다. 파일 이름은 자유입니다.
3. ComfyUI를 재시작합니다. `Load CLIP`에서 모델 파일을 선택하고
   `type=stable_diffusion`, `device=default`로 설정합니다.
4. `Load CLIP`의 `CLIP` → `Gemma TIPO Native`의 `clip`을 연결합니다.
   선택적으로 `Load Image`의 `IMAGE` → `image`를 연결합니다.
5. `prompt`에 요청을 넣고 `tags` → `Preview as Text`의 `source`를 연결해 실행합니다.
   이미지 생성에는 이 `tags` 문자열을 해당 생성 모델의 텍스트 입력에 연결합니다.
   Gemma의 CLIP을 이미지 생성 모델의 CLIP으로 사용하지는 않습니다.

`type=stable_diffusion`은 현재 ComfyUI 로더가 체크포인트에서 Gemma 구조를
자동 판별하는 경로입니다. 일반 SD/SDXL CLIP 모델 파일을 선택하라는 뜻은 아닙니다.

### 이미 가지고 있는 TIPO 모델은?

| 보유 모델 | 네이티브 사용 |
| --- | --- |
| ComfyUI 형식의 Gemma 4 E2B safetensors | `Load CLIP`에서 직접 로드 |
| 튜닝 완료된 전체 HF Gemma 4 E2B safetensors + tokenizer/config | 아래 도구로 이름·포장을 변환. 학습된 비전 연결부도 함께 보존 |
| GGUF 텍스트 + mmproj만 있음 | 직접 로드 불가. 기존 GGUF 노드를 쓰거나 원래 전체 HF 체크포인트 확보 |
| LoRA/adapter만 있음 | 이 도구의 입력이 아님. 해당 원본 모델에 정상 병합한 전체 체크포인트 필요 |
| 일반 Gemma 4 E2B 원본 | 실행 가능. 기존 TIPO 튜닝을 가진 모델은 아니므로 품질을 따로 비교해야 함 |

기존 GGUF를 safetensors로 이름만 바꾸거나, 별개의 원본 모델에 튜닝된 mmproj를
붙이지 마세요. **네이티브 실행 전환 자체는 재학습이 필요 없으며**, 원래 TIPO
성능을 보존하려면 대응하는 학습 가중치와 토크나이저가 필요합니다.

### HF 전체 체크포인트 변환

소스 폴더에는 `config.json`, `tokenizer.json`, `model.safetensors` 또는
`model.safetensors.index.json`과 해당 shard 파일들이 있어야 합니다.
지원 구조는 Gemma 4 E2B의 텍스트 35층·hidden size 1536·어휘 262144입니다.
Python 표준 라이브러리만 사용하는 스트리밍 복사로, 양자화·다운로드·학습을 하지
않습니다. 텍스트·PLE·비전·오디오 가중치의 바이트와 정밀도를 그대로 보존합니다.
토크나이저를 결과 파일에 포함하며 원본과 기존 출력 파일을 덮어쓰지 않습니다.

ComfyUI 폴더에서, ComfyUI의 Python으로 실행하세요. 경로는 실제 경로로 바꿉니다.

```bash
python custom_nodes/RS-Seasonal-Prompt-Generator/scripts/prepare_gemma4_native.py \
  --source /path/to/full-hf-gemma4-e2b \
  --output models/text_encoders/gemma4-e2b-native.safetensors
```

Windows portable의 최상위 폴더에서:

```powershell
.\python_embeded\python.exe .\ComfyUI\custom_nodes\RS-Seasonal-Prompt-Generator\scripts\prepare_gemma4_native.py --source "D:\Models\full-hf-gemma4-e2b" --output ".\ComfyUI\models\text_encoders\gemma4-e2b-native.safetensors"
```

전체 BF16 원본으로 만든 파일은 약 **9.57GiB**이며 그만큼의 추가 디스크 공간이
필요합니다. 이는 종전 **PLE Q4 + 비전 FP16 GGUF**와 다른 실행 구성입니다.
네이티브 변환이 3GiB급 모델을 만드는 것은 아닙니다. 파일 크기와 실제 VRAM은
다르며, GPU 적재·오프로딩은 ComfyUI 설정과 다른 모델의 점유량에 따라 달라집니다.

### 입력·출력과 제한

- `prompt`: 한국어·영어 요청 또는 일부 태그. 이미지 없이도 실행됩니다.
  이미지가 있으면 비어 있어도 됩니다. 둘 다 비어 있으면 빈 문자열을 반환합니다.
- `image`: 선택 입력, ComfyUI RGB IMAGE 한 장 (`[1,H,W,3]`). 배치는 먼저 분리합니다.
- `tag_length`: 원하는 확장량. 모델의 실제 태그 개수를 보장하는 제한은 아닙니다.
- `max_tokens`: 출력 토큰 상한. 초기값 384; 길게 설명하거나 잘리면 조정합니다.
- `temperature`: 0은 greedy, 양수는 샘플링. `seed`는 ComfyUI에 전달합니다.
- `ban_tags`: 쉼표로 구분한 **영문 태그**. 요청에도 알려주고 `tags`에서 사후 제거합니다.
  한국어 문장의 모든 부정·변경 지시를 자동으로 강제하는 기능은 아닙니다.
- `tags`: 추론 문구·중복·금지 태그를 정리한 결과. 선택적인 kgen 정렬기는 사용하지 않습니다.
- `raw_output`: 모델 원문. 금지 태그가 포함될 수 있으므로 진단용으로 사용합니다.

예: `금발 소녀가 겨울 공원에서 코트를 입고 산책` + `ban_tags=scarf, masterpiece`.
작가·캐릭터·작품 보존을 모델에 요청하지만, 이름 식별·조건 준수·확장 품질을
보장하지는 않습니다. 이 노드에 VRS 검색이나 한국어 사전 조회가 자동 연결되지는
않습니다. VRS 참고 태그를 직접 `prompt`에 넣을 수 있습니다.

### 문제 해결

| 증상 | 확인할 것 |
| --- | --- |
| 노드가 검색되지 않음 | 설치 폴더 중첩과 ComfyUI 시작 로그, 재시작 여부 |
| 네이티브 생성이 없다는 오류 | `CLIP.generate`와 Gemma 4 지원이 들어간 ComfyUI 필요 |
| Gemma CLIP이 아니라는 오류 | 일반 CLIP/GGUF 대신 전체 네이티브 Gemma 4 E2B 파일 선택 |
| 텐서 누락/크기 불일치 | 모델 구조·토크나이저·비전 projector가 함께 있는 전체 체크포인트 확인 |
| 모델이 목록에 없음 | `models/text_encoders` 경로와 `.safetensors` 확장자, 새로고침/재시작 |
| GPU 메모리 부족·CPU 오프로딩 | `device=default`, 다른 모델 점유량과 ComfyUI 메모리 설정 확인 |
| 빈 응답/추론만 출력 | 출력 토큰 수와 해당 모델·채팅 템플릿의 대응 확인 |

취소·OOM·모델 오류는 ComfyUI로 전달하며, 이 노드는 CPU에서 몰래 재시도하거나
패키지 설치로 복구하지 않습니다. [API 그래프 예제](examples/gemma-tipo-native.api.json)는
`POST /prompt`의 요청 본문으로 사용합니다. 캔버스용 JSON과는 다른 API 형식입니다.
파일의 `clip_name`을 로컬 이름으로 바꾸세요. 이미지는 `LoadImage`와 선택 입력
`image` 연결을 추가하면 됩니다.

## English

Gemma TIPO Native uses ComfyUI's own PyTorch Gemma 4 runtime and model management
to expand Korean/English requests or a single optional image into prompt tags.
There are **no extra pip dependencies, automatic downloads, llama.cpp processes,
GGUF files or separate mmproj files** on this path.

### Setup

1. Install this repository under ComfyUI's `custom_nodes` and use a ComfyUI version
   with Gemma 4 and `CLIP.generate` support. Do not replace ComfyUI's torch.
2. Put a **ComfyUI-format full Gemma 4 E2B safetensors** in `models/text_encoders`.
3. Restart. Connect `Load CLIP` (`type=stable_diffusion`, `device=default`) to
   `Gemma TIPO Native`. The loader detects Gemma from its weights; do not select
   an ordinary SD/SDXL CLIP checkpoint.
4. Optionally connect `Load Image` to `image`. Enter a request and connect `tags`
   to `Preview as Text` or your image model's text input. Do not use Gemma CLIP as
   the image model's conditioning encoder.

For a full HF checkpoint, run `scripts/prepare_gemma4_native.py` using the commands
above with your paths. The standard-library converter supports unquantized E2B
(35 layers, hidden size 1536, vocabulary 262144), including sharded safetensors.
It renames tensors and embeds the tokenizer, preserving all language, PLE, vision
and audio payloads/precision. It does not download, train, quantize or overwrite
models. The full BF16 example is **9.57GiB**; reserve additional disk space.
This is different from the earlier PLE-Q4/FP16-vision GGUF configuration and does
not provide its size reduction. VRAM depends on ComfyUI loading/offload settings.

Existing GGUF/mmproj cannot be loaded directly. To retain TIPO tuning, obtain its
matching full HF weights and tokenizer, including its trained vision projector.
Adapters must first be merged into the correct base outside this tool. A generic
base Gemma can run here but does not thereby inherit TIPO fine-tuning. Runtime
conversion itself does not require training.

### Controls and behavior

`prompt` accepts Korean/English or tags; `image` accepts one RGB ComfyUI IMAGE
`[1,H,W,3]`. Either input is sufficient. `tag_length` is a soft target;
`max_tokens` caps generation. Temperature 0 uses greedy decoding; otherwise
sampling and `seed` are passed to ComfyUI.

`ban_tags` is a comma-separated **English tag** list, applied as a prompt hint
and post-generation filter. It does not parse all Korean exclusion instructions.
`tags` strips reasoning, duplicates and banned tags without invoking kgen.
`raw_output` is unfiltered diagnostic text and may contain banned tags. Names,
instruction adherence and expansion quality still require review. VRS/dictionary
lookup is not automatically connected; you may pass retrieved tags as the prompt.

If the model is absent from the loader, check `models/text_encoders` and restart
or refresh. Missing generation support requires a suitable ComfyUI version.
Wrong CLIP errors require a Gemma checkpoint, and tensor errors require matching
complete weights. Empty/reasoning-only replies require checking output length and
model/template pairing. For OOM/offloading, inspect other loaded models and
ComfyUI memory options. Cancellation and runtime errors propagate; this node does
not retry on CPU or install packages during execution.

The [API graph example](examples/gemma-tipo-native.api.json) is the request body
for `POST /prompt`, **not a canvas workflow export**. Change `clip_name` to your
local filename. Add `LoadImage` and connect `image` for image input. Legacy GGUF
nodes remain available with their separate `requirements-legacy.txt` dependencies.
