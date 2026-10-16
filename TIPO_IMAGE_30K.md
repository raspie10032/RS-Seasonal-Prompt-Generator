# Gemma TIPO Image 30K — release and limitations / 배포 안내

[한국어 README](README_KOR.md) · [English README](README.md) · [Q4 installation](GGUF_NATIVE.md)

## 한국어

**이미지에서 프롬프트에 쓸 태그를 얻기 위한 실험용 LoRA입니다. 결과가 취향에 맞지
않거나 이미지·지시와 다를 수 있습니다.** 자동 정답 태거, 캐릭터/작가 식별기,
완성된 한국어 편집 모델로 보장하지 않습니다. 생성 전에 결과를 확인하고 수정하세요.

- 배포 모델: [raspie/gemma4-tipo-lora-image-30k](https://huggingface.co/raspie/gemma4-tipo-lora-image-30k)
- 선택 이름: **`tipo_lora=image-30k`**, 약 **45.3MiB**. 처음 선택하여 실행할 때 다운로드합니다.
- 학습: 이미지 기본 사례 **30,000장 + 보조 3,000행**, 총 33,000행, 1회 학습.
  언어 부분 rank 8 / alpha 16 LoRA이며 비전·PLE·토큰 임베딩은 고정했습니다.
- 기반: `p-e-w/gemma-4-E2B-it-heretic-ara`의 고정 판본
  `c9a1d4c031981f14d86eeb0c7d87de7fafd34513`. Google 원본의 Heretic ARA 파생 모델입니다.
- 애니·일러스트 및 스타일화한 그래픽을 위한 이미지 태그 보강에 초점을 둡니다.
  자연어는 원본 Gemma에 맡기는 설정을 권장합니다.

### 설치와 권장 설정

1. 저장소를 업데이트하고 **ComfyUI의 Python**으로 `requirements.txt`를 설치합니다.
   기존 큐를 마친 뒤 ComfyUI를 재시작합니다. [Windows/Linux 명령](GGUF_NATIVE.md).
2. `Gemma Prompt Expander Native`에서 **`model_source=heretic_ara_q4`**를 선택하고
   `clip`을 연결하지 않습니다. 파일 경로는 비워두면 자동으로 다운로드합니다.
3. **`tipo_lora=image-30k`**를 선택합니다. 별도 LoRA 로더는 필요 없습니다.
4. 우선 **`output_format=tags`**로 이미지에 관련된 태그를 확인하세요.
   자연어도 원하면 `output_format=both`, **`tipo_lora_scope=tags_only`**를 선택합니다.
   이 경우 원본 Gemma가 장면을 문장으로 쓰고, LoRA가 그 장면을 태그로 표현합니다.
5. 평가 시작 설정은 `temperature=0.5`, `repetition_penalty=1.15`,
   `max_tokens=512`, `seed=42`입니다. 취향에 맞는 결과를 보장하는 설정은 아닙니다.

기존 워크플로의 기본값 `tipo_lora=none`, `model_source=connected_clip`은 유지합니다.
`tipo_lora_scope`도 기존 기본값 `both`를 유지하므로 위 설정을 직접 선택하세요.
`Gemma TIPO Native`의 태그 전용 노드에서도 `image-30k`를 선택할 수 있습니다.
이미 병합한 모델 위에 같은 LoRA를 중복 적용하지 마세요. 시험하려면
[이미지 태그 API 예제](examples/gemma-image-30k.api.json)를 사용합니다.
API 요청 JSON이며 캔버스 워크플로 파일은 아닙니다. `LoadImage`의 파일명을 바꾸세요.

### 실행 사양과 부담

**가벼운 CPU용 플러그인이 아닙니다. LoRA 파일이 작아도 Gemma 전체 추론이 필요합니다.**

| 항목 | 안내 |
| --- | --- |
| GPU | 검증 장비는 RTX 4070 Ti SUPER / RTX 5060 Ti, 각각 16GB VRAM. 여유 있게 쓰려면 이 급의 NVIDIA CUDA GPU를 권장합니다. 최소 사양을 검증한 것은 아닙니다. |
| RAM | 32GB 이상을 여유 있는 운영 기준으로 권장합니다. 측정한 최소 요구량은 아닙니다. 다른 생성 모델·VRS와 함께 쓰면 더 필요할 수 있습니다. |
| Q4 다운로드 | 텍스트+토크나이저 약 3.22GiB, 이미지 사용 시 비전 포함 약 4.14GiB. LoRA 약 45.3MiB 추가. 디스크 크기이며 VRAM 요구량이 아닙니다. |
| 대체 베이스 | 연결형 BF16 네이티브 모델은 약 9.57GiB. Q4 경로에서는 별도로 받을 필요가 없습니다. |
| 시간 | Linux / 5060 Ti의 24장 태그 평가에서 노드 호출 시간 중앙값 약 36초, 범위 약 16~126초. 출력 길이와 적재·다른 작업에 따라 달라집니다. 속도 보장이 아닙니다. |
| 두 출력 | 자연어 → 태그 두 번 생성하므로 태그만 생성할 때보다 작업이 늘어납니다. |

GPU 메모리는 압축 가중치 외에도 복원 버퍼·이미지 처리·KV 캐시·LoRA가 필요합니다.
이미지 생성 모델과 동시에 올리면 OOM이나 CPU 오프로딩으로 인한 지연이 생길 수 있습니다.
한 이미지·짧은 출력부터 시작하고, 메모리가 부족하면 다른 모델을 해제하거나
출력 길이를 줄이세요. CPU 전용·8GB GPU·AMD/Apple·Windows 실행은 이번 배포의
실기 검증 범위가 아닙니다. 설치 명령 제공과 해당 환경의 실행 검증은 구분합니다.

### 실제 결과와 남은 한계

ComfyUI Q4 개발 이미지 **24장**의 동일 조건 비교입니다. 이미지와 사용자가 준 이름을
입력했습니다. 이름을 이미지에서 알아맞힌 시험은 아닙니다.

| 항목 | 원본 | 이전 3천 장 비교 후보 | 이번 3만 장 |
| --- | ---: | ---: | ---: |
| PixAI 태그와의 평균 F1 | 0.141 | 0.271 | **0.352** |
| 입력 이름 누락 / 93회 | 10 | 17 | **2** |
| 빈 출력 / 24장 | 0 | 0 | 0 |

**F1은 PixAI가 만든 태그와의 일치 지표이며 35.2%의 인간 평가 정확도를 뜻하지 않습니다.**
이전 3천 장 비교 후보는 별도 내부 실험이며 공개 `pilot-001/002`와 다른 모델입니다.
최종 분리 평가 자료는 사용하지 않았고, 제한된 개발 자료의 결과만 공개합니다.

한국어 중심 요청 **12건**에서는 필수 태그 누락이 이전 3천 장 17개 → 이번 18개,
지정 이름 누락은 0개 → 3개였습니다. VRS 참고를 준 **6건**에서는 이름 누락 3개와
충돌 태그를 내는 요청 2건이 남았습니다. 단어 대조는 동의어도 누락으로 셀 수 있습니다.

실제 출력에는 **짧은 머리를 긴 머리로 변경, 여성 한 명에 남성 태그 추가,
지정한 작가·캐릭터·작품명 누락**이 있었습니다. VRS를 연결해도 참고 이미지의 특징이
섞일 수 있습니다. `ban_tags`의 정확한 문자열 제거만으로 의미상 충돌이 해결되지는 않습니다.
학습이 성인 표현을 포함하는 태그 영역을 다루므로 원치 않는 태그가 나올 수도 있습니다.

두 출력 비교에서는 **LoRA를 태그 단계에만 적용**했습니다. 자연어 개선은 입증하지
않으며 원본 Gemma의 잘못된 이미지 해석이 태그로 이어질 수 있습니다. 이미지 태그가
목적이면 우선 태그 단독 모드를 비교하세요. 단보루 중심 자료로 학습했으므로 e621 전체
어휘·다국어 번역 완전성도 보장하지 않습니다.

개인 이미지·원본 태그·프롬프트·VRS 경험·평가 원문은 공개하지 않습니다. HF에는
어댑터·설정·문서·집계된 학습/평가 요약만 제공합니다. 자동 추가 학습은 없습니다.
마음에 들지 않으면 **`tipo_lora=none`**으로 원본 출력을 사용하거나 CSV 노드를 사용하세요.

## English

**Image 30K is an experimental image-tag adapter. Results may be incorrect or simply
not match your taste.** Review and edit its output before image generation. It is not
a reliable artist/character identifier or a finished instruction-editing model.

Download: [raspie/gemma4-tipo-lora-image-30k](https://huggingface.co/raspie/gemma4-tipo-lora-image-30k).
The ~45.3MiB adapter is fetched on first execution with **`tipo_lora=image-30k`**.
It was trained once on **30,000 base image cases + 3,000 supplementary rows**;
rank 8 / alpha 16 language LoRA, with frozen vision, PLE and embeddings. Use the
exact Heretic ARA base revision above, not an already merged adapter checkpoint.

Update this node pack, install `requirements.txt` with ComfyUI's Python, and restart
after your queue finishes. Choose **`model_source=heretic_ara_q4`**, disconnect `clip`,
and leave model paths empty for automatic, hash-checked downloads. No extra LoRA loader,
llama.cpp, PEFT or Transformers is needed for this inference path. See [Q4 setup](GGUF_NATIVE.md).

Start with **`output_format=tags`** for image tags. For both outputs, select
**`output_format=both`, `tipo_lora_scope=tags_only`**: base Gemma writes prose, the adapter
renders that scene as tags. Existing workflow defaults remain unchanged; select these
settings explicitly. The tag-only Gemma TIPO Native node also supports this adapter.
Evaluation settings were temperature 0.5, repetition penalty 1.15, max tokens 512 and seed 42.
The [image API example](examples/gemma-image-30k.api.json) requires your own input image;
it is a POST /prompt body, not a canvas workflow. Offline adapter files belong in
`ComfyUI/models/loras/rs_gemma_tipo/image-30k/`; both files must match the pinned hashes.

**This is computationally demanding despite the small adapter.** Tested GPUs were
16GB RTX 4070 Ti SUPER and RTX 5060 Ti under Linux. We recommend that class of NVIDIA
CUDA GPU and 32GB+ system RAM for headroom, not as verified minimum requirements.
The Q4 text/tokenizer download is ~3.22GiB, or ~4.14GiB with vision, plus the adapter.
The alternative connected BF16 base is ~9.57GiB and is unnecessary for the Q4 path.
Disk sizes are not VRAM budgets. Image processing, dequantization buffers, KV cache,
LoRA, VRS and other generation models need additional memory. OOM/offloading can cause
failures or substantial delays. Start with one image and shorter output; unload other
models when needed. CPU-only, 8GB cards, AMD/Apple and Windows runtime were not validated.

In the 24-image 5060 Ti development run, node call time was about **36 seconds median,
16–126 seconds range**. These measured calls include variation in output/loading;
they are not throughput promises. The dual-output mode runs two sequential generations.

On those 24 images, macro F1 against **machine-produced PixAI tags** was
base **0.141**, previous internal 3k candidate **0.271**, Image 30K **0.352**.
Missing supplied names were **10 / 17 / 2** out of 93 occurrences respectively;
all models returned nonempty outputs. Names were supplied, not recognized from pixels.
Teacher agreement is not human accuracy; the 3k candidate is not public pilot-001/002.

Korean editing/VRS checks did **not** establish general improvement over the 3k candidate.
Across 12 context requests, missing required tags changed 17→18 and missing specified
names 0→3. In six retrieved-reference cases, the new model still omitted three names
and emitted conflicting tags in two requests. Lexical checks can count synonyms as misses.
Observed failures include short→long hair, an extra male subject and lost names.
Unwanted adult-related tags are possible. Exact banned-tag removal cannot fix all
semantic contradictions. VRS references may contaminate appearance. Prose was generated
by the same base in this comparison, so no prose-quality improvement is claimed.
Danbooru-focused training does not establish complete e621 or multilingual coverage.

This is a development-only evaluation; the reserved final set remains unused. Public
files contain adapter weights, config, documentation and aggregate metrics only, with
no source images, personal prompts, VRS experience or raw evaluation answers. No further
training runs automatically. Select **`tipo_lora=none`** to return to the base, or use
the model-free seasonal CSV node if the compute cost or outputs are not worthwhile.
