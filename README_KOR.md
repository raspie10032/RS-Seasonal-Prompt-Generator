# RS-Seasonal-Prompt-Generator

[English](README.md) | 한국어

계절별 CSV, 자연어 설명, 개인 이미지·태그 자료에서 이미지 생성용 프롬프트 재료를
만듭니다. **ComfyUI 노드 9개**와 독립 실행 도구 **VRS-INT**를 제공합니다.

**VRS-INT는 코드·문서·가상 예제만 공개합니다.** 초기 기억, 수집한 단보루 이미지와
태그, 특징 벡터, 추천 결과, 모델 가중치는 포함하지 않습니다. 각자 원하는 자료로
기억을 구성하며, 작가·캐릭터·작품명은 핵심 정보로 보존합니다.

**새 배포: [TIPO Image 30K](TIPO_IMAGE_30K.md)** — `tipo_lora=image-30k`로 이미지 태그를 보강합니다.
**고사양 GPU 권장(16GB NVIDIA 검증), 출력이 취향·이미지·지시와 다를 수 있습니다.**
한국어 수정·이름·외형 보존에는 오류가 남습니다. [설치·자원 부담·실제 평가](TIPO_IMAGE_30K.md)를 먼저 읽어주세요.

## 원하는 기능부터 선택하기

| 원하는 작업 | 사용할 기능 | 필요한 것 |
| --- | --- | --- |
| 계절·의상·배경 무작위 조합 | Seasonal Fashion Prompt Generator | ComfyUI. 모델·추가 패키지 불필요 |
| 텍스트·이미지 맥락을 태그형·자연어 지시문형으로 확장 | **[Gemma Prompt Expander Native](PROMPT_EXPANSION.md)** | 내부 Q4 자동 다운로드 또는 Gemma 4 CLIP 연결. 개인 태그 사전은 선택 입력 |
| ComfyUI PyTorch로 텍스트·이미지 태그 확장 | **Gemma TIPO Native** | Gemma 4 지원 ComfyUI + 내부 Q4 GGUF 또는 네이티브 safetensors |
| 한국어 요청으로 영문 태그·문장 작성 | RS AI Prompt Writer | ComfyUI + 별도 Qwen Writer. Gemma TIPO와 다른 모델 |
| 한국어·영어 설명이나 태그 확장 | Gemma TIPO 텍스트 모드 | ComfyUI + GGUF 실행 환경·모델. GPU 지원 실행 환경과 `gpu_layers=-1`로 GPU 적재 |
| 참고 이미지에서 프롬프트용 태그 생성 | Gemma TIPO 이미지 모드 | ComfyUI + GGUF·호환 mmproj·llama.cpp 비전 실행 파일 |
| 개인 자료에서 비슷한 그림의 태그·이름 회수 | VRS-INT | Python 3.11 이상, NVIDIA CUDA 환경, 별도 비전 인코더, 개인 이미지·태그 쌍 |

VRS-INT의 목적은 **새로운 프롬프트에 쓸 참고 재료를 가져오는 것**입니다. 인코더를
재학습하거나 원작자를 확정하는 기능은 아니며, 완성된 자연어 프롬프트를 작성하지는
않습니다. 반환된 태그를 이미지 프롬프트에 사용하거나 별도 생성기에 전달하세요.
현재 공개본에는 [네이티브 VRS 노드 4개](vrs_int/COMFYUI_NATIVE.md)가 있으며,
경험 구축·불러오기·검색 결과를 네이티브 Gemma Prompt Expander에 연결합니다.

**선택적 TIPO LoRA:** 네이티브 Gemma 노드 안의 `tipo_lora`를 선택하면 처음에만
약 45.3MiB의 실험 후보를 받습니다. 베이스는 **내부 Q4(텍스트 약 3.22GiB, 이미지 포함
약 4.14GiB)** 또는 기존 BF16(약 9.57GiB)을 사용합니다. [Q4 설치 안내](GGUF_NATIVE.md).
기본값은 `none`이고 추가 로더 노드는 없습니다.
[베이스 설치·다운로드·제약](NATIVE_TIPO.md#built-in-experimental-lora--노드-내부-실험-lora)을 확인하세요.

- **ComfyUI 사용자:** 아래 [설치](#comfyui에-설치)부터 시작하세요.
- **VRS-INT만 사용:** [VRS-INT 전체 안내](vrs_int/README_KOR.md)로 바로 이동하세요. ComfyUI와 전체 Gemma 언어모델은 필요 없습니다.
- **자료 준비:** [이미지·텍스트 쌍 예제](examples/vrs-int/README_KOR.md)를 따라가세요.
- **영문 안내:** [Overview](README.md) · [VRS-INT](vrs_int/README.md) · [Pair examples](examples/vrs-int/README.md).

## ComfyUI에 설치

### 1. `custom_nodes`에 저장소 넣기

ComfyUI 폴더에서 실행합니다.

```bash
git clone https://github.com/raspie10032/RS-Seasonal-Prompt-Generator.git custom_nodes/RS-Seasonal-Prompt-Generator
```

ZIP으로 받았다면 다음 구조로 압축을 풉니다. 저장소 폴더가 한 번 더 중첩되지 않게 합니다.

```text
ComfyUI/
  custom_nodes/
    RS-Seasonal-Prompt-Generator/
      __init__.py
      seasonal_fashion_prompt_node.py
      gemma_tipo_node.py
      data/
```

ComfyUI를 재시작하고 **`prompt`** 분류 또는 노드 이름 검색으로 추가합니다.
프롬프트 생성 노드는 `STRING`을 출력하며, 네이티브 VRS 노드는 인코더·기억·생성 상태
연결도 출력합니다. 문자열을 텍스트 인코더에 연결할 때 ComfyUI 버전에 따라
프롬프트·텍스트 위젯을 입력으로 변환해야 합니다. 텍스트 미리보기 노드를 연결하면
이미지 생성 전에 문자열을 확인할 수 있습니다.

### 2. Gemma TIPO Native 준비 (권장)

**내부 Q4는 ComfyUI의 Python으로 `requirements.txt`를 설치합니다.** 순수 Python
`gguf` 파서만 추가합니다. `model_source=heretic_ara_q4`를 선택하고 `clip`을 해제하면
노드가 필요한 모델을 받습니다. [Q4 설치·다운로드·예제](GGUF_NATIVE.md)를 참고하세요.
llama.cpp, C++ 빌드 도구, CUDA 휠이나 torch 교체는 필요 없습니다.

기존 전체 safetensors는 `model_source=connected_clip`(기본값)으로 다음처럼 연결합니다.

1. Gemma 4와 `CLIP.generate`를 지원하는 ComfyUI를 사용하세요.
2. **ComfyUI 형식의 Gemma 4 E2B safetensors**를 `ComfyUI/models/text_encoders/`에 넣습니다.
3. `Load CLIP`에서 해당 모델, `type=stable_diffusion`, `device=default`를 선택합니다.
4. `CLIP`을 **Gemma TIPO Native**에 연결하고 요청을 입력합니다. 이미지는 선택 입력입니다.
5. `tags`를 `Preview as Text` 또는 이미지 생성용 텍스트 입력에 연결합니다.

**대응하는 Gemma 4 E2B GGUF + mmproj는 [내부 Q4 경로](GGUF_NATIVE.md)로 사용합니다.** 원래 튜닝된
전체 HF safetensors가 있으면 [변환 도구·모델 준비 안내](NATIVE_TIPO.md)로
학습된 텍스트·비전 연결부를 함께 보존하세요. 일반 Gemma 원본은 같은 TIPO
튜닝 모델이 아닙니다. 변환 자체에는 재학습이 필요 없습니다.

[네이티브 상세 사용법·Windows 명령·문제 해결](NATIVE_TIPO.md) ·
[API 워크플로 예제](examples/gemma-tipo-native.api.json)

기존 **Gemma TIPO** GGUF 노드와 **RS AI Prompt Writer**는 별도 선택 기능입니다.
그 경로에만 `requirements-legacy.txt`가 필요합니다. GPU를 쓰려면 환경에 맞는
llama-cpp-python CUDA 빌드를 먼저 선택하세요. 단순 pip 설치는 CPU 빌드를
설치하거나 소스 빌드를 시작할 수 있습니다. 정렬기만 원하면 ComfyUI의 Python으로
`python -m pip install --no-deps tipo-kgen`을 사용하세요. VRS-INT 설치는 별도입니다.

## 계절별 프롬프트 생성기

1. **Seasonal Fashion Prompt Generator**를 추가합니다.
2. 계절을 선택하고 의상은 `random`, 배경은 `background_toggle=on`으로 시작합니다.
3. 출력을 텍스트 미리보기 또는 이미지 모델의 긍정 프롬프트 입력에 연결합니다.
4. 실행합니다. 모델로 태그를 확장하려면 출력을 **Gemma TIPO.prompt**에 연결합니다.

| 입력 | 동작 |
| --- | --- |
| `season` | `random`, `spring`, `summer`, `autumn`, `winter` |
| `background_toggle` | `on`: 계절 배경, `off`: 배경 묶음 제외, `simple`: 흰색·회색·검은색 단순 배경 |
| `weather_toggle`, `time_toggle` | `random`, `off`, 특정 값. 배경이 `on`일 때 사용 |
| `top_toggle`, `bottom_toggle`, `one_piece_toggle` | 상의·하의·원피스. `random` 또는 `off` |
| `accessory_toggle`, `hat_toggle`, `shoes_toggle`, `socks_toggle` | 액세서리·모자·신발·양말 무작위 선택 또는 끄기 |
| `composition_toggle`, `gaze_toggle`, `pose_toggle`, `body_direction_toggle` | 구도·시선·자세·몸 방향. `random`, `off`, 특정 값 |
| `additional_situation_toggle` | 배경·날씨·시간 조건에 맞는 CSV 상황 설명 추가 |
| `wanna_shot` | `Yes`이면 일반 장면 대신 카페·커피·텀블러 특수 장면 사용 |
| `seed` | 0을 포함한 고정 시드로 선택 재현 |

같은 시드와 입력은 같은 결과를 반환합니다. `0`도 고정 시드이며 전역 난수 상태를
바꾸지 않습니다. 상의·하의를 독립적으로 켤 수 있고, 원피스와 투피스가 모두
가능하면 둘 중 하나를 고릅니다.

### CSV를 취향에 맞게 바꾸기

[`data/`](data/)의 UTF-8 파일을 수정하되 헤더·열 구조를 유지합니다. 캐시를 다시
읽도록 ComfyUI를 재시작합니다. 저장소를 갱신하기 전 개인 수정본을 보관하세요.

| 파일 | 내용 |
| --- | --- |
| `seasonal_fashion.csv` | 계절, 의상 분류, 항목. 분류: `top`, `bottom`, `one_piece`, `accessory`, `hat`, `shoes`, `socks` |
| `seasonal_backgrounds.csv` | 계절, 배경 분류, 설명 |
| `seasonal_weather.csv`, `seasonal_times.csv` | 계절별 날씨·시간 |
| `general_composition.csv`, `gaze_direction.csv`, `poses.csv`, `body_directions.csv` | 구도·시선·자세·몸 방향 |
| `additional_situations.csv` | 상황, `\|`로 구분한 일치 조건, 설명 |

쉼표가 들어가는 값은 CSV 따옴표로 감쌉니다. 상황 조건은 `|`로 OR, `&`로 AND를
지정합니다. 예: `misty & lakeside|foggy & lakeside`.

## AI로 자연스러운 프롬프트 작성하기

[VRS-INT](vrs_int/README.md)는 별도로 실행하는 실험용 기억 구축·이미지 참고 태그 추천기입니다.
**코드를 배포하고 자료는 사용자가 취향에 맞게 선택해 구성하는 방식**입니다.
이미지·태그·인코더 가중치는 포함하지 않으며, 로컬 이미지 묶음별 구축·추가를 지원합니다.
작가 후보는 창작 참고용이며 입력 이미지의 실제 작가를 확인했다는 뜻이 아닙니다.
[이미지·텍스트 쌍 예제](examples/vrs-int/README.md)에 JSONL 템플릿, 이미지별 JSON을
묶는 스크립트, 모델 없는 입력 검사, 구축·추가·추천 명령을 제공합니다.
자연어 설명은 메타데이터로 보존하며, 현재 추천에는 분류된 태그 배열을 사용합니다.

기존 단보루 관측에 VRS 판정을 적용한 별도 [4070 GPU 연관도 실험](experiments/README.md)도
있습니다. [전체 자료 구축](experiments/FULL_DATASET.md)과
[보류 이미지 128장의 태그 추천 검사](experiments/IMAGE_RETRIEVAL.md)를 진행했습니다.
이미지 검사에서 과거 특징 캐시의 픽셀 변환 결함이 확인돼 별도 수정 비교를 했습니다.
이제 별도 [이미지 참고 → 새 프롬프트 생성 실험](experiments/IMAGE_PROMPT.md)에서
수정 이미지 인덱스 → VRS 참고 태그 → 계절 CSV → Writer까지 연결했습니다.
태그 정답 복원이 아닌 새 장면 작성이 목적이며, ComfyUI 노드에는 아직 연결하지 않았습니다.

“하얀 셔츠와 파란 치마를 입은 여성이 공원에 서 있다”처럼 장면을 설명하려면
**RS AI Prompt Writer (Korean / English → Prompt)**를 사용하세요. 지시를 따르는
범용 모델로 핵심 설정을 유지하면서 프롬프트를 작성합니다. 기존 TIPO는 태그 확장과
이미지 입력용으로 유지합니다. TIPO는 지시를 강화해도 입력 내용을 빠뜨리거나 바꾸는
경우가 실제 검사에서 확인되었습니다.

1. `prompt` 카테고리에서 **RS AI Prompt Writer** 노드를 추가합니다.
2. `prompt`에 한국어/영어 설명을 쓰거나 Seasonal Fashion Prompt Generator의
   `STRING` 출력을 연결합니다.
3. `model`은 자동으로 둡니다. 첫 실행에는 Unsloth의 Qwen3-4B-Instruct-2507
   Q4_K_M 파일 약 **2.50 GB**를 받습니다. 리비전을 고정하고 SHA-256을 검증합니다.
   로컬에서 실행하며 API 키나 유료 API 호출은 필요하지 않습니다.
4. `tags`는 쉼표로 구분한 영어 태그, `natural_language`는 자연스러운 영어 문단을
   출력합니다. 결과를 **Preview as Text** 또는 이미지 모델의 텍스트 인코더에 연결합니다.
5. 처음에는 `creativity=balanced`, `length=medium`, `temperature=0.6`으로 시작합니다.

| 설정 | 동작 |
| --- | --- |
| `faithful` | 입력을 번역·재구성하고 새로운 세부 설정은 만들지 않도록 지시합니다. |
| `balanced` | 지정된 설정을 유지하고 어울리는 조명·분위기·구도를 보충합니다. |
| `creative` | 명시한 요구사항을 유지하면서 미지정 부분을 풍부하게 만듭니다. |
| `length` | short / medium / long 목표 길이. 입력 내용 보존을 우선합니다. |
| `seed` | 동일 모델·실행 환경에서 샘플링을 재현합니다. 하드웨어·빌드가 다르면 결과도 달라질 수 있습니다. |
| `gpu_layers` | `-1`(기본값): GPU 전 레이어. 호환되는 llama-cpp-python GPU 빌드가 필요합니다. `0`은 명시적 CPU 선택이며 자동 CPU 재시도는 없습니다. |
| `cpu_threads` | 기본값 `2`: 생성·입력 처리의 CPU 보조 스레드 수. 데스크톱 응답성을 위해 낮게 유지합니다. |

위 설정은 모델에 주는 지시이며 의미 보존을 보장하지는 않습니다. 중요한 요구사항은
출력에서 확인하세요. 추가 GPU 검사에서는 우산만 명시한 입력에 비 오는 날씨를
덧붙이는 등 balanced 모드가 미지정 설정을 추가하는 사례가 남았습니다.
문장형에 단편적인 구절이나 작은 문법 오류가 섞일 수도 있습니다.
이 노드는 텍스트를 받으며 이미지 입력은 Gemma TIPO를 사용합니다.
사용자 모델은 `ComfyUI/models/gguf/`에 두며 채팅 템플릿이 있는 지시 모델을 사용하세요.
노드 등록만으로는 모델을 다운로드하지 않습니다. 생성 실패 시 원문을 성공 결과처럼
반환하지 않고 노드 오류를 표시합니다. 기존 CUDA 휠이 라이브러리를 찾지 못하면
**ComfyUI와 같은 Python 환경**에서 GPU 휠과 맞는 런타임을 사용해야 합니다.
`RS_TIPO_FORCE_CPU=1`은 이 노드에도 적용되지만 깨진 휠 자체를 복구하지는 않습니다.
GPU 요청 시 CPU 전용 빌드는 모델 다운로드 전에 거부합니다. CPU 실행을 원할
때에만 `gpu_layers=0`을 선택하세요. GPU가 여러 개면 Python 시작 전에
`CUDA_VISIBLE_DEVICES`를 원하는 GPU UUID로 지정할 수 있습니다.
이 환경 변수는 ComfyUI 프로세스 전체에 적용됩니다.

기본 모델 출처: [Qwen 원본](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[Unsloth GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF), Apache-2.0.
고정 리비전 `a06e946bb6b655725eafa393f4a9745d460374c9`, 파일
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, SHA-256
`3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`.

## Gemma TIPO: 텍스트 또는 이미지에서 태그 생성

아래는 **기존 GGUF 노드**의 설명입니다. llama.cpp 없는 설치는 [Gemma TIPO Native](NATIVE_TIPO.md)를 사용하세요.

하나로 통합된 노드입니다. **`image` 를 연결하면 → 비전 모드**
(이미지를 Gemma-4 비전 = llama.cpp `mtmd` + 자동 다운로드된
**co-trained 비전 모델 + mmproj** 로 캡션; `gpu_layers=-1`이면 GPU 자동 선택).
**이미지 없음 → 텍스트 모드**(`prompt`, 한국어/영어 자연어 또는
태그를 `llama-cpp-python` 으로 in-process 확장). 어느 쪽이든 출력은
동일한 `kgen.formatter` 후처리(정렬 / 중복 제거 / 중복 일반태그 제거 /
인원수 충돌 정리)를 거칩니다.

> ⚠️ **비전 모드는 TIPO식이며, 엄밀한 1:1 태거가 아닙니다.**
> co-trained 비전 모델 + mmproj 로 이미지의 핵심 내용을 그림에서
> 가져오지만, 그 위에 **그럴듯하게 확장**해 더 풍부한 Danbooru 태그
> 세트를 만듭니다(TIPO식 창의적 업샘플링): 이미지에 그대로 있지 않은
> 태그가 일부 포함되고 `seed` 에 따라 실행마다 달라집니다. 레퍼런스
> 이미지를 다양한 text-to-image 프롬프트로 바꾸는 데 좋습니다 —
> 정확하고 충실한 라벨링이 필요하면 전용 태거를 쓰세요.

텍스트 모드는 태그뿐 아니라 **한국어 또는 영어 자연어**도 받습니다:

| 입력 | 출력 (예시) |
| --- | --- |
| `spring, white blouse, pleated skirt, park` | `spring, white blouse, pleated skirt, park, outdoors, school uniform, long sleeves, sitting, solo, …` |
| `빨간 원피스, 밀짚모자, 해변` (한국어) | `straw hat, red one-piece swimsuit, sun hat, beach, ocean, swimsuit, outdoors, …` |

입력:
- `prompt`: 변환할 텍스트(한국어/영어 자연어 또는 태그).
- `model`: **`ComfyUI/models/gguf/`** 에서 찾은 `*.gguf` 파일
  드롭다운. `(auto: download default)` 를 고르면 기본 모델
  (`gemma4-tipo-ko-v2-Q4_K_M.gguf`,
  [`raspie/gemma4-tipo-ko-gguf`](https://huggingface.co/raspie/gemma4-tipo-ko-gguf))
  을 최초 사용 시 해당 폴더로 다운로드합니다 — 다운로드 중
  **노드에 진행 바가 표시**되며 이후 재사용(다시 다운로드하지
  않음). 또는 직접 만든 `.gguf` 를 `ComfyUI/models/gguf/` 에 넣고
  선택할 수 있습니다(새 파일을 보려면 노드 목록을 새로고침).
- `tag_length`: 목표 상세도(`very_short` … `very_long`).
- `sort`: `kgen.formatter` 가 적용하는 프롬프트 정렬 프리셋(태그는
  special/characters/copyrights/artist/general/quality/meta/rating
  로 분류됨):
  - `danbooru` (기본): artist, rating, special, characters,
    copyrights, general, quality, meta.
  - `quality_first`: quality 태그를 앞으로.
  - `artist_first`: artist/character 를 앞으로.
  - `general_only`: special + characters + general 태그만 유지.
  - `simple`: special, rating, general 만.
- `temperature`: 샘플링 temperature.
- `ban_tags`: 결과에서 제거할 콤마 구분 태그.
- `seed`: 변형을 위해 확장을 바꿉니다.
- *(선택)* `image`: ComfyUI IMAGE를 연결하면 **비전 모드**(이미지
  → 태그)로 전환됩니다. **완전 자동** — 최초 비전 사용 시 OS/아키텍처와
  GPU에 맞는 프리빌트 llama.cpp mtmd 바이너리(Windows/NVIDIA → CUDA +
  cudart; Linux/NVIDIA·AMD/Intel → Vulkan; macOS → Metal; 그 외 CPU,
  빌드 불필요)와 트레인된 비전 모델 + mmproj를 자동 다운로드합니다.
  수동으로 설치하거나 설정할 것이 없습니다.
- *(선택, 비전 전용)* `mmproj_path`: **비워 두세요** — 트레인된 비전
  mmproj가 자동 다운로드됩니다(비전 모델과 페어). 직접 만든 mmproj를
  가리킬 때만 설정하세요.
- *(선택, 두 모드 공통)* `gpu_layers`: **`0` = CPU**(기본값, 저장된
  워크플로 호환), **`-1` = GPU 자동 선택/전 레이어**, 양수 = 레이어 수.
  텍스트는 GPU 지원 `llama-cpp-python` 빌드가 필요합니다. 비전에서 GPU
  초기화·추론에 실패하면 CPU로 재시도합니다. `RS_TIPO_FORCE_CPU=1`은
  두 모드 모두 CPU 강제, `LLAMA_MTMD_CLI`는 로컬 실행 파일 경로입니다.
  CPU 모드에서는 비전 프로젝터의 GPU 오프로드도 끕니다.

참고:
- 설명이 길고 자세할수록 더 많고 더 정확한 태그가 나옵니다.
- 한국어 설명은 성공한 태그 출력에 섞이지 않습니다. 기존의 콤마 구분
  영어 입력 태그는 먼저 보존한 뒤 정리합니다.
- 금지 태그는 대소문자·공백/밑줄 차이를 무시합니다. 선택적인 카테고리
  정렬 라이브러리가 없어도 금지 태그와 중복 제거는 적용됩니다.
- 토큰 절약을 위해 중복 일반 태그는 제거됩니다: 같은 핵심 명사를
  가진 더 구체적인 태그(예: `white skirt`)가 있으면 단순 `skirt` 는
  제거됩니다. `dress`/`dress shirt` 처럼 구분되는 태그는 유지됩니다.
- 충돌하는 인원수 태그는 그룹별로 정리됩니다: `1girl` 이 있으면
  `4girls` / `multiple girls` 는 제거(`1girl` + `1boy` 는 유지 —
  그룹은 독립적).
- 의존성·모델·추론 실패 시 텍스트는 원래 프롬프트, 비전은 빈 문자열을
  반환합니다. `[TIPO]` / `[TIPO-V]` 콘솔 로그로 폴백과 생성 성공을
  구분하세요. 비전은 한 장만 받으므로 배치는 한 장을 선택해 연결합니다.
- 다운로드는 임시 파일을 거치며 빈 파일과 불완전한 전송을 거부합니다.
  이는 전송 완료 검사이며 암호학적 모델 무결성 검증은 아닙니다.

### 지원 환경 및 요구사항

| | |
| --- | --- |
| **OS** | Windows / Linux / macOS — ComfyUI가 동작하고 `llama-cpp-python` 휠이 존재하는 모든 환경(x86-64, Apple Silicon/arm64) |
| **노드 1 (Seasonal)** | 모든 ComfyUI 설치. 의존성 없음, 모델 없음. |
| **노드 2 (Gemma TIPO)** | 비전: `gpu_layers=-1`에서 GPU 자동 선택(CUDA / Vulkan / Metal), CPU 폴백. 텍스트: 기본 CPU(GPU는 CUDA `llama-cpp-python` 빌드로). |
| **RAM** | 최소 ~8 GB (4-bit 모델에 ≈5 GB 사용) · 16 GB 이상 권장 |
| **디스크** | 텍스트 ~4 GB(3.2 GB GGUF + 의존성); 비전은 추가로 ~3.2 GB 비전 GGUF + ~1 GB mmproj + GPU 프리빌트(최초 비전 사용 시 1회) |
| **속도** | **비전**은 `gpu_layers=-1`에서 GPU 사용 가능; 속도는 하드웨어에 따라 다름. **텍스트**는 `llama-cpp-python` in-process(기본 CPU 휠이라 느림; 속도는 GPU 빌드 설치). |
| **Python** | ComfyUI 내장 환경; `llama-cpp-python >= 0.3.23`(자동 설치/업그레이드, 1회 재시작 안내) |

### 설치 (노드 2 전용)

**수동 설치가 필요 없습니다.** 노드 2를 실제로 처음 실행할 때
필요한 의존성을 활성 환경에 자동 설치합니다: 정렬용 `tipo-kgen`,
**텍스트 모드에서만** `llama-cpp-python` **>= 0.3.23** 프리빌트 휠.
자동 설치는 llama-cpp-python 소스 빌드를 시작하지 않으며, 맞는 휠이
없으면 설치 안내와 함께 원래 프롬프트로 폴백합니다. 너무 오래된 빌드는 gemma4 아키텍처를 로드할 수 없어
기존의 너무 오래된 `llama-cpp-python` 은 자동 업그레이드됩니다 — 세션
중간에 그렇게 되면 ComfyUI가 **1회 재시작**을 요청합니다(C 확장은
핫 리로드 불가). 노드 1은 아무것도 설치하지 않습니다.

대신 미리 수동 설치하려면(선택):

```
pip install -r requirements-legacy.txt
```

프리빌트 `llama-cpp-python` 휠 설치가 사용자 플랫폼/Python 에 맞는
휠을 찾지 못하면 직접 1회 설치하세요:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

## VRS-INT: 개인 참고 기억

```text
개인 이미지 + 분류된 태그 → 기존 비전 인코더 → 판본이 있는 개인 기억
새 참고 이미지           → 같은 인코더      → 이웃의 태그 + 이름 후보
```

1. [실행 환경과 비전 전용 인코더 준비](vrs_int/README_KOR.md#준비).
2. [JSONL 또는 이미지별 설명 파일 작성](examples/vrs-int/README_KOR.md).
3. [입력 검사 → 최초 구축 → 추천](vrs_int/README_KOR.md#최초-구축과-추천).
4. [새 묶음 추가 또는 태그 수정](vrs_int/README_KOR.md#자료-추가와-수정).
5. [백업·이전·판본 되돌리기](vrs_int/README_KOR.md#백업과-이전-되돌리기).

Gemma E2B의 시각 인코더만 사용하며 언어모델·`mmproj.gguf`를 읽지 않습니다.
명시적으로 분류한 작가·캐릭터·작품은 한 번만 등장해도 보존합니다. 설명문과 원본
태그도 감사 기록에 남지만 **설명문만 있고 태그가 없으면 구축할 수 없습니다.**

개인 자료는 저장소 밖이나 Git 제외 폴더인 `vrs-int-data/`, `vrs-int-models/`에
보관하세요. 압축 기억에도 태그·출처 메타데이터가 들어 있습니다. **압축은 익명화가
아닙니다.** 자동 수집·업로드·공용 기억 동기화는 하지 않습니다.

## 현재 Gemma 실행 기준과 경험 연결

[Gemma 4 TIPO 실행 기준](vrs_int/GEMMA4_RUNTIME.md)에서 **비전 FP16·PLE Q4 후보**의
선택 방법과 기존 자동 모델의 차이를 확인하세요. 층·어휘는 유지하며, 혼합 정밀도
후보는 새 TIPO 학습 완성본이 아닙니다. 내부 FFN 계산 재사용은 포함하지 않습니다.

- [생성 이미지 메타데이터와 선택적 태깅](vrs_int/GENERATED_IMAGES.md)
- [한국어 텍스트 지식풀·번역 사전](vrs_int/TAG_DICTIONARY.md)
- [VRS 참고 상태 → TIPO 확장](vrs_int/GENERATION_STATE.md)

VRS 상태는 CLI와 [네이티브 VRS 노드](vrs_int/COMFYUI_NATIVE.md)에서 연결합니다.
한국어 제외 지시나 장면 조건을 모델이 모두
지킨다고 보장하지 않으므로 생성 결과를 확인하세요. 비전 TIPO 호출은 종료 후
해제되며, 기존 텍스트 TIPO는 모델을 캐시합니다. CPU 보조 스레드는 기본 2개입니다.

## 문제 해결

| 증상 | 확인할 것 |
| --- | --- |
| 노드가 안 보임 | 폴더 중첩, 재시작, 시작 로그 import 오류 확인. `prompt` 또는 `VRS Native` 검색. [네이티브 VRS 조건](vrs_int/COMFYUI_NATIVE.md) 참고 |
| TIPO가 원문 그대로 반환 | 의존성·모델·추론 오류 시 원문 반환 가능. `[TIPO]` 로그 확인 |
| 이미지 출력이 빈 문자열 | `[TIPO-V]` 로그, 모델·프로젝터 호환성, 실행 라이브러리, 이미지, 추론 제한 600초 확인 |
| `unknown model architecture: gemma4` | ComfyUI의 Python에서 Gemma4 지원 llama-cpp-python ≥0.3.23 확인. 업그레이드 후 재시작 |
| CPU가 계속 바쁨 | TIPO의 `gpu_layers=-1`과 GPU 빌드 확인. 비전은 실패 시 CPU 재시도가 가능하므로 로그 확인. VRS-INT 설정과 별개 |
| 설치했는데 패키지가 없다고 나옴 | 실제 ComfyUI의 Python에 설치했는지 확인 |
| 모델 다운로드 실패 | 주소·인터넷·디스크·로그 확인. 파일이 있는 것만으로 다운로드 완료라고 판단하지 않기 |
| VRS-INT 오류 | [VRS-INT 문제 해결](vrs_int/README_KOR.md#문제-해결) 참고 |

## 갱신과 검증

수정하지 않은 Git 설치는 저장소 폴더에서 갱신합니다.

```bash
git pull --ff-only
```

노드를 사용하면 ComfyUI도 재시작합니다. 개인 CSV·코드 수정본은 먼저 보존하세요.
코드 갱신은 자료를 자동 수집하거나 VRS-INT 기억을 변환·활성화하지 않습니다.
개인 기억의 정확한 snapshot 해시와 인코더 출처 기록을 함께 보관하세요.

공개 오프라인 검사는 가상 자료와 대체 인코더를 사용합니다.

```bash
python -m pip install numpy Pillow
python -m unittest discover -s tests -v
python scripts/check_code_only.py
```

입력·메타데이터·판본·수정·패키지·게시 규칙을 확인합니다. CI는 Linux·Windows·macOS와
Python 3.11·3.13 조합으로 실행합니다. GPU 추론·추천 품질 평가는 아닙니다.
게시 검사는 스테이징된 변경을 포함한 Git 인덱스를 검사하므로 diff도 검토하세요.
이전 노드 변경: [한글](Update_log_KOR.txt) / [영문](Update_log_ENG.txt).

## 라이선스와 출처

코드는 [Apache-2.0](LICENSE)입니다. 모델 가중치·데이터·이미지에는 각각의 이용
조건이 적용되며 코드 라이선스로 재허가되는 것이 아닙니다.

| 프로젝트 | 사용한 부분 |
| --- | --- |
| [KGen / TIPO](https://github.com/KohakuBlueleaf/KGen), [z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension) | 태그 확장·분류 후처리·TIPO 학습 자료 계보 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | GGUF 추론 |
| [Gemma](https://ai.google.dev/gemma) | 기반 모델 계열. 별도 모델 이용 조건 적용 |
| [raspie/gemma4-tipo-ko-gguf](https://huggingface.co/raspie/gemma4-tipo-ko-gguf) | 기본 한국어 TIPO 모델 |
| [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | 기존 기반 프로젝터 출처. 현재 비전 기본값은 대응 학습 mmproj |
| [p-e-w/gemma-4-E2B-it-heretic-ara](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara) | TIPO 기반 계보, 별도 VRS-INT 시각 인코더 출처 |

처음 ChatGPT로 작성했고 이후 Claude Code와 Codex로 개발했습니다.

VRS inside ComfyUI / VRS 네이티브 구축·검색: [설치·연결·API 예제](vrs_int/COMFYUI_NATIVE.md)
