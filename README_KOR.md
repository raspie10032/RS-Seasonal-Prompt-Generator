# RS-Seasonal-Prompt-Generator

*[English](README.md) | 한국어*

 CSV 데이터에서 패션 아이템·배경·날씨·시간·추가 상황 디테일을 무작위로
 조합해 계절별 패션 프롬프트를 생성합니다.

> **v2.8.0** 에서 제대로 **파인튜닝된 비전 모델**
> (`Gemma-tipo-vision-v1`) 이 적용되었습니다. 이제 비전 파이프라인은
> 텍스트 TIPO 모델 + 베이스 unsloth mmproj 조합이 아니라, **학습된
> 멀티모달 프로젝터** 가 들어간 전용 페어를 사용합니다. 출력이
> 실제로 **이미지 기반** 으로 바뀌었습니다 — 캐릭터·색상·옷·구도
> 모두 이미지에 묶임 (인기 Danbooru 캐릭터 식별률 ~60–80% 확인). 단순
> 그럴듯한 환각이 아닙니다. 비전 모드 최초 사용 시 자동 다운로드.
> 텍스트 모드 기본 모델은 기존 `gemma4-tipo-ko-v2` 유지.
>
> **v2.7.0** 에서 텍스트 TIPO 노드와 비전 TIPO 노드를 하나로 통합.
> **v2.7.1** 에서 비전 모드 모든 플랫폼 자동화.
> **v2.7.3** 에서 텍스트 기본 모델을 `gemma4-tipo-ko-v2-Q4_K_M.gguf`로 업그레이드.

이 팩은 **독립적인 두 개의 노드**를 제공합니다(카테고리: `prompt`):
1) Seasonal Fashion Prompt Generator (CSV, 의존성 없음), 2) Gemma TIPO
(Prompt / Image → Tags). 이 폴더를 `ComfyUI/custom_nodes/` 에
클론/복사한 뒤 ComfyUI를 재시작하면 설치됩니다. 각 노드는 `STRING`
을 출력하며 서로 연결(체이닝)할 수 있습니다.

## 쇼케이스

![ComfyUI의 Gemma TIPO 노드](assets/gemma-tipo-example.png)

한국어 자연어 입력 → 깔끔한 영어 Danbooru 태그 출력:

> **입력:** `소녀가 하얀 세일러복과 스카프를 하고 있음.`
>
> **출력:** `white sailor collar, white skirt, black hairband, blue eyes,
> blush stickers, long sleeves, looking at viewer, open mouth, red scarf,
> school uniform, serafuku, sitting, solo focus, sparkle sticker, striped
> clothes, sweatdrop, collared shirt, white background`

그리고 **비전 모드 (v2.8.0+)** — 이미지를 연결하면 학습된 비전
모델이 이미지에 기반한 Danbooru 태그를 생성합니다(캐릭터·색상·옷·구도
정확):

![ComfyUI의 Gemma TIPO 비전 모드](assets/gemma-tipo-vision-example.png)

## 노드 1 — Seasonal Fashion Prompt Generator

원래의 생성기입니다. `season` 을 고르고 패션 / 배경 / 날씨 / 포즈 /
상황 옵션을 토글하면, 번들된 CSV 데이터에서 무작위로 조합한
콤마 구분 프롬프트를 반환합니다. **의존성 없음** — 아무것도
설치하지 않고 모델도 실행하지 않습니다.

## 노드 2 — Gemma TIPO (Prompt / Image → Tags)

하나로 통합된 노드입니다. **`image` 를 연결하면 → 비전 모드**
(이미지를 Gemma-4 비전 = llama.cpp `mtmd` + 자동 다운로드된 **학습된
`Gemma-tipo-vision-v1` 페어** 로 캡션). **이미지 없음 → 텍스트 모드**
(`prompt`, 한국어/영어 자연어 또는 태그를 `llama-cpp-python` 으로
in-process 확장). 어느 쪽이든 출력은 동일한 `kgen.formatter`
후처리(정렬 / 중복 제거 / 중복 일반태그 제거 / 인원수 충돌 정리)를
거칩니다.

> **비전 모드 (v2.8.0+):** Danbooru 이미지+태그 페어로 파인튜닝된
> 실제 비전 모델(`Gemma-tipo-vision-v1` — 학습된 멀티모달 프로젝터 +
> LM LoRA)이 이미지를 처리합니다. 출력이 **이미지 기반** 입니다:
> 캐릭터·색상·옷·구도 모두 이미지에 반영되며, 인기 Danbooru 캐릭터
> 식별률 ~60–80% + 이미지별 속성 캡처. "TIPO식 확장" 성격도 일부
> 남아 있어(인접 태그가 그럴듯하게 채워짐) 이미지 *묘사* 와 풍부한
> 프롬프트 *생성* 양쪽 다 적합합니다.
>
> **주의사항:**
> - 풀 컬러 완성된 anime/manga(Danbooru 분포)에 가장 강합니다.
>   흑백 스케치·라인아트·아마추어 그림·실사 사진은 도메인 밖이고,
>   출력이 색상·옷 환각을 내놓습니다.
> - Danbooru 커버리지가 적은 작품(예: 클래식 디지몬)은 캐릭터 식별이
>   크게 떨어집니다.
> - Q4 양자화로 인해 이미지 grounding은 맞아도 캐릭터 이름을 잘못
>   부르는 경우가 가끔 있습니다(예: Arlecchino를 다른 캐릭터로).
> - 모델이 **태그 포맷**으로 강하게 파인튜닝되어 있어 자연어 산문을
>   직접 요청하면 환각이 심합니다. 자연어가 필요하면 태그 출력을 다른
>   텍스트 LM으로 한 번 더 변환하세요.
>
> 이전 버전(≤2.7.3)은 텍스트 TIPO 모델 + 베이스 unsloth mmproj 조합
> 으로 동작했으며, 실제 이미지 기반이 아니라 환각만 의존했습니다.

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
  → 태그)로 전환됩니다. **완전 자동** — 최초 비전 사용 시 OS/아키텍처에
  맞는 프리빌트 llama.cpp mtmd 바이너리(Windows / Linux / macOS ·
  x64 / arm64, 빌드 불필요)와 학습된 비전 페어
  (`Gemma-tipo-vision-v1-E2B-heretic-ara-Q4_K_M.gguf` +
  `Gemma-tipo-vision-v1-E2B-heretic-ara.mmproj-f16.gguf`, 총 ~4.4 GB
  1회) 를 자동 다운로드합니다. 수동 설치/설정 없음.
- *(선택, 비전 전용)* `mmproj_path`: **비워 두세요** — 비전-v1 LM
  사용 시 학습된 mmproj, 레거시 텍스트 모델 사용 시 베이스 unsloth
  mmproj를 자동으로 맞춥니다. 직접 만든 mmproj를 가리킬 때만 설정.
  `gpu_layers`: `0` = CPU(기본); 값을 올리면 레이어를 GPU로
  오프로드합니다.

참고:
- 설명이 길고 자세할수록 더 많고 더 정확한 태그가 나옵니다.
- 원본 입력 텍스트(한국어 포함)는 태그 출력에 **섞이지 않습니다**.
- 토큰 절약을 위해 중복 일반 태그는 제거됩니다: 같은 핵심 명사를
  가진 더 구체적인 태그(예: `white skirt`)가 있으면 단순 `skirt` 는
  제거됩니다. `dress`/`dress shirt` 처럼 구분되는 태그는 유지됩니다.
- 충돌하는 인원수 태그는 그룹별로 정리됩니다: `1girl` 이 있으면
  `4girls` / `multiple girls` 는 제거(`1girl` + `1boy` 는 유지 —
  그룹은 독립적).
- 어떤 실패(의존성 누락, 모델 사용 불가, 추론 오류)에서도 입력
  텍스트를 그대로 폴백 반환합니다 — 절대 하드 에러를 내지 않습니다.

### 지원 환경 및 요구사항

| | |
| --- | --- |
| **OS** | Windows / Linux / macOS — ComfyUI가 동작하고 `llama-cpp-python` 휠이 존재하는 모든 환경(x86-64, Apple Silicon/arm64) |
| **노드 1 (Seasonal)** | 모든 ComfyUI 설치. 의존성 없음, 모델 없음. |
| **노드 2 (Gemma TIPO)** | CPU 전용 동작(GPU 불필요); 속도를 위해 GPU 선택 |
| **RAM** | 최소 ~8 GB (4-bit 모델에 ≈5 GB 사용) · 16 GB 이상 권장 |
| **디스크** | 텍스트만: ~4 GB(3.2 GB GGUF + 의존성). 비전 모드 사용 시: ~8.5 GB(비전 페어 추가 4.4 GB). |
| **속도** | CPU 추론은 동작하지만 느림(프롬프트 하나에 수 분 소요 가능); 속도를 위해 GPU 가속 `llama-cpp-python` 빌드 권장 |
| **Python** | ComfyUI 내장 환경; `llama-cpp-python >= 0.3.23`(자동 설치/업그레이드, 1회 재시작 안내) |

### 설치 (노드 2 전용)

**수동 설치가 필요 없습니다.** 노드 2를 실제로 처음 실행할 때
의존성이 활성 환경에 1회 자동 설치됩니다: `tipo-kgen`(순수 파이썬)과
`llama-cpp-python` **>= 0.3.23**(프리빌트 CPU 휠 인덱스에서, 네이티브
빌드 없음). 너무 오래된 빌드는 gemma4 아키텍처를 로드할 수 없어
기존의 너무 오래된 `llama-cpp-python` 은 자동 업그레이드됩니다 — 세션
중간에 그렇게 되면 ComfyUI가 **1회 재시작**을 요청합니다(C 확장은
핫 리로드 불가). 노드 1은 아무것도 설치하지 않습니다.

대신 미리 수동 설치하려면(선택):

```
pip install -r requirements.txt
```

프리빌트 `llama-cpp-python` 휠 설치가 사용자 플랫폼/Python 에 맞는
휠을 찾지 못하면 직접 1회 설치하세요:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

## 레퍼런스 & 크레딧

| 프로젝트 | 용도 | 라이선스 |
| --- | --- | --- |
| KohakuBlueleaf 의 [KGen / TIPO](https://github.com/KohakuBlueleaf/KGen) | 태그 분류 및 프롬프트 정렬을 위한 `kgen.formatter` | Apache-2.0 |
| KohakuBlueleaf 의 [z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension) | TIPO 노드 흐름의 레퍼런스 구현; 학습 데이터 계보 | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF 추론 엔진; 비전 모드용 프리빌트 `llama-mtmd-cli` | MIT |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | in-process GGUF 추론을 위한 Python 바인딩 | MIT |
| Google 의 [Gemma](https://ai.google.dev/gemma) ([`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)) | TIPO GGUF가 파생된 베이스 모델 패밀리 | Gemma Terms of Use |
| [`p-e-w/gemma-4-E2B-it-heretic-ara`](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara) | 기본 한국어 TIPO GGUF를 파인튜닝한 디센서드 gemma-4-E2B(과도한 검열이 노골적 Danbooru 태그 출력을 망가뜨리지 않도록 사용) | Gemma Terms of Use |
| Unsloth 의 [`unsloth/gemma-4-E2B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | base gemma-4-E2B `mmproj-F16.gguf` — 비전-v1 외 모델용 레거시 폴백(≤2.7.3 동작)만 사용 | Gemma Terms of Use |
| [`raspie/gemma4-tipo-ko-gguf`](https://huggingface.co/raspie/gemma4-tipo-ko-gguf) | 노드 2의 기본 텍스트 TIPO 모델 **및** 학습된 `Gemma-tipo-vision-v1` 페어 자동 다운로드 | Gemma Terms of Use |

TIPO 확장 개념(짧은 프롬프트를 상세한 Danbooru 태그 세트로
"업샘플"/확장)은 KohakuBlueleaf 의 KGen/TIPO 와
[z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension)
에서 비롯되었으며, 기본 모델의 파인튜닝 학습 데이터도 같은 계보를
따릅니다.
