# RS-Seasonal-Prompt-Generator

[English](README.md) | 한국어

계절별 CSV, 자연어 설명, 개인 이미지·태그 자료에서 이미지 생성용 프롬프트 재료를
만듭니다. **ComfyUI 노드 2개**와 독립 실행 도구 **VRS-INT**를 제공합니다.

**VRS-INT는 코드·문서·가상 예제만 공개합니다.** 초기 기억, 수집한 단보루 이미지와
태그, 특징 벡터, 추천 결과, 모델 가중치는 포함하지 않습니다. 각자 원하는 자료로
기억을 구성하며, 작가·캐릭터·작품명은 핵심 정보로 보존합니다.

## 원하는 기능부터 선택하기

| 원하는 작업 | 사용할 기능 | 필요한 것 |
| --- | --- | --- |
| 계절·의상·배경 무작위 조합 | Seasonal Fashion Prompt Generator | ComfyUI. 모델·추가 패키지 불필요 |
| 한국어·영어 설명이나 태그 확장 | Gemma TIPO 텍스트 모드 | ComfyUI + GGUF 실행 환경·모델. 공개 텍스트 경로는 CPU 사용 |
| 참고 이미지에서 프롬프트용 태그 생성 | Gemma TIPO 이미지 모드 | ComfyUI + GGUF·호환 mmproj·llama.cpp 비전 실행 파일 |
| 개인 자료에서 비슷한 그림의 태그·이름 회수 | VRS-INT | Python 3.11 이상, NVIDIA CUDA 환경, 별도 비전 인코더, 개인 이미지·태그 쌍 |

VRS-INT의 목적은 **새로운 프롬프트에 쓸 참고 재료를 가져오는 것**입니다. 인코더를
재학습하거나 원작자를 확정하는 기능은 아니며, 완성된 자연어 프롬프트를 작성하지는
않습니다. 반환된 태그를 이미지 프롬프트에 사용하거나 별도 생성기에 전달하세요.
현재 공개본에는 VRS-INT ComfyUI 노드·Writer 자동 연결이 없습니다.

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
두 노드의 출력은 `STRING`입니다. 텍스트 인코더에 연결할 때 ComfyUI 버전에 따라
프롬프트·텍스트 위젯을 입력으로 변환해야 합니다. 텍스트 미리보기 노드를 연결하면
이미지 생성 전에 문자열을 확인할 수 있습니다.

### 2. 선택 기능인 TIPO 준비

계절 생성기는 Python 표준 라이브러리만 사용하므로 바로 실행할 수 있습니다.
TIPO는 실제 생성 시 필요한 의존성 설치와 모델 다운로드를 시도합니다. 노드 패키지
로드만으로 모델을 내려받지는 않습니다. 미리 설치하려면 **ComfyUI의 Python**으로
저장소 폴더에서 실행하세요.

```bash
python -m pip install -r requirements.txt
```

Windows portable에서는 `ComfyUI_windows_portable` 폴더에서 실행합니다.

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\RS-Seasonal-Prompt-Generator\requirements.txt
```

의존성은 `tipo-kgen`, `llama-cpp-python>=0.3.23`입니다. 이미 로드된 구버전을
업그레이드했다면 ComfyUI를 재시작합니다. 이 파일은 **TIPO용**이며 VRS-INT의
CUDA·Transformers 환경을 설치하지 않습니다. 다운로드에는 인터넷과 저장 공간이
필요하며 유료 추론 API는 사용하지 않습니다.

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
| `seed` | 무작위 선택 시드. 아래 공개본 동작 참고 |

같은 선택을 반복하려면 `1234567890123456` 같은 **16자리 시드**를 사용하고
ComfyUI의 시드 변경 방식도 고정합니다. 현재 공개본은 `0`을 포함한 작은 시드를
정규화할 때 무작위 값이 개입합니다. 상의·하의는 둘 다 있어야 선택되는 구조라
하나만 켜면 해당 옷이 빠질 수 있습니다. 투피스는 상·하의를 켜고 원피스를 끄세요.
셋 다 있으면 투피스와 원피스 중 하나를 고릅니다.

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

쉼표가 들어가는 값은 CSV 따옴표로 감쌉니다. 상황 조건은 `|`로 나눈 값의 정확한
일치를 확인하며, 현재 공개본은 `&`를 AND 조건으로 해석하지 않습니다.

## Gemma TIPO: 텍스트 또는 이미지에서 태그 생성

![ComfyUI의 Gemma TIPO](assets/gemma-tipo-example.png)

### 텍스트 모드

1. **Gemma TIPO (Prompt / Image → Tags)**를 추가하고 `image`를 연결하지 않습니다.
2. 한국어·영어 설명 또는 쉼표 태그를 입력합니다. 예: `봄 공원, 흰 블라우스, 주름치마`.
3. `model=(auto: download default)`로 두면 `gemma4-tipo-ko-v2-Q4_K_M.gguf`를 첫 실행에 `ComfyUI/models/gguf/`로 내려받습니다.
4. `tag_length=long`, `sort=danbooru`, `temperature=0.5`부터 시작합니다.
5. 출력 태그를 확인한 뒤 사용합니다. 세부 내용이 추가·누락·변경될 수 있어 정확한 번역을 보장하지 않습니다.

**공개 텍스트 엔진은 `n_gpu_layers=0`으로 실행합니다.** GPU용 패키지를 설치하거나
노드의 `gpu_layers`를 바꾸는 것만으로 GPU로 이동하지 않습니다. CPU 부담이 크고
느릴 수 있습니다. VRS-INT의 CUDA 실행 경로와 구분하세요.

### 이미지 모드

![Gemma TIPO 이미지 모드](assets/gemma-tipo-vision-example.png)

**Load Image → Gemma TIPO.image**로 연결하고 출력을 미리보기나 텍스트 인코더로
보냅니다. 이미지를 연결하면 `prompt` 입력은 무시됩니다. 한 번에 한 장을 사용하세요.

`model`은 자동, `mmproj_path`는 빈칸이면 공개 기본값인 TIPO GGUF와 기반 Gemma E2B의
`mmproj-F16.gguf`를 사용합니다. 첫 실행에서 플랫폼별 llama.cpp `b9209` 실행 파일도
`ComfyUI/models/tipo_bin/`에 준비합니다. **VRS-INT의 별도 인코더와 다릅니다.**

공개 실행 파일 선택기는 일반·CPU 릴리스를 선택하며 CUDA 자동 선택은 구현하지
않았습니다. 호환 GPU 비전 실행 파일을 준비했다면 ComfyUI 시작 전 `LLAMA_MTMD_CLI`에
전체 경로를 지정하고 실행 라이브러리를 준비한 뒤 `gpu_layers`를 양수로 설정합니다.
실제 GPU 적재는 실행 파일·하드웨어에 달려 있습니다. 공개 위젯은 `0`~`100`을 받으며
`-1`은 지원하지 않습니다.

이미지 모드는 그림을 참고로 프롬프트 재료를 확장하므로 그림에 없는 내용도 나올 수
있습니다. 모든 출력 태그를 검증된 이미지 설명으로 취급하지 마세요.

### TIPO 입력 설정

| 입력 | 동작 |
| --- | --- |
| `prompt` | 한국어·영어 설명 또는 태그. 텍스트 모드용 |
| `model` | 자동 기본값 또는 `ComfyUI/models/gguf/`의 `.gguf`. 파일 추가 후 새로고침·재시작 |
| `tag_length` | `very_short`, `short`, `long`, `very_long`. 정확한 개수 제한은 아님 |
| `sort` | `danbooru`, `quality_first`, `artist_first`, `general_only`, `simple` |
| `temperature`, `seed` | 생성 변동 조절. 환경이 다르면 같은 시드도 결과가 달라질 수 있음 |
| `ban_tags` | 쉼표로 구분한 제외 태그. 분류 후처리가 정상 수행됐을 때 적용 |
| `image` | 연결하면 이미지 모드 |
| `mmproj_path` | 이미지 모드용. 빈칸은 기본값, 직접 지정하면 호환 프로젝터 필요 |
| `gpu_layers` | 공개본은 이미지 모드에만 적용. 기본 `0`, 양수는 GPU 적재 요청 |

`general_only`는 special·캐릭터·일반 태그를 남겨 작가·작품 분류가 빠집니다.
`simple`은 special·등급·일반 태그만 남깁니다. **작가·캐릭터·작품을 후처리에서도
유지하려면 `danbooru` 또는 `artist_first`를 사용하세요.**
TIPO는 수 GB의 모델 저장 공간과 실행 메모리를 사용하며 이미지는 프로젝터·실행
파일도 필요합니다. 작은 VRS-INT 인코더 경로의 요구량과는 별개입니다.

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

## 문제 해결

| 증상 | 확인할 것 |
| --- | --- |
| 노드가 안 보임 | 폴더 중첩, 재시작, 시작 로그 import 오류, `prompt` 분류. VRS-INT는 CLI라 노드 목록에 없음 |
| TIPO가 원문 그대로 반환 | 의존성·모델·추론 오류 시 원문 반환 가능. `[TIPO]` 로그 확인 |
| 이미지 출력이 빈 문자열 | `[TIPO-V]` 로그, 모델·프로젝터 호환성, 실행 라이브러리, 이미지, 추론 제한 600초 확인 |
| `unknown model architecture: gemma4` | ComfyUI의 Python에서 Gemma4 지원 llama-cpp-python ≥0.3.23 확인. 업그레이드 후 재시작 |
| CPU가 계속 바쁨 | 공개 TIPO 텍스트는 CPU 실행. 비전도 CUDA 자동 설치가 아님. VRS-INT GPU 설정과 별개 |
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
python -m pip install numpy
python -m unittest discover -s tests -p "test_vrs_int*.py" -v
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
| [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | 공개 TIPO 기본 프로젝터 |
| [p-e-w/gemma-4-E2B-it-heretic-ara](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara) | TIPO 기반 계보, 별도 VRS-INT 시각 인코더 출처 |

처음 ChatGPT로 작성했고 이후 Claude Code와 Codex로 개발했습니다.
