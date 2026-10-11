# VRS-INT: 개인 이미지·태그 기억

[English](README.md) | 한국어 | [전체 사용법](../README_KOR.md) | [쌍 구성 예제](../examples/vrs-int/README_KOR.md)

개인 이미지와 분류된 태그로 판본이 있는 기억을 구성하고, 새 참고 이미지로
**새로운 프롬프트에 쓸 재료**를 검색합니다. 기존 Gemma E2B 비전 인코더가 특징을
만들고 VRS-INT가 관측과 태그를 저장·회수합니다. 구축은 **인코더 가중치 학습이 아닙니다.**

**코드·가상 예제만 배포합니다.** 자료·초기 기억·벡터·가중치는 직접 준비합니다.
이 독립 저장·검색 경로는 네이티브 VRS 판정 코어를 호출하지 않습니다. WD14 같은
분류 모델이나 원작자 식별 서비스도 아닙니다. 자연어 프롬프트 작성기는 별도입니다.

## 목차

- [준비](#준비)
- [최초 구축과 추천](#최초-구축과-추천)
- [추천 결과 읽기](#추천-결과-읽기)
- [자료 추가와 수정](#자료-추가와-수정)
- [백업과 이전, 되돌리기](#백업과-이전-되돌리기)
- [저장 구조와 자원, 호환성](#저장-구조와-자원-호환성)
- [명령어 참고](#명령어-참고)
- [문제 해결](#문제-해결)

## 준비

### 1. 코드와 별도 Python 환경

모든 명령은 `vrs_int/` 안이 아닌 **저장소 최상위 폴더**에서 실행합니다.
ComfyUI 설치나 이 저장소 자체의 패키지 설치는 필요 없습니다.

```bash
git clone https://github.com/raspie10032/RS-Seasonal-Prompt-Generator.git
cd RS-Seasonal-Prompt-Generator
```

Linux에서 Python 3.11 이상을 사용합니다.

```bash
python3 -m venv .venv-vrs
source .venv-vrs/bin/activate
python -m pip install --upgrade pip
```

Python 3.11이 설치된 Windows PowerShell:

```powershell
py -3.11 -m venv .venv-vrs
.\.venv-vrs\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

활성화가 어려우면 `python` 대신 `.venv-vrs/bin/python` 또는
`.\.venv-vrs\Scripts\python.exe`를 직접 실행해도 됩니다. 실행 정책을 바꿀 필요는
없습니다. 아래 작업 명령은 한 줄씩 작성되어 있어 자리표시자를 바꾸면 Bash와
PowerShell에서 사용할 수 있습니다.

| 작업 | 의존성 |
| --- | --- |
| `check-input`, 이미지별 설명 파일 준비, `inspect`, 명시적 인코더 다운로드 | Python 표준 라이브러리 |
| 벡터가 있는 기억의 verify·저장·export·install·activate, 오프라인 테스트 | NumPy. GPU·모델 불필요 |
| `build`, `recommend` | CUDA PyTorch, NumPy, Pillow, Gemma4 PIL 지원 Transformers, Accelerate, bitsandbytes, safetensors, 호환 NVIDIA GPU |

먼저 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서
운영체제·드라이버에 맞는 CUDA 빌드를 선택합니다. 그 명령을 적용할 때도 현재 가상
환경의 `python -m pip`를 사용하세요. 이어서 실행합니다.

```bash
python -m pip install numpy pillow transformers accelerate bitsandbytes safetensors
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from transformers import Gemma4VisionModel; from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil; print('Gemma4 PIL API available')"
```

두 번째 명령은 `CUDA: True`, 세 번째는 두 클래스 import 성공이어야 합니다.
해당 API를 제공하는 Transformers 버전이 필요합니다. 최상위 `requirements.txt`는
TIPO용이므로 이것만으로 준비되지 않습니다. [Gemma4 공식 문서](https://huggingface.co/docs/transformers/model_doc/gemma4)와
[bitsandbytes 호환 안내](https://huggingface.co/docs/bitsandbytes/installation)를 참고하세요.
macOS 오프라인 CI 통과가 GPU 실행 지원을 의미하지는 않습니다. 이 어댑터에는
CPU·MPS·다른 GPU 백엔드로의 대체 실행이 없습니다.

### 2. 비전 전용 인코더 준비

다음은 **명시적으로 모델을 내려받는 단계**이며 추론은 실행하지 않습니다.

```bash
python -m vrs_int.prepare_encoder ./vrs-int-models/gemma4-e2b-vision
```

출력 폴더는 **아직 존재하지 않는 경로**여야 합니다. 도구는 HTTP 바이트 범위 요청으로
연속된 시각 인코더 텐서만 가져오며 전체 언어모델을 내려받지 않습니다. 가중치 출처는
`p-e-w/gemma-4-E2B-it-heretic-ara`의 고정 리비전
`c9a1d4c031981f14d86eeb0c7d87de7fafd34513`입니다. 전처리 설정은
`unsloth/gemma-4-e2b-it`에서 준비 시점의 리비전을 확인한 뒤 기록합니다.

다음 폴더 전체를 함께 보관합니다.

```text
vrs-int-models/gemma4-e2b-vision/
  model.safetensors
  config.json
  preprocessor_config.json
  provenance.json
```

**`mmproj.gguf`가 아닌 별도 시각 인코더**입니다. 실행 시 선형층 NF4, BF16 연산,
PIL 전처리, 토큰 평균과 정규화를 사용해 768차원 특징을 만듭니다. `build`·`recommend`는
로컬 파일을 검증하며 없는 모델을 자동으로 받지 않습니다. 모델에는 코드와 별도
이용 조건이 적용됩니다.

나중의 추가·조회에도 이 폴더를 그대로 사용하세요. 준비 명령을 다시 실행하면 전처리
설정 리비전이 달라질 수 있어, 기반 가중치가 같아도 자동 호환된다고 볼 수 없습니다.

### 3. 사용할 GPU 지정

```bash
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
```

사용할 카드의 전체 `GPU-...` UUID를 복사해 아래 명령의 **`GPU-YOUR-UUID`**와
바꿉니다. 여러 GPU가 있으면 의도한 카드를 명확하게 선택할 수 있습니다.
`build`·`recommend`는 CUDA가 없으면 실패하며 CPU 추론으로 재시도하지 않습니다.

## 최초 구축과 추천

### 1. 실제 이미지와 설명 자료 준비

[이미지·텍스트 쌍 예제](../examples/vrs-int/README_KOR.md)를 먼저 준비하세요.
로컬 배치 예시는 다음과 같습니다.

```text
vrs-int-data/
  pairs/
    1001.png
    1002.jpg
  batch-01.jsonl
  query.png
```

저장소 예제는 가상 이름과 `g` 등급만 사용하며 **이미지 파일은 제공하지 않습니다.**
자기 이미지와 그에 맞는 태그를 준비하세요. JSONL 한 줄이 기록 하나이며, 이미지
경로는 JSONL 위치가 아닌 `--images` 폴더 기준입니다. 조회용 `query.png`는 기억에
없는 이미지여도 되며 별도 태그가 필요 없습니다.

### 2. 모델 실행 전 입력 검사

```bash
python -m vrs_int check-input --manifest ./vrs-int-data/batch-01.jsonl --images ./vrs-int-data/pairs
```

ID·태그 필드·등급·경로·파일 해시·채택 태그를 검사합니다. 픽셀을 디코딩하거나
태그가 그림과 맞는지 검증하지는 않습니다. 모델·GPU를 로드하지 않으므로 보고된
줄 번호의 문제를 먼저 수정하세요.

### 3. 기억 구축

```bash
python -m vrs_int build --manifest ./vrs-int-data/batch-01.jsonl --images ./vrs-int-data/pairs --encoder ./vrs-int-models/gemma4-e2b-vision --store ./vrs-int-data/memory --gpu GPU-YOUR-UUID --label personal-01
```

출력 JSON의 `snapshot`은 **64자리 SHA-256 내용 해시**입니다. `observations`,
`vectors`, `source_records`, `snapshot_bytes`에서 구축된 판본의 규모를 확인합니다.
출력과 해시를 보관하세요. 아래 **`SNAPSHOT_1`은 실제 해시로 바꿔야 합니다.**
괄호를 붙이거나 `personal-01` 같은 label을 넣지 않습니다.

```bash
python -m vrs_int verify --store ./vrs-int-data/memory --snapshot SNAPSHOT_1
python -m vrs_int inspect --store ./vrs-int-data/memory --snapshot SNAPSHOT_1
```

`verify`는 참조한 모든 객체를 읽어 해시·개수를 검사합니다. `inspect`는 인코더 호환
조건을 포함한 판본 정보를 보여줍니다. 원본 태그 전체 열람이나 검증을 대신하지 않습니다.

### 4. 새로운 이미지로 추천

```bash
python -m vrs_int recommend --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --encoder ./vrs-int-models/gemma4-e2b-vision --image ./vrs-int-data/query.png --gpu GPU-YOUR-UUID --ratings g,s --neighbors 32 --limit 20 --output ./vrs-int-data/recommendation-01.json
```

출력 UTF-8 JSON을 편집기로 엽니다. 기존 출력 파일은 덮어쓰지 않으므로 매번 새 이름을
사용합니다. `--ratings g,s`는 **검색할 기억 속 이웃의 등급**을 제한합니다.
`g,s,q,e`를 모두 지원하고 생략 시 네 등급 전부가 기본값입니다. 조회 이미지 등급
판정이나 출력 태그 정화 기능은 아닙니다. 쉼표 사이에는 공백을 넣지 마세요.
`--neighbors`는 1~256, `--limit`은 1~100입니다.

## 추천 결과 읽기

| 필드 | 의미 |
| --- | --- |
| `snapshot`, `query_image_sha256` | 사용한 기억과 입력 이미지의 정확한 판본 |
| `contract_sha256` | 이웃이 있을 때 반환하는 인코더·저장 호환 조건 해시 |
| `candidates` | `--limit`개까지의 상위 태그. 투표 결과에 따라 이름 분류도 포함 |
| `identities.artist`, `.character`, `.copyright` | 명시적으로 분류된 작가·캐릭터·작품별 상위 3개 후보 |
| `score` | 유사도를 반영한 이웃의 태그 투표. **정확도나 정체성 확률이 아님** |
| `support` | 해당 태그를 가진 검색 이웃 수 |
| `observation_keys`, `image_ids` | 후보를 뒷받침하는 관측·이미지 ID |
| `neighbors` | 가까운 관측의 ID·코사인 유사도·등급·이미지 해시 |
| `visual_verdict: "U"` | 시각적 귀속은 증명되지 않음. 후보가 있어도 의도적으로 유지 |

벡터가 있고 `split=train`인 기록만 추천에 참여합니다. 입력과 SHA-256이 같은
이미지는 제외하고, 같은 파일인 이웃도 한 번만 셉니다. 리사이즈·재인코딩한 그림은
해시가 달라 같은 그림이라고 판별하지 못합니다. 이름 분류 필드가 없으면 이름을
추측해 작가·캐릭터·작품으로 분류하지 않습니다.

적합한 이웃이 없으면 목록은 비어 있고 판정은 `U`입니다. 목록이 있어도 의미상
확신도를 보장하는 문턱값은 없으므로 약한 후보일 수 있습니다. 몇 장으로 만든 기억은
연결 확인용이지 충분한 추천 범위를 갖췄다는 의미가 아닙니다.

프롬프트 작성 시에는 사용자가 지정한 작가·캐릭터·작품을 유지하고, 회수한 의상·배경·
구도 중 원하는 재료를 고릅니다. 추천 이름은 창작 참고 후보입니다. 선택한 태그를
ComfyUI나 별도 작성기에 복사하세요. 자동 문장 조합은 아직 연결하지 않았습니다.

## 자료 추가와 수정

`batch-02.jsonl`에 **새 기록 또는 수정할 완전한 기록만** 넣습니다. 해당 이미지도
준비한 후 검사하고 이전 해시를 `--base`에 지정합니다.

```bash
python -m vrs_int check-input --manifest ./vrs-int-data/batch-02.jsonl --images ./vrs-int-data/pairs
python -m vrs_int build --manifest ./vrs-int-data/batch-02.jsonl --images ./vrs-int-data/pairs --encoder ./vrs-int-models/gemma4-e2b-vision --store ./vrs-int-data/memory --gpu GPU-YOUR-UUID --base SNAPSHOT_1 --label personal-02
```

새 해시를 **`SNAPSHOT_2`**로 보관하고 다음 조회에 직접 지정합니다. label은 설명용이며
자동 이동하는 별칭이 아닙니다. 구축은 이전 판본을 덮어쓰거나 새 판본을 활성화하지 않습니다.

- 같은 ID(`danbooru:<양의 정수>`)는 기존 관측 수정, 새 ID는 추가입니다.
- 수정은 **전체 대체**입니다. 합칠 태그만 보내지 말고 작가·캐릭터·작품·일반 태그·설명문을 모두 다시 넣으세요.
- CLI는 태그만 수정해도 제출한 이미지를 다시 인코딩합니다. 변경하지 않는 이전 이미지는 필요 없지만 수정·추가 이미지 파일은 있어야 합니다.
- 같은 ID에 같은 내용을 다시 넣어도 관측·투표가 하나 더 늘지 않습니다. 저장 객체가 모두 같으면 기존 해시를 반환합니다. 경로·메타데이터 변경도 내용 변경에 해당합니다. ID가 다르면 같은 이미지도 별도 관측으로 남을 수 있지만 검색은 동일 이미지 해시를 한 번만 셉니다.
- 원래 ID를 안정적·고유하게 유지하세요. 현재 키 공간은 `danbooru:`이므로 다른 자료원의 같은 숫자 ID를 그대로 합치지 마세요.
- 인코더·벡터 codec·태그 정책·전처리를 호환되게 유지합니다. `--buckets`는 처음 구조를 정하며 갱신은 이전 구조를 상속합니다.

[두 묶음 예제](../examples/vrs-int/README_KOR.md#새-이미지-추가와-태그-수정)는
처음 2장, 기존 1장 수정, 신규 1장 추가를 보여줍니다. 수정 전 원본 기록은 이전
snapshot에 남고, 대체된 새 기록 안에 이력이 누적되는 구조는 아닙니다.

## 백업과 이전, 되돌리기

### 개인 백업

`memory/` 전체와 정확한 인코더 폴더를 보존합니다. 원본 이미지·메타데이터는 별도
백업하세요. 기억에는 벡터와 기록이 들어 있으며 **이미지 파일 자체는 없습니다.**
일상 추천에는 기억·호환 인코더·새 입력 이미지가 필요하고, 인코더 변경 후 다시
특징을 추출하려면 원본 이미지가 필요합니다.

### 전체 패키지와 증분 패키지

출력은 ZIP이 아닌 **디렉터리**이며 새 경로여야 합니다.

```bash
python -m vrs_int export --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --output ./vrs-int-data/package-01
python -m vrs_int export --store ./vrs-int-data/memory --snapshot SNAPSHOT_2 --base SNAPSHOT_1 --output ./vrs-int-data/package-02-delta
python -m vrs_int install --store ./vrs-int-data/restored --package ./vrs-int-data/package-01
python -m vrs_int install --store ./vrs-int-data/restored --package ./vrs-int-data/package-02-delta
python -m vrs_int verify --store ./vrs-int-data/restored --snapshot SNAPSHOT_2
```

증분에는 **정확한 직전 부모 판본**이 필요하며 먼저 설치해야 합니다. 임의 버전 건너뛰기나
다른 사람 기억과의 자동 병합은 지원하지 않습니다. `SNAPSHOT_2`를 전체 export하면
부모 없이 설치되지만 부모 이력은 포함되지 않습니다. 같은 저장소에서는 여러 판본이
공유하는 객체를 한 번만 저장합니다.

`export`는 원본 태그·caption·기록에 포함된 출처 정보를 담고, 원본 이미지·인코더는
담지 않습니다. **사람이 읽는 JSONL 태그 내보내기나 익명화 패키지가 아닙니다.**
코드만 공개한다면 개인 패키지도 게시하지 마세요. SHA-256은 내용 변화를 확인하지만
게시자 신원을 보증하지 않습니다. `install`은 검증·설치 후 자동 활성화하지 않습니다.

### 선택적 활성화와 되돌리기

CLI 추천은 언제나 `--snapshot`을 직접 지정하며 `CURRENT`를 자동으로 읽지 않습니다.
활성화 포인터는 외부 도구에서 사용할 때 선택적으로 관리합니다.

```bash
python -m vrs_int activate --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --expected-current none
python -m vrs_int activate --store ./vrs-int-data/memory --snapshot SNAPSHOT_2 --expected-current SNAPSHOT_1
python -m vrs_int activate --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --expected-current SNAPSHOT_2
```

마지막 명령은 포인터를 되돌립니다. CLI 조회도 되돌리려면 `--snapshot SNAPSHOT_1`을
사용하세요. `none`은 기존 포인터가 없을 때만 유효합니다. 예상 포인터가 달라졌으면
충돌로 처리해 다른 활성화를 조용히 덮어쓰지 않습니다. 자동 정리 기능은 없으므로
이전 객체·이력은 별도 관리 전까지 남습니다.

## 저장 구조와 자원, 호환성

```text
memory/
  objects/<sha256>.z       압축 태그 사전·기록·벡터·감사 자료
  snapshots/<sha256>.json 불변 판본 정보·호환 조건·부모·개수
  CURRENT                 선택적으로 활성화한 판본 해시
```

이름은 버킷별 사전의 ID로 저장하고 객체는 zlib로 압축합니다. 명시적으로 분류된
작가·캐릭터·작품은 한 번 등장해도 보호합니다. 일반 태그의 품질·해상도·워터마크 등은
이유를 기록하고 제외합니다. 조회 태그는 소문자화·밑줄의 공백 변환·중복 제거를 거칩니다.
원본 기록에는 원래 표기와 `caption`이 남고, 설명문을 임베딩하지는 않습니다.

`inspect`는 판본 정보만 보여줍니다. CLI 원본 태그 열람·JSONL 내보내기는 아직 없으므로
직접 읽으려면 입력 manifest·sidecar도 보관하세요. 압축 객체는 되돌릴 수 있는 기록이며
`q/e` 태그를 숨기는 수단이 아닙니다.

| 형식 | 용도와 한계 |
| --- | --- |
| `f32le` 기본값 | 생성된 FP32 벡터 유지. 직렬화·압축·메타데이터 전 768 × 4 = 이미지당 3,072바이트 |
| `i8` 선택값 | 손실 양자화. 이미지당 768개 코드와 스케일, 기타 저장 부가분. 이웃·이름 후보가 달라질 수 있음 |
| GPU 검색 | 두 형식 모두 FP32로 풀어 제한된 크기의 묶음으로 계산. `i8` 저장이 검색 VRAM 4분의 1을 뜻하지 않음 |

인코더 가중치 NF4와 저장 벡터 `i8`은 별개입니다. 비교하려면 서로 다른 저장소에
`--codec f32le`, `--codec i8`로 구축하세요. 기존 판본에 다른 codec을 덧붙이지 않습니다.
PQ·ANN 검색 인덱스는 구현하지 않았습니다.

CLI는 CPU 연산 스레드 2개, PyTorch interop 1개로 제한하며 지원하는 운영체제에서는
CPU affinity도 2개로 제한하고 우선순위를 낮춥니다. I/O·이미지 디코딩·압축·해시는
여전히 CPU를 사용합니다. **PyTorch CUDA 할당기 한도를 20%로 설정**하지만 GPU를
예약하거나 드라이버·라이브러리 전체 VRAM을 엄격히 제한하는 장치는 아닙니다.
구축·이관은 디스크 여유 20% 미만이면 시작하지 않습니다. 시작 검사이므로 완료 후
여유를 예측하지 않으며, 이전 판본·전체 export에도 추가 공간이 듭니다.

GPU에는 제한된 벡터 묶음만 올리지만, 추천 가능한 훈련 기록은 시스템 RAM에 캐시하고
전체를 순차 비교합니다. 첫 조회는 압축 자료를 읽어야 하고 자료가 커지면 RAM·시간도
늘어납니다. 백만 장 규모 성능은 검증하지 않았습니다. 작은 인코더만으로 일정한
VRAM·RAM 사용량이나 추천 정확도를 보장하지 않습니다.

호환 조건에는 인코더 출처 해시·전처리·PIL·NF4·pooling·차원·codec·태그 정책을
기록합니다. 변경 시 자동 병합 대신 별도 기반 판본·재추출이 필요합니다. 조건이 같아도
모든 라이브러리·드라이버 조합의 비트 단위 동일 결과를 보장하지는 않습니다.
중요한 기억과 함께 환경 버전도 보관하세요(`python -m pip freeze`).

## 명령어 참고

정확한 문법은 `python -m vrs_int --help`, `python -m vrs_int COMMAND --help`로 확인합니다.

| 명령 | 역할 |
| --- | --- |
| `check-input --manifest FILE --images DIR` | 모델 없이 입력·경로·해시 확인 |
| `build --manifest FILE --images DIR --encoder DIR --store DIR --gpu UUID` | 구축. 선택: `--base HASH`, `--codec f32le/i8`, `--label TEXT`, `--buckets N` |
| `recommend --store DIR --snapshot HASH --encoder DIR --image FILE --gpu UUID --output FILE` | 조회. 선택: `--ratings`, `--neighbors`, `--limit` |
| `inspect --store DIR --snapshot HASH` | 판본 정보 |
| `verify --store DIR --snapshot HASH` | 참조 자료·해시·개수 검사 |
| `export --store DIR --snapshot HASH --output DIR` | 전체 패키지. `--base PARENT_HASH` 추가 시 증분 |
| `install --store DIR --package DIR` | 검증·설치. 자동 활성화 없음 |
| `activate --store DIR --snapshot HASH --expected-current HASH_OR_none` | 명시적 포인터 변경·되돌리기 |
| `update --store DIR --snapshot HASH --operations FILE --contract FILE --label TEXT` | 사전 인코딩된 관측과 예상 revision으로 수정하는 고급 기능 |
| `migrate --memory DIR --corrected DIR --encoder DIR --store DIR` | 이전 실험 전용 구조 이관. 임의 데이터셋 입력 기능이 아님 |

일반 추가·수정은 **`build --base`**를 사용하세요. 고급 `update`는 UTF-8 JSONL
연산을 받습니다. `upsert`는 `key`·완전한 인코딩 `entry`·`expected_revision`이 필요하고
새 키에만 null revision을 사용합니다. `delete`는 기존 키·revision·사유가 필요합니다.
revision은 `store.revision(entry)`로 얻습니다. `--contract`에는 snapshot 전체가 아닌
생산자의 완전한 호환 조건 JSON을 넣습니다. 충돌 시 새 판본 게시 전에 중단합니다.
API는 [store.py](store.py)와 [저장 테스트](../tests/test_vrs_int.py)를 참고하세요.
`migrate`는 [import_snapshot.py](import_snapshot.py)의 내부 실험 구조를 요구하며,
그 실험 자료는 의도적으로 배포하지 않습니다.

## 문제 해결

| 오류·결과 | 해결 방향 |
| --- | --- |
| `No module named vrs_int` | 저장소 최상위에서 올바른 Python으로 실행 |
| `CUDA required; no CPU fallback` | CUDA PyTorch·드라이버·전체 GPU UUID 확인. CPU 대체 실행은 지원하지 않음 |
| Gemma4/PIL import 또는 NF4 오류 | 안내한 두 API와 bitsandbytes 호환성 확인. TIPO requirements만으로 부족 |
| `Encoder file or digest mismatch` | 원래의 변경되지 않은 인코더 폴더 복원. 출처 해시를 고쳐 검사 우회하지 않기 |
| `Query encoder incompatible` / `Incompatible update` | 원래 인코더·조건·codec 사용 또는 원본 이미지로 별도 구축 |
| 준비·export·추천·sidecar 준비에서 `FileExistsError` | 새로운 출력 경로 사용. 준비 실패 시 부분 폴더가 남을 수 있으므로 확인·보관 후 다른 경로에 재시도 |
| 인코더 다운로드 HTTP range 오류 | 서버·프록시가 범위 요청을 지원해야 함. 전체 언어모델 자동 다운로드로 대체하지 않음 |
| 잘못된 경로·이미지 누락·중복 ID | `--images` 내부의 `/` 구분 상대 경로 사용. 묶음 내 양의 정수 ID당 기록 하나 |
| 추천 목록이 비어 있음 | `train` 벡터·등급 필터·입력 이미지 자기 제외 확인. 벡터 없는 태그 기록만 늘려도 이미지 검색 사례는 늘지 않음 |
| 이름 후보가 없거나 부정확 | 분류 필드와 관련성 있는 다양한 이미지 필요. 빈도·점수는 정체성 정확도가 아님 |
| `Delta requires its exact immediate parent` | 실제 직전 부모로 증분 export하거나 전체 패키지 사용. 증분 전에 정확한 부모 설치 |
| `Active snapshot changed` | `CURRENT`를 읽고 의도한 해시 확인. 다른 활성화를 무작정 덮어쓰지 않기 |
| `Keep at least 20% disk free` | 충분한 여유가 있는 볼륨 선택. 임시 자료·보존 판본 공간도 고려 |
| `Bucket too large` | 더 큰 `--buckets`로 별도 기반 구축. 객체 압축 해제 한도 64 MiB |
| CUDA 메모리 부족·RAM 증가 | 다른 GPU 프로세스·할당기 한도·자료 크기 확인. 현재 CLI에 인코더 배치 크기·ANN 조절 옵션은 없음 |

## 개발 검증

```bash
python -m pip install numpy
python -m unittest discover -s tests -p "test_vrs_int*.py" -v
python scripts/check_code_only.py
```

가상 자료와 대체 인코더를 사용하며 GPU·모델 다운로드 없이 동작을 검사합니다.
시각적 추천 품질 검사는 아닙니다. 게시 검사는 Git 인덱스를 보므로 스테이징된 내용도
확인하세요. 개인 기억·패키지·가중치는 PR·릴리스에 포함하지 않습니다.
