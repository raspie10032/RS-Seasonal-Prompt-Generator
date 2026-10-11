# VRS-INT 이미지·텍스트 쌍 구성 예제

[English](README.md) | 한국어 | [VRS-INT 안내](../../vrs_int/README_KOR.md) | [전체 사용법](../../README_KOR.md)

**자신의 이미지와 태그**로 개인 참고 기억을 준비하는 예제입니다. 가상 작가·캐릭터·
작품명과 `g` 등급만 사용하며 **이미지와 초기 기억은 제공하지 않습니다.** 실제 사용하는
이미지에 맞춰 태그를 수정하세요. 가상 예제 자체는 입력 형식을 설명하는 자료입니다.

이미지 한 장에 기록 하나를 대응시킵니다. 추천에는 구조화된 태그를 사용하고 자연어
설명문은 원본 검토용으로 보존합니다. 설명문을 임베딩·태그 변환하거나 인코더를
재학습하는 과정은 아닙니다.

## 시작 전

- Python 3.11 이상을 사용하고 저장소 최상위에서 명령을 실행합니다.
- 입력 준비·검사는 표준 라이브러리만 사용합니다. 구축·조회는 [CUDA·인코더 준비](../../vrs_int/README_KOR.md#준비)가 필요합니다.
- 개인 자료는 저장소 밖 또는 Git 제외 폴더인 `vrs-int-data/`에 둡니다.
- 아래 **JSONL 직접 작성**과 **이미지별 JSON 작성** 중 하나를 선택합니다. 최종 입력 형식은 같습니다.

## 방법 A: JSONL 직접 작성

이미지 기준 폴더를 정합니다. 예를 들어:

```text
vrs-int-data/
  pairs/
    1001.png
    1002.jpg
    1003.webp
  batch-01.jsonl
  batch-02.jsonl
  query.png
```

저장소의 [`batch-01.jsonl`](batch-01.jsonl)은 첫 두 기록입니다. 개인 폴더로 복사한 뒤
이미지 이름과 태그를 수정하세요. [`batch-02.jsonl`](batch-02.jsonl)은 다음 묶음에서
기존 기록을 수정하고 새 기록을 추가하는 예제입니다.

아래는 읽기 쉽게 펼친 한 기록입니다. **`.jsonl` 안에서는 객체 전체를 한 줄에 넣고**
바깥 배열이나 기록 사이 쉼표는 사용하지 않습니다. UTF-8로 저장하며 UTF-8 BOM도
입력에서 허용합니다.

```json
{
  "id": 1001,
  "image": "1001.png",
  "rating": "g",
  "tags_general": ["1girl", "solo", "blue_hair", "long_hair", "white_dress", "standing", "flower_field"],
  "tags_artist": ["example_artist_a"],
  "tags_character": ["example_character_a"],
  "tags_copyright": ["example_series_a"],
  "tags_meta": ["highres", "signature"],
  "caption": "푸른 머리의 인물이 꽃밭에서 흰 드레스를 입고 서 있다.",
  "split": "train"
}
```

### 필드 설명

| 필드 | 필수 여부 | 규칙과 용도 |
| --- | --- | --- |
| `id` | 필수 | 양의 JSON 정수. 같은 묶음에서 중복 불가. `danbooru:1001`이라는 고정 키가 됨. `"1001"` 같은 문자열은 불가 |
| `image` | JSONL에서 필수 | `--images` 기준 `/` 구분 상대 경로. 예: `spring/1001.png`. 절대 경로나 이미지 폴더 밖 접근 불가 |
| `rating` | 필수 | `g` 일반, `s` 민감, `q` 의심, `e` 노골적 등급 중 하나. 프로그램이 추론하지 않음 |
| `tags_general` | 필드 자체는 선택 | 의상·색·자세·배경 등 일반 태그. 비어 있지 않은 문자열의 배열 |
| `tags_artist` | 선택 | 작가명 배열. 명시적 분류 이름은 한 번 등장해도 보호 |
| `tags_character` | 선택 | 캐릭터명 배열. 수정 시에도 유지할 이름을 모두 기재 |
| `tags_copyright` | 선택 | 작품·시리즈명 배열. 자료원의 태그 분류이며 이용허락 정보가 아님 |
| `tags_meta` | 선택 | 메타 태그. 파일·품질·워터마크 등 불필요 태그는 사유를 남기고 추천에서 제외될 수 있음 |
| `tags` | 선택 | 분류 없는 태그 배열. 분류 필드와 합쳐 사용. 이름은 분류 필드 사용 권장 |
| `caption` | 선택 | 언어 제한 없는 문자열. 원본에 보존하며 임베딩·자동 태그 변환하지 않음 |
| `split` | 선택 | 기본 `train`, 또는 `val`·`test`. `train`만 추천 이웃으로 사용. 제출한 이미지는 모두 인코딩 |
| `image_sha256` | 선택 | 이미지 파일의 정확한 SHA-256. 제공하면 일치해야 하며 생략하면 계산 |
| 기타 JSON 필드 | 선택 | 원본 메타데이터로 보존. 자동 해석하지 않으므로 나중에 내보낼 자료에는 사적 정보 포함 여부 확인 |

지원하는 태그 배열을 합쳐 **최소 하나의 채택 태그**가 있어야 합니다. caption만 있는
기록, 모든 태그가 비어 있거나 전부 제외되는 기록은 거부합니다.
`"tags": "blue_hair long_hair"`는 잘못된 형식이며
`"tags": ["blue_hair", "long_hair"]`처럼 배열을 사용합니다. 단보루의 `tag_string*`
필드도 직접 입력으로 대체하지 못하므로 이 배열 형식으로 변환해야 합니다.

로컬에서 ID를 부여해도 현재 키 공간은 `danbooru:`입니다. ID를 안정적으로 유지하고
서로 다른 자료원의 같은 숫자 ID를 섞지 마세요. 파일명은 ID와 같을 필요는 없으며
올바른 파일을 가리키면 됩니다.

조회 태그는 소문자화·밑줄의 공백 변환·중복 제거를 거칩니다. `blue_hair`는 `blue hair`가
되지만 원본 문자열도 보존합니다. 이름은 명시적 분류로 보호합니다. `tags`에만 넣은
이름은 작가·캐릭터·작품별 후보 묶음에 올바르게 나타난다고 보장할 수 없습니다.

## 방법 B: 이미지 옆에 JSON 두기

JSONL을 한 줄씩 직접 만들기 번거로울 때 사용합니다. **이미지 전체 파일명 뒤에
`.json`**을 붙입니다. `1001.json`이 아니라 `1001.png.json`입니다.

```text
vrs-int-data/pairs/
  1001.png
  1001.png.json
  1002.jpg
  1002.jpg.json
```

위에서 펼친 JSON을 `1001.png.json`으로 저장합니다. `image` 필드는 생략할 수 있고
도구가 파일명에서 자동으로 채웁니다. 직접 넣으면 상대 파일명과 정확히 같아야 합니다.
`1002.jpg.json`에도 그 이미지의 ID와 전체 태그를 작성합니다. 이렇게 이미지별 JSON은
여러 줄로 작성해도 됩니다.

다음 명령으로 JSONL을 만듭니다.

```bash
python examples/vrs-int/make_manifest.py --pairs ./vrs-int-data/pairs --output ./vrs-int-data/batch-01.jsonl
```

도구는 `--pairs` 하위의 **모든 `*.json`을 재귀적으로 읽고** 검사한 뒤 새 `.jsonl`을
만듭니다. 무관한 JSON을 같은 폴더에 두지 마세요. 이미지·모델을 수정하거나 다운로드·
추론하지 않습니다. 설명 JSON이 없는 이미지는 포함하지 않습니다. 출력은 기존 파일을
덮어쓰지 않습니다. 완성된 파일 게시에 같은 폴더의 하드 링크를 사용하므로, 네트워크·FAT
볼륨에서 거부되면 로컬 NTFS·ext4·Btrfs 등 하드 링크 지원 파일시스템을 사용하세요.

후속 묶음에는 해당 기록만 가진 별도 sidecar 폴더를 사용하거나 JSONL을 직접
작성하세요. 전체 sidecar 폴더로 다시 만들면 모든 기록을 다시 제출하므로 전 이미지가
재인코딩됩니다.

## 입력 검사

```bash
python -m vrs_int check-input --manifest ./vrs-int-data/batch-01.jsonl --images ./vrs-int-data/pairs
```

저장소 예제를 직접 검사하려면 `--manifest`를
`./examples/vrs-int/batch-01.jsonl`로 바꿉니다. 이때도 `--images` 폴더에는 실제
`1001.png`, `1002.jpg` 파일이 있어야 합니다.

첫 묶음의 내용을 그대로 사용해 검사에 성공하면:

- `records`: 2, `ratings`: `g` 2건, `splits`: `train` 2건.
- `typed_unique_tags`: 작가 2종, 캐릭터 1종, 작품 1종.
- `removed_tag_occurrences`: 3회(`highres` 2회, `signature` 1회).
- `input_manifest_sha256`: JSONL 파일 바이트의 해시. 서식만 바꿔도 바뀝니다.

메타데이터·파일 존재·경로·해시 검사이며 이미지 디코딩이나 의미 검증은 아닙니다.
임의 파일의 확장자를 `.png`로 바꾸면 이 단계는 통과해도 실제 인코딩은 실패할 수 있습니다.

## 구축과 조회

[인코더·CUDA 준비](../../vrs_int/README_KOR.md#준비)를 마친 뒤 `GPU-YOUR-UUID`를
선택한 카드의 전체 UUID로 바꿉니다.

```bash
python -m vrs_int build --manifest ./vrs-int-data/batch-01.jsonl --images ./vrs-int-data/pairs --encoder ./vrs-int-models/gemma4-e2b-vision --store ./vrs-int-data/memory --gpu GPU-YOUR-UUID --label personal-01
```

반환된 64자리 `snapshot` 해시를 보관하고 아래 `SNAPSHOT_1`과 바꿉니다.
새 참고 이미지를 `query.png`로 준비합니다.

```bash
python -m vrs_int verify --store ./vrs-int-data/memory --snapshot SNAPSHOT_1
python -m vrs_int recommend --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --encoder ./vrs-int-models/gemma4-e2b-vision --image ./vrs-int-data/query.png --gpu GPU-YOUR-UUID --ratings g,s --neighbors 32 --limit 20 --output ./vrs-int-data/recommendation-01.json
```

조회 이미지에 미리 태그를 붙일 필요는 없습니다. 기억 속 파일을 그대로 조회하면
자기 자신은 제외하므로 다른 참고 이미지로 일반 검색 흐름을 확인하세요. 2장짜리
예제는 사용 흐름을 보여줄 뿐 추천 품질을 입증하지 않습니다. 결과는 이웃의 투표로
만든 참고 후보이며 원작자 확률이 아닙니다.

## 새 이미지 추가와 태그 수정

두 번째 예제 묶음의 변화는 다음과 같습니다.

| ID | 변경 | 이번 묶음에 필요한 이미지 |
| --- | --- | --- |
| `1001` | 작가·캐릭터·작품을 유지한 전체 기록으로 대체하고 `blue_sky` 추가 | `1001.png` |
| `1003` | 다른 가상 이름의 새 사례 추가 | `1003.webp` |
| `1002` | 변경 없음. 이번 묶음에서 생략 | 갱신에 이전 원본 불필요 |

[`batch-02.jsonl`](batch-02.jsonl)을 개인 폴더로 복사·수정한 뒤 실행합니다.

```bash
python -m vrs_int check-input --manifest ./vrs-int-data/batch-02.jsonl --images ./vrs-int-data/pairs
python -m vrs_int build --manifest ./vrs-int-data/batch-02.jsonl --images ./vrs-int-data/pairs --encoder ./vrs-int-models/gemma4-e2b-vision --store ./vrs-int-data/memory --gpu GPU-YOUR-UUID --base SNAPSHOT_1 --label personal-02
```

새 판본에는 4건이 아닌 **3건의 관측**이 있습니다. `1001`은 수정되며 별도 투표로
추가되지 않습니다. 새 해시를 `SNAPSHOT_2`로 보관하고 다음 조회에 직접 사용합니다.
이전 snapshot도 계속 읽을 수 있습니다.

**수정은 기록 전체 대체입니다.** `blue_sky`만 보내면 새 기록에서 기존 태그·이름이
빠집니다. 현재 유지할 전체 태그, 설명문, 등급, split을 모두 넣어야 합니다. CLI는
태그만 바꿔도 제출한 이미지를 다시 인코딩하므로 수정 이미지도 보관해야 합니다.

묶음마다 다른 이미지 폴더를 써도 됩니다. `--images`에는 이번 묶음의 파일만 있으면
됩니다. 저장된 상대 경로는 출처 정보이며 추천 시 과거 이미지를 다시 열지 않습니다.
추후 인코더 변경으로 특징을 재추출하려면 원본은 별도로 보관하세요.

## 보관할 것과 공개할 것

| 자료 | 보관 목적 | 기억 export에 포함? |
| --- | --- | --- |
| 원본 이미지 | 재추출·직접 시각 검토 | 아니오 |
| JSONL·sidecar | 직접 읽는 원본 태그·설명·수정 | 원본 기록 내용은 압축 기억에 보존. 별도 파일 자체를 복사하지는 않음 |
| 기억 폴더 | 조회·정확한 판본 이력·갱신 | 선택한 snapshot의 필요 객체만 export. 전체 이력은 별도 백업 필요 |
| 정확한 인코더 폴더 | 호환되는 구축·조회 | 아니오 |
| 새 조회 이미지 | 현재 조회·나중 재현 | 자동으로 기억에 추가하지 않음 |

공개 저장소에는 코드·문서·가상 서식만 둡니다. 개인 이미지 쌍, 생성한 JSONL, 기억
패키지, 추천 결과, 가중치를 커밋하지 마세요. 기억을 압축해도 출처 태그와 사적
메타데이터가 사라지지 않습니다. 이전·되돌리기·오류 처리는 [전체 VRS-INT 안내](../../vrs_int/README_KOR.md)를 참고하세요.
