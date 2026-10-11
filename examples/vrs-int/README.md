# VRS-INT: 이미지·텍스트 쌍으로 개인 기억 만들기

Gemma의 기존 비전 인코더는 그대로 사용하고, 각자가 선택한 이미지와 태그를
VRS-INT 기억으로 구성하는 예제입니다. 코드는 공유하고 자료와 인코더 가중치는
별도로 준비합니다. 여기의 ID·작가·캐릭터·작품은 **입력 형식 설명용 가상 값**입니다.
실제 단보루 게시물의 내용이나 작가 정보가 아니며, 이미지 파일은 포함하지 않습니다.
실사용 시에는 실제 게시물 ID와 그 이미지에 대응하는 메타데이터로 모두 교체하세요.

현재 키 형식은 `danbooru:<양의 정수 ID>`입니다. 단보루 외 자료에 임의 ID를
붙여 섞으면 충돌할 수 있습니다. 다른 출처의 ID 이름공간은 아직 지원하지 않습니다.

## 이미지와 텍스트가 연결되는 방식

```text
이미지 파일 ── 기존 비전 인코더 ── 특징 벡터 ─┐
                                          ├─ 같은 관측 ID로 저장
분류된 태그·등급·출처 정보 ────────────────────┘

새 이미지 ── 같은 인코더 ── 비슷한 관측 검색 ── 참고 태그·이름 후보
```

여기서 텍스트의 주된 입력은 **태그 배열**입니다. 이미지·태그 쌍을 추가해도 인코더의
가중치를 재학습하지 않습니다. 선택적 자연어 `caption`은 원본 메타데이터로 보존하지만
임베딩·태그 변환·검색 투표·Writer 전달에는 사용하지 않습니다. 설명문만 넣고 태그를
비우면 구축할 수 없습니다. 신규 프롬프트의 장소나 행동 지시는 자료의 사실 태그에
섞기보다, 이 기억을 사용하는 프롬프트 생성기에 별도로 전달하는 구성입니다.

## 1. 한 장에 한 기록

[batch-01.jsonl](batch-01.jsonl)은 2장으로 첫 기억을 구성하는 템플릿입니다.
**JSONL은 한 줄에 JSON 객체 하나**이며, 파일 전체를 `[...]`로 감싸지 않습니다.
설명을 위해 첫 줄을 펼치면 다음과 같습니다.

```json
{
  "id": 1001,
  "image": "1001.png",
  "rating": "g",
  "tags_general": ["1girl", "solo", "blue_hair", "white_dress", "flower_field"],
  "tags_artist": ["example_artist_a"],
  "tags_character": ["example_character_a"],
  "tags_copyright": ["example_series_a"],
  "tags_meta": ["highres", "signature"],
  "caption": "푸른 머리의 인물이 꽃밭에 서 있다.",
  "split": "train"
}
```

| 필드 | 입력 방법과 용도 |
|---|---|
| `id` | 필수. 실제 단보루 게시물의 양의 정수 ID. 한 묶음에서 중복 불가 |
| `image` | 필수. `--images` 기준 상대 경로. 하위 폴더 구분자는 `/` 사용 |
| `rating` | 필수. 출처의 `g`, `s`, `q`, `e` 중 하나를 그대로 기록. 등급 추정이나 자동 기본값 없음 |
| `tags_general` | 외형·의상·행동·장면 등 일반 태그의 문자열 배열 |
| `tags_artist` | 작가명 배열. 여러 작가가 명시된 자료면 모두 기록 |
| `tags_character` | 캐릭터명 배열. 여러 캐릭터도 모두 기록 |
| `tags_copyright` | 작품·시리즈명 배열. 원본 분류를 유지 |
| `tags_meta` | 원본의 메타 태그 배열. 정책에 해당하는 품질·서명 등의 태그는 추천에서 제외 |
| `tags` | 선택. 분류를 모르는 태그 배열. 다른 필드와 합쳐 사용하며 중복은 정규화 후 제거 |
| `caption` | 선택. 자연어 설명 문자열. 원본 기록으로만 보존 |
| `split` | 생략 시 `train`. `val`, `test`는 저장·인코딩하지만 추천 이웃에서 제외 |
| `image_sha256` | 선택. 이미지 파일의 소문자 SHA-256. 지정하면 실제 파일과 일치해야 함 |

태그는 `"blue_hair"`처럼 각 항목을 배열에 넣습니다. 공백 구분 문자열이나 쉼표 구분
문자열 하나는 받지 않습니다. 내부에서는 소문자화하고 `_`를 공백으로 바꿉니다.
원본 표기는 감사 기록에 남습니다. 분류가 불명확하면 이름을 추측해 `tags_artist`에
넣지 말고 `tags`에 둡니다. 별칭을 자동으로 같은 인물·작가로 합치는 기능은 없습니다.

작가·캐릭터·작품은 **한 번만 등장해도 보존**합니다. 이 세 필드의 명시적 이름은
불필요 태그 제외 규칙보다 우선합니다. `highres`, `signature` 같은 제외 태그도 원본과
제외 사유는 남습니다. 하나의 이미지에 인물이 여럿이면 전체 태그를 연결할 뿐,
누가 어느 옷을 입었는지에 대한 영역별 주석은 이 형식으로 표현하지 않습니다.

## 2. 직접 JSONL을 쓰거나, 이미지 옆에 JSON 파일 두기

이미지와 같은 이름에 `.json`을 붙인 설명 파일을 준비하면
[make_manifest.py](make_manifest.py)가 JSONL로 묶습니다. 복사해 수정할 설명 파일은
[pairs/1001.png.json](pairs/1001.png.json), [pairs/1002.jpg.json](pairs/1002.jpg.json)에 있습니다.

```text
vrs-int-data/
  batch-01/
    images/
      1001.png
      1001.png.json
      1002.jpg
      1002.jpg.json
    manifest.jsonl          ← 스크립트 출력
```

```bash
# 저장소 루트에서 실행. 실제 이미지·ID·태그를 먼저 준비합니다.
python examples/vrs-int/make_manifest.py \
  --pairs ./vrs-int-data/batch-01/images \
  --output ./vrs-int-data/batch-01/manifest.jsonl
```

스크립트는 하위 폴더의 `.json`도 읽으며 이미지 경로를 파일명에서 채웁니다.
설명 파일에 `image`를 직접 적었다면 해당 경로와 일치해야 합니다. 이 폴더에는
이미지 설명 JSON만 넣으세요. JSON 설명이 없는 이미지는 포함하지 않습니다.
메타데이터가 없는 `.txt`를 문장 분해하거나 이름 분류를 추측해 변환하지 않습니다.
단보루 API 응답을 그대로 입력하는 수집기 또한 아니므로 위 필드로 정리해야 합니다.

기존 출력은 덮어쓰지 않습니다. 모든 기록이 검사된 후 JSONL을 만들며, 이미지나
설명 파일을 수정하지 않습니다. 이 스크립트와 아래 입력 검사는 **Python 표준
라이브러리만** 사용하고 GPU·인코더·네트워크를 사용하지 않습니다.

## 3. GPU를 켜기 전에 검사

```bash
python -m vrs_int check-input \
  --manifest ./vrs-int-data/batch-01/manifest.jsonl \
  --images ./vrs-int-data/batch-01/images
```

기록 수, 등급·split별 수, 분류별 이름 수, 제외 태그 횟수, 입력 파일 해시를 출력합니다.
잘못된 필드, 중복 ID, 빈 태그, 누락 파일, 이미지 루트 바깥 경로, 지정 해시 불일치는
줄 번호와 함께 오류로 끝납니다. **픽셀 디코딩·이미지 손상 여부·태그 의미 일치성은
검사하지 않습니다.** 대규모 자료에서는 파일 해시를 읽는 만큼 디스크 I/O가 발생합니다.

## 4. 기존 인코더로 구축하고, 다음 묶음을 추가

GPU 라이브러리와 별도 인코더는 [VRS-INT 준비 문서](../../vrs_int/README.md#준비)를
따릅니다. `--encoder`는 Gemma 비전 타워를 분리한 폴더이며 `mmproj.gguf` 파일이
아닙니다. 기존 기억에 추가하거나 조회할 때는 같은 인코더·전처리 계약을 사용합니다.
`GPU-YOUR-UUID`는 `nvidia-smi -L`에서 확인한 사용할 NVIDIA GPU UUID로 바꿉니다.

```bash
python -m vrs_int build \
  --manifest ./vrs-int-data/batch-01/manifest.jsonl \
  --images ./vrs-int-data/batch-01/images \
  --encoder ./vrs-int-models/vision-encoder --store ./vrs-int-data/my-memory \
  --gpu GPU-YOUR-UUID --label my-first-batch
```

출력 JSON의 `snapshot`을 기록합니다. 다음 묶음은 [batch-02.jsonl](batch-02.jsonl)
템플릿처럼 구성할 수 있습니다. 예제는 **1001번 수정 + 1003번 신규 추가**입니다.
수정할 때는 추가 태그만 적지 말고 그 관측의 **전체 태그·이름·메타데이터**를
다시 적으세요. 새 기록이 이전 기록을 대체하므로, 빠뜨린 작가명은 새 판본에서 빠집니다.
수정 관측의 이미지도 다시 인코딩하므로 해당 파일이 필요합니다. 변경하지 않는
기존 1002번은 목록에 넣을 필요가 없고 원본 파일도 다시 읽지 않습니다.

```bash
python -m vrs_int check-input \
  --manifest ./vrs-int-data/batch-02/manifest.jsonl \
  --images ./vrs-int-data/batch-02/images

python -m vrs_int build \
  --manifest ./vrs-int-data/batch-02/manifest.jsonl \
  --images ./vrs-int-data/batch-02/images \
  --encoder ./vrs-int-models/vision-encoder --store ./vrs-int-data/my-memory \
  --gpu GPU-YOUR-UUID --base PREVIOUS_SNAPSHOT_HASH --label my-second-batch
```

이 예제에서는 총 관측이 2개에서 3개가 되고, 이전 snapshot은 그대로 남습니다.
한 묶음 안에서 같은 ID를 두 번 넣는 것은 오류입니다. 같은 이미지 파일의 다른 ID는
출처로 남지만 추천에서 중복 투표하지 않습니다. 원본을 지우는 명령은 제공하지 않으며,
다른 인코더로 특징을 재추출하려면 원본이 다시 필요합니다.

## 5. 새로운 이미지로 참고 태그 가져오기

```bash
python -m vrs_int recommend \
  --store ./vrs-int-data/my-memory --snapshot EXACT_SNAPSHOT_HASH \
  --encoder ./vrs-int-models/vision-encoder \
  --image ./vrs-int-data/query/new-image.png --gpu GPU-YOUR-UUID \
  --ratings g,s,q,e --output ./vrs-int-data/recommendation.json
```

여기서 `EXACT_SNAPSHOT_HASH`는 조회할 구축 결과의 해시입니다. `--ratings g,s,q,e`는
전체 등급을 사용하고 `--ratings g,s`처럼 좁힐 수도 있습니다. 추천에는 과거 이미지
폴더·caption 파일·Writer가 필요하지 않습니다. 입력 이미지와 바이트 해시가 같은
관측은 조회에서 제외하므로 별도의 이미지를 사용하세요. 기억이 1~2장뿐이면 관련
태그를 얻을 선택지가 매우 적습니다. 예제는 사용법 설명이며 추천 품질 평가가 아닙니다.

출력의 `candidates`는 참고 태그, `identities.artist/character/copyright`는 이름 후보,
`observation_keys`는 후보를 지지하는 관측 ID입니다. 후보 점수는 이웃 투표량이며
원작자 확률이 아닙니다. 목표는 이 재료를 새로운 프롬프트에 활용하는 것입니다.

개인 이미지·태그·출력은 Git에서 제외된 `/vrs-int-data/`, 별도 인코더는
`/vrs-int-models/`에 두면 됩니다. 이 예제 폴더에는 공개 가능한 형식·스크립트만
남깁니다. 이미지·추가 JSONL과 모델·기억 파일은 제외 규칙 및 CI로 검사하지만,
개인 자료는 파일 형식과 무관하게 예제 폴더 밖에 보관하세요.
