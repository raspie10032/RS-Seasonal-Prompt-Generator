# VRS-INT 0.1 — 개인별 이미지·태그 기억

**코드를 공유하고, 자료는 각자가 취향에 맞게 선택해 구성합니다.**
이미지·태그 데이터베이스·인코더 가중치는 코드 배포에 포함하지 않습니다.
사용자가 준비한 단보루 이미지와 태그 목록으로 기억을 구축하고,
새 이미지와 비슷한 사례의 태그를 창작 재료로 추천하는 실험용 CLI입니다.
현재 ComfyUI 노드 및 자동 다운로드·정기 수집은 포함하지 않습니다.

직접 생성한 이미지의 메타데이터를 가져오려면
[생성 이미지 캡션과 PixAI 보완 안내](GENERATED_IMAGES.md)를 참고하세요.
NovelAI·ComfyUI 등의 생성 프롬프트를 보존하고, 읽을 수 없는 이미지에는
로컬 PixAI 태거를 선택적으로 적용하는 별도 가져오기 경로입니다.

검색한 경험을 TIPO 생성에 사용하려면 [생성 상태 비교 실험](GENERATION_STATE.md)을
참고하세요. 요청만 전달 / 같은 참고 태그 나열 / 구조화한 상태를 전달하는 세 방식을
비교하는 별도 GPU 실행기이며, 신경망 내부 상태 주입이나 노드 자동 연결은 아닙니다.

현재 Gemma 모델 선택과 혼합 정밀도 후보의 사용법·검증 범위는
[Gemma 4 TIPO 실행 기준](GEMMA4_RUNTIME.md)에 정리했습니다.

개인 한국어 태그 CSV를 **기본 텍스트 지식풀 + 번역 사전**으로 연결하려면
[텍스트 지식·태그 사전 안내](TAG_DICTIONARY.md)를 참고하세요.
이미지 없이도 한국어 문장의 용어를 찾아 영문 태그·한국어 설명·이름 후보를
조회할 수 있고, 한국어 생성 요청에는 이 용어 지식을 함께 전달합니다.
`recommend`와 생성 상태 내보내기에 `--tag-dictionary`를 지정할 수
있고, 상태의 `--resolve`는 선택한 이름과 서로 맞는 참고 재료를 추립니다.
기존 이미지 경험을 재구축하지 않으며, 이 사전 파일은 개인 경로에 따로 둡니다.

## 무엇이 남고 무엇이 필요한가

- 기억에는 태그, 작가·캐릭터·작품 분류, 이미지 해시, 출처 ID, 원본 메타데이터,
  제외한 태그와 사유, 정상 특징 벡터가 들어갑니다. 이미지 픽셀은 넣지 않습니다.
- 작가·캐릭터·작품은 한 번만 등장해도 보존합니다. 명시적으로 분류된 이름은
  품질·해상도·워터마크·서명 등 제외 목록보다 우선합니다.
- 추천에는 **VRS-INT 기억 + 동일한 이미지 인코더 + 실행 라이브러리**가 필요합니다.
  과거 원본 이미지·옛 특징 캐시·Writer LLM은 필요하지 않습니다.
- 구축은 한 번에 한 이미지씩 처리합니다. 여러 이미지 묶음을 순차적으로 추가할 수
  있으므로 전체 원본을 동시에 확보할 필요가 없습니다. 프로그램은 원본을 삭제하지 않습니다.
  나중에 인코더를 바꾸려면 원본을 다시 확보해 특징을 재추출해야 합니다.
- 이름 후보는 **비슷한 사례의 창작 참고 후보**입니다. 실제 작가·캐릭터·작품을
  확인한 결과나 정답 확률이 아닙니다. 사용자가 지정한 이름을 대체하는 기능은 없습니다.

`INT`는 프로젝트 이름이며 모든 값을 정수로 저장한다는 뜻은 아닙니다.
기본 특징 형식은 `f32le`이고 손실 없는 저장 압축을 사용합니다.
`i8`은 추천 변화가 생기는 선택적 실험 형식입니다.

## 준비

처음 구성할 때는 **[이미지·텍스트 쌍 예제와 실행 안내](../examples/vrs-int/README.md)**를
참고하세요. 최초 구축·증분 수정 JSONL 템플릿, 이미지별 JSON을 묶는 스크립트,
모델 없이 실행하는 `check-input` 검사를 제공합니다. 작가·캐릭터·작품을 별도 필드로
보존하며, 자연어 `caption`은 감사 기록으로만 남고 검색 특징이나 추천 태그로 변환되지 않습니다.

저장소 루트에서 `python -m vrs_int --help`로 실행합니다. Python 3.11 이상,
저장·버전 관리·특징 검증에는 표준 라이브러리와 NumPy가 필요합니다.
이미지 구축·추천에는 CUDA PyTorch, Transformers의 Gemma4VisionModel,
bitsandbytes, accelerate, Pillow, safetensors가 필요합니다.
검사한 조합은 torch 2.14.0+cu130, transformers 5.16.1, bitsandbytes 0.50.2,
accelerate 1.15.0, Pillow 12.3.0입니다. 새 환경의 자동 설치·패키징은 아직 제공하지 않습니다.

가중치는 별도로 준비한 vision-only 폴더를 `--encoder`로 지정합니다.
폴더에는 config.json, preprocessor_config.json, model.safetensors 및 파일별
SHA-256을 적은 provenance.json이 있어야 합니다.
[`prepare_encoder.py`](prepare_encoder.py)는 명시적으로
실행할 때 고정 모델의 시각 가중치만 내려받는 도구입니다. 프로세서 판본도
provenance에 기록합니다. 다른 결과 해시로 만들어진 인코더는 호환된다고 가정하지 않습니다.
구축·추천 자체는 다운로드하지 않으며 CUDA가 없으면 오류로 멈춥니다.

```bash
# 사용자가 명시적으로 실행할 때 시각 가중치만 다운로드합니다.
python -m vrs_int.prepare_encoder ./vrs-int-models/vision-encoder
```

이미지는 `--images` 아래에 두고, JSONL에는 한 줄에 한 이미지의 정보를 적습니다.
다음 이름은 입력 형식을 보여주기 위한 가상 예시입니다.

```json
{"id":123,"image":"images/123.jpg","rating":"g","tags":["blue_hair","highres"],"tags_artist":["example_artist"],"tags_character":["example_character"],"tags_copyright":["example_series"]}
```

`id`는 단보루의 양의 정수 ID, `image`는 이미지 루트 기준 상대 경로입니다.
`rating`은 g/s/q/e 중 하나를 명시합니다. 모든 등급을 지원합니다.
태그 필드는 문자열 배열입니다. `tags_general`, `tags_meta`도 지원합니다.
분류가 없는 태그는 unknown으로 남기며 이름만 보고 작가인지 추정하지 않습니다.
`split`은 기본 train이며 val/test로 지정한 자료는 추천 이웃에서 제외합니다.
기존 단보루 API 응답을 이 형식으로 자동 변환하는 수집기는 아직 없습니다.

```bash
python -m vrs_int check-input \
  --manifest /path/to/batch1.jsonl --images /path/to/image-root
```

이 명령은 표준 라이브러리만으로 필드·중복 ID·경로·파일 해시를 검사합니다.
픽셀 디코딩·태그 의미·인코더 호환성 검사는 아니며 모델을 로드하지 않습니다.

## 개인 기억 구축 및 추가

아래 경로와 GPU UUID는 사용자 환경에 맞춰 지정합니다.

```bash
python -m vrs_int build \
  --manifest /path/to/batch1.jsonl --images /path/to/image-root \
  --encoder /path/to/vision-encoder --store /path/to/my-vrs-int \
  --gpu GPU-YOUR-UUID --label my-first-collection
```

명령이 출력한 `snapshot` 해시를 보관합니다. 새 묶음을 받았다면:

```bash
python -m vrs_int build \
  --manifest /path/to/batch2.jsonl --images /path/to/next-image-root \
  --encoder /path/to/vision-encoder --store /path/to/my-vrs-int \
  --gpu GPU-YOUR-UUID --base PREVIOUS_SNAPSHOT_HASH --label my-next-collection
```

같은 관측 ID의 새 태그는 해당 관측을 대체하며 표를 하나 더 세지 않습니다.
수정 기록에는 유지할 작가·캐릭터·작품을 포함한 전체 태그·메타데이터와 이미지를
다시 제공합니다. 변경하지 않는 관측은 새 묶음에 넣지 않아도 유지됩니다.
같은 내용을 재입력하면 같은 snapshot을 돌려줍니다. 동일 이미지 해시의 다른 ID는
출처로 보존하지만 추천에서 중복 투표하지 않습니다. 큰 자료는 묶음별로 추가하세요.
갱신 구현은 변경 대상 bucket을 RAM에 유지하므로 매우 큰 단일 갱신의 RAM 상한은
보장하지 않습니다. encoder/전처리/특징 codec/태그 정책이 다르면 혼합을 거부합니다.

CLI는 CPU 스레드·실행 범위를 2개로 제한하고 우선순위를 낮춥니다.
PyTorch의 GPU 할당 예산은 장치 메모리의 20%이며 외부 라이브러리 할당까지 포함한
하드웨어 전체 상한은 아닙니다. 현재는 GPU에서 제한된 벡터 묶음을 정확 검색합니다.
첫 조회는 모든 저장 bucket을 해제해 이미지 검색 대상을 캐시하므로 CPU·RAM·I/O를
사용합니다. 천만 장 규모의 검색 시간·RAM 사용량은 아직 검증하지 않았습니다.

## 새 이미지에서 참고 태그 추천

```bash
python -m vrs_int recommend \
  --store /path/to/my-vrs-int --snapshot EXACT_SNAPSHOT_HASH \
  --encoder /path/to/vision-encoder --image /path/to/new-image.png \
  --gpu GPU-YOUR-UUID --ratings g,s,q,e --output /path/to/recommendation.json
```

일반 태그, 작가·캐릭터·작품 후보, 근거 관측 ID를 JSON으로 반환합니다.
품질·연도 상투 태그(`amazing quality`, `very aesthetic`, `newest`, `year 2024`
등)는 추천 순위를 자르기 전에 제외합니다. 명시적으로 작가·캐릭터·작품으로
분류된 이름은 우선 보호합니다. 원본 캡션·태그·특징과 snapshot은 그대로 두며,
기존 기억에도 다음 조회부터 적용됩니다. 결과의 `recommendation_policy`와
`excluded_tags`에 필터 판본과 제외 사유를 기록합니다.
Quality/year boilerplate is filtered before the recommendation limit; explicit
artist/character/copyright labels stay protected. This query-time policy leaves
stored captions, tags, vectors, and snapshot hashes unchanged.
점수는 이웃의 가중 지지량입니다. `visual_verdict=U`는 신원·시각적 정답을
확정하지 않았다는 뜻입니다. 이 독립 경로는 VRS 관측 기억을 압축·검색하지만
기존 native VRS 판정 코어를 호출하거나 신경망을 재학습하지 않습니다.

## 코드 버전과 기억 버전

공용 대형 자료를 운영하지 않아도 개인 기억의 재현·되돌리기를 위해 버전은 유지합니다.
코드 버전, 저장 형식, 인코더/태그 정책 계약, 기억 내용 해시를 구분합니다.
태그 판본이 바뀌면 새 snapshot이 생기고 이전 snapshot은 그대로 남습니다.
설치가 자동 활성화로 이어지지 않으며 추천에는 정확한 snapshot을 명시합니다.

원할 경우 자신의 기억 전체나 직전 판본 대비 변경 bucket만 내보낼 수 있습니다.
공유 패키지에는 원본 메타데이터의 로컬 경로 등도 포함될 수 있습니다.
코드만 배포하는 기본 흐름에는 이 기능을 사용할 필요가 없습니다.

```bash
python -m vrs_int export --store /path/to/my-vrs-int \
  --snapshot NEW_HASH --base EXACT_PARENT_HASH --output /path/to/delta
python -m vrs_int install --store /path/to/receiver --package /path/to/delta
python -m vrs_int activate --store /path/to/receiver \
  --snapshot NEW_HASH --expected-current OLD_HASH
```

최초 전체 내보내기는 `--base`를 생략합니다. 처음 활성화는 `--expected-current none`입니다.
되돌리기는 이전 snapshot을 지정해 activate합니다. 잘못된 부모, 관측 판본 충돌,
파일 해시 오류, 인코더 혼합은 거부합니다. 내용 해시는 게시자의 신원을 인증하는
전자서명이 아닙니다. 구버전 정리·가비지 수집·자동 동기화는 구현하지 않았습니다.

## 코드 배포와 개인 경험 데이터

이 저장소는 **VRS-INT 코드·문서·가상 입력 예제만 배포**합니다. 초기 기억을 포함해
실제 이미지, 수집한 원본 태그, 이미지 특징, 스냅샷, 추천 결과, 모델 가중치를
Git·릴리스에 포함하지 않습니다. `g/s/q/e` 지원은 프로그램의 입력 기능이며,
어느 등급의 실제 자료도 기본 제공한다는 뜻이 아닙니다.

- 개인 입력·기억·추천 결과: 저장소 밖 또는 `./vrs-int-data/`에 보관합니다.
- 별도 인코더: 저장소 밖 또는 `./vrs-int-models/`에 보관합니다.
- 공개 예제: `examples/vrs-int/`의 가상 `g` 등급 메타데이터만 사용합니다.
- `export`는 사용자가 명시적으로 실행하는 로컬 파일 내보내기입니다.
  자동 업로드나 공용 기억 갱신은 없고, 코드 버전과 개인 기억 판본은 별개입니다.

`.gitignore`는 개인 폴더와 경험·모델 산출물을 제외합니다. CI의
`python scripts/check_code_only.py`는 Git에 추적된 파일을 추가로 검사합니다.
압축 기억도 원본 태그와 메타데이터를 포함하므로 익명화된 자료로 취급하지 않습니다.

## 오프라인 검증과 현재 한계

```bash
python -m pip install numpy
python -m unittest discover -s tests -p 'test_vrs_int*.py' -v
python scripts/check_code_only.py
```

검사는 가상 입력과 대체 인코더를 사용하며 모델 다운로드나 GPU 추론을 실행하지
않습니다. 입력 검사, 이름·원본 메타데이터 보존, 수정·중복 처리, 이전 판본 유지,
무손실 압축, 이관 자료 해시 검증, 전체·증분 패키지 검증, 되돌리기를 다룹니다.
실제 작가 적중률·취향별 유용성·대규모 검색 성능을 입증하는 검사는 아닙니다.

현재 실험용 CLI이며 ComfyUI 통합·자동 수집·새 환경 자동 설치는 제공하지 않습니다.
전체 검색 비용과 첫 조회의 RAM 사용량은 자료량에 따라 커집니다.
