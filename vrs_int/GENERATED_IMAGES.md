# 생성 이미지로 개인 VRS-INT 기억 만들기

생성 이미지에 남아 있는 프롬프트를 캡션과 태그로 보존하고, 같은 Gemma
이미지 인코더로 검색 특징을 만드는 로컬 가져오기 도구입니다. 이미지 픽셀과
모델 가중치는 기억에 포함하지 않습니다. 원본 파일을 수정하거나 삭제하지 않습니다.
생성 당시 요청과 이미지의 실제 내용이 다를 수 있으므로 출처와 불확실성을 함께 남깁니다.

## 캡션의 출처

| 입력 | 기억에 보존하는 내용 |
| --- | --- |
| NovelAI 메타데이터 | 기본 프롬프트, 캐릭터별 프롬프트와 위치, 분리한 네거티브, 허용한 생성 설정 |
| ComfyUI API prompt 그래프 | 저장 이미지로 이어지는 연결의 포지티브·네거티브, 지원되는 텍스트 노드의 실제 값 |
| A1111 스타일 parameters | 포지티브와 네거티브, 허용한 생성 설정 |
| 메타데이터를 읽지 못한 이미지 | 선택적으로 PixAI가 추정한 태그와 점수, 모델 판본·임계값 |

원래 캡션은 생성 의도입니다. PixAI 캡션은 시각적 태그 추정입니다.
장문의 자연어 해설을 새로 쓰는 LLM은 이 경로에 사용하지 않습니다.
임의의 PNG 필드나 전체 워크플로는 복사하지 않으며, 그래프도 실행하지 않습니다.
지원되지 않는 ComfyUI 텍스트 노드는 경고를 남깁니다. 이미 읽을 수 있는
포지티브가 있는 기록에는 보완 태거를 자동으로 덧붙이지 않습니다.

작가 스타일·캐릭터·작품을 별도 필드에 보존합니다. 선택한 태그 사전에 분류가
있는 이름과 명시적인 작가 표기를 활용합니다. 분류를 알 수 없는 텍스트도
프롬프트·일반 태그로 남지만 이름의 종류를 확정하지 않습니다.
PixAI의 `style`은 작가 스타일 **참고 후보**로 `tags_artist`에 저장하며
그림의 실제 작가를 확인한 결과가 아닙니다.

숫자 가중치가 0 이하인 구간(예: `-2::hat, artist:example::`)은
긍정 추천 태그로 사용하지 않습니다. 원문과 이름은 캡션·생성 기록에 그대로
보존하고 새 가져오기에서는 `suppressed_prompt_fragments`에도 따로 남깁니다.
`generation-metadata-v2`로 만든 기존 기억도 추천 시 저장된 원문을 다시
해석합니다. 저장된 판본과 이미지 특징은 바꾸지 않고, 결과의
`generated_tag_view`에 추천용 해석기의 판본을 기록합니다. 이 수정 때문에
기존 카탈로그를 다시 스캔하거나 이미지 특징을 재추출할 필요는 없습니다.
새 스캔은 파서 판본이 달라지므로 새 카탈로그 경로를 사용하세요.

내용 등급은 전부 `u`(미확인)입니다. PixAI가 예측한 등급은 감사 기록에만
남기며, 메타데이터나 태거 점수만으로 `g/s/q/e`를 확정하지 않습니다.
`u`는 일반 등급을 뜻하지 않습니다.

## 1. 메타데이터 스캔

저장소 루트에서 기존 VRS-INT 환경을 사용합니다. 스캔에는 Pillow,
구축에는 CUDA PyTorch와 [인코더 의존성](README.md)이 필요합니다.
입력 폴더는 여러 번 지정할 수 있습니다. 아래 경로와 GPU UUID는 사용자의
경로와 `nvidia-smi --query-gpu=uuid,name --format=csv` 결과로 바꾸세요.

```bash
python -m vrs_int.generated scan \
  --root /path/to/generated-images \
  --root /path/to/other-output \
  --catalog ./vrs-int-generated/catalog.sqlite3 \
  --cpu-threads 2
```

선택적으로 `--categories /path/to/categorized-tags.jsonl`을 추가하면
`tags_artist`, `tags_character`, `tags_copyright`가 있는 기존 JSONL에서
이름의 분류를 읽습니다. 사전 파일도 사용자가 별도로 준비합니다.

PNG·WebP·JPEG의 내장 정보를 확인합니다. 보고서는 `catalog.scan.json`,
캡션과 태그는 `catalog.jsonl`에 남습니다. 스캔은 메타데이터 처리 단계이며,
이 단계만으로 이미지 검색 기억이 완성되지는 않습니다.
스캔 완료 시점 이후에 생성된 파일은 다음 스캔 대상으로 남습니다.

## 2. 한 GPU에서 검색 기억 구축

```bash
python -m vrs_int.generated build \
  --catalog ./vrs-int-generated/catalog.sqlite3 \
  --encoder ./vrs-int-models/vision-encoder \
  --store ./vrs-int-generated/memory \
  --checkpoint ./vrs-int-generated/features.sqlite3 \
  --gpu GPU-YOUR-UUID \
  --batch-size 8 --cpu-threads 2 --preprocess-workers 2 \
  --gpu-fraction 0.5 --delay 0.02
```

동일한 명령으로 다시 실행하면 이미 추출한 특징을 체크포인트에서 이어갑니다.
구축 전에 파일 해시를 다시 확인하며, 스캔 후 바뀐 파일은 오류로 남깁니다.
GPU 메모리 부족은 배치를 줄여 재시도하고, 한 장도 처리할 수 없으면 중단합니다.
`--gpu-fraction`은 다른 프로세스가 이미 쓰는 메모리를 고려한 시작 시점의
VRAM 예산입니다. 동시 실행 프로그램이 나중에 메모리를 더 확보할 수 있으므로
모든 시점에 장치 전체 사용량을 보장하는 제한은 아닙니다.

체크포인트·카탈로그는 원본 파일의 내용 해시와 가져오기 설정에 연결됩니다.
새 스캔 판본에는 새 체크포인트와 출력 폴더를 쓰세요. 이 가져오기 경로는
기존 개인 기억의 삭제나 갱신을 자동으로 동기화하는 서비스가 아닙니다.

## 3. 두 GPU와 PixAI를 함께 사용하기

Linux의 `taskset`과 서로 다른 CUDA GPU 두 개를 사용하는 선택적 실행기입니다.
PixAI 가중치와 공식 전처리 코드는 **미리 받은 로컬 폴더**를 지정합니다.
가져오기 중 모델을 다운로드하지 않습니다. PixAI 로딩은 해당 폴더의
커스텀 Python 코드를 실행하므로 신뢰하는 모델 폴더를 사용하세요.

```bash
python -m vrs_int.generated_pool \
  --catalog ./vrs-int-generated/catalog.sqlite3 \
  --encoder ./vrs-int-models/vision-encoder \
  --pixai /path/to/pixai-tagger-v1.0/snapshot \
  --output ./vrs-int-generated/full \
  --gpu GPU-FIRST-UUID --gpu GPU-SECOND-UUID \
  --cpu-threads 14 --shards 16
```

첫 GPU는 메타데이터가 있는 이미지의 특징을 추출하고, 두 번째 GPU는
읽을 수 없는 파일에 PixAI 태깅을 적용합니다. 태깅이 끝나면 두 GPU가
남은 특징 묶음을 나눠 처리합니다. 지정 CPU 수를 둘로 나눠 고정하고,
낮은 CPU 우선순위와 최대 90%의 시작 시점 VRAM 예산을 사용합니다.
GPU 연산 사용률 자체를 90%로 제한하는 기능은 아닙니다.

인코더 전용 의존성이 별도 폴더에 있다면
`--encoder-pythonpath /path/to/encoder-dependencies`로 지정할 수 있습니다.
실행기는 `progress.json`, `resources.jsonl`, 각 작업 로그를 남기고,
성공 후 `report.json`과 `memory/`를 만듭니다. 최종 병합은 각 묶음의
해시와 인코더 호환 조건을 확인합니다. `complete-with-errors`이면
각 `.build.json`과 체크포인트의 `errors`를 확인해야 합니다.
실패하거나 중단하면 체크포인트를 남기고 자식 작업을 종료합니다.

완료 후 원본 카탈로그와 최종 기억의 캡션·메타데이터·핵심 이름을 전수 대조할 수 있습니다.

```bash
python -m vrs_int.generated_audit \
  --catalog ./vrs-int-generated/catalog.sqlite3 \
  --run ./vrs-int-generated/full \
  --output ./vrs-int-generated/full/source-audit.json
```

이 감사는 기록 보존 검사이며, 이미지와 태그의 의미적 일치율 검사는 아닙니다.

## 4. 새 이미지로 추천하기

최종 보고서의 `snapshot` 값을 아래 자리에 넣습니다.

```bash
python -m vrs_int verify \
  --store ./vrs-int-generated/full/memory --snapshot SNAPSHOT

python -m vrs_int recommend \
  --store ./vrs-int-generated/full/memory --snapshot SNAPSHOT \
  --encoder ./vrs-int-models/vision-encoder \
  --image /path/to/reference.png --ratings u \
  --gpu GPU-YOUR-UUID --output ./vrs-int-generated/recommendation.json
```

**생성 이미지 기억만 조회하려면 `--ratings u`를 지정합니다.** 현재 CLI 기본값은
`g,s,q,e,u`이고, `g,s`로 좁히면 미확인 등급의 사례가 검색되지 않습니다.
구체적인 CLI 옵션은 `python -m vrs_int recommend --help`로
확인하세요. 추천에는 같은 인코더와 새 참고 이미지가 필요하고, 과거 원본 이미지를
다시 열지는 않습니다. 결과는 이웃 사례의 창작 참고 태그이며 실제 속성 확정은 `U`입니다.

## 보존과 공유

완성된 기억에도 원래 프롬프트, 네거티브, 이름, 설정, 경로를 포함한 출처 정보가
남을 수 있습니다. `vrs-int-generated/`는 Git 제외 대상이고 공개 코드 배포에
넣지 않습니다. 가중치·카탈로그·JSONL·특징·기억·실행 로그는 개인 자료입니다.
이번 도구는 원본을 자동으로 지우지 않습니다. 인코더 교체나 재구축에는 원본이
다시 필요하므로, 보존 여부는 별도로 결정해야 합니다.

## English overview

This local importer turns embedded generation prompts into captions and tags, then
computes image features with the existing Gemma vision encoder. NovelAI character
prompts and negatives remain separate. ComfyUI graphs are read without execution;
only supported text nodes connected to saved outputs are resolved. Metadata captions
describe generation intent. Optional PixAI fallback captions contain visual tag
predictions with scores and model provenance. All records have rating `u` (unknown),
so use `--ratings u` to query only these unrated records (the CLI default also includes `u`).

Run `generated scan`, then `generated build` for one GPU, or `generated_pool` for
two Linux GPUs with a local PixAI model. The commands above are complete examples
after replacing paths and GPU UUIDs. Reports and checkpoints allow inspection and
resumption; only a final report with a verified snapshot represents completed memory.
Keep all generated artifacts private. Source images remain unchanged and are not
embedded in the store. The importer neither trains model weights nor publishes data.

Nonpositive numeric emphasis is excluded from positive recommendation tags. Original
captions, names and generation metadata remain intact; new records also retain those
spans in `suppressed_prompt_fragments`. Recommendation reparses older generated
records from their stored prompt without rewriting snapshots or re-encoding images.
The result identifies this interpretation as `generated_tag_view`. Start any new
scan in a new catalog when changing parser versions; existing build checkpoints
and stored records do not require a rescan for this recommendation correction.
