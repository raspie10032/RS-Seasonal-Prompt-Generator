# Gemma context expansion: tags + natural-language instructions

[한국어 README](README_KOR.md) · [English README](README.md) · [Native model setup](NATIVE_TIPO.md)

## 한국어

**Gemma Prompt Expander Native (Tags / Natural Language)**는 텍스트·이미지의
맥락을 읽고 새 장면을 구성하는 노드입니다. 같은 Gemma 4 E2B `CLIP`을 사용하며
추가 pip 설치·llama.cpp·별도 번역 모델이 필요 없습니다. 기존 `Gemma TIPO Native`의
태그 전용 입력 형식·출력 포트·학습 템플릿은 유지됩니다.

### 연결과 두 출력

1. [네이티브 모델 준비](NATIVE_TIPO.md)를 마치고 `Load CLIP`의 `CLIP` 출력을
   새 노드의 `clip`에 연결합니다. 네이티브 전체 Gemma 4 E2B가 필요합니다.
2. `prompt`에 유지할 것·바꿀 것·제외할 것을 적습니다. 이미지 한 장은 선택 입력입니다.
3. `output_format`을 고릅니다.

| 설정 | 실행 | 출력 포트 |
| --- | --- | --- |
| `both` (기본) | 자연어 장면 작성 → 같은 장면을 태그로 변환, 순차 2회 | `tags` + `natural_language` |
| `tags` | 텍스트·이미지에서 태그 작성, 1회 | `tags`; 자연어는 빈 문자열 |
| `natural_language` | 텍스트·이미지에서 이미지 생성 지시문 작성, 1회 | `natural_language`; 태그는 빈 문자열 |

두 형식의 **출력 언어는 영어**입니다. 한국어·영어·일본어 입력은 모델에 원문 그대로
전달합니다. `natural_language`는 설명문이나 태그 목록이 아니라 `Create ...` 형태의
이미지 생성용 문장을 요청합니다. 문장 출력에는 쉼표 기준 태그 후처리를 적용하지 않습니다.

`both`의 두 번째 호출은 첫 장면과 원래 제약을 받으며 이미지를 다시 인코딩하지 않습니다.
동일 모델을 순차 사용하므로 모델을 두 벌 로드하지 않지만 호출 시간·문맥 비용은 늘어납니다.
`max_tokens`는 **호출 한 번당** 한도입니다. `tag_length`는 목표 상세도이며 개수 보장이 아닙니다.
두 출력의 의미 일치도는 여전히 검증 대상입니다.

예: `파란 머리는 유지하고 겨울 야시장에서 차를 들고 서 있게. 코트는 입히고 목도리는 빼줘.`
`ban_tags`에는 `scarf`처럼 영어 제외 항목을 넣습니다. 두 호출에 제외 지시를 전달하고,
태그 출력에만 기존 필터를 적용합니다. 자연어에서 제외 대상을 확실히 제거하거나,
한국어 부정을 자동 판정해 태그에서 강제로 지우는 기능은 아닙니다.

[API 그래프 예제](examples/gemma-prompt-expand-native.api.json)는 모델 파일 이름을
수정해 `/prompt`에 제출할 API 형식입니다. 캔버스 드래그용 UI workflow JSON은 아닙니다.
두 출력은 각각 `PreviewAny`나 생성 모델의 문자열 입력으로 연결할 수 있습니다.

### 로컬 태그 사전과 한국어 보강

`tag_set=danbooru` 또는 `e621`을 선택합니다. `tag_knowledge_path`는 선택 입력입니다.
비워두면 모델의 기존 지식만 사용합니다. 파일을 지정하면 요청과 확장 장면에서 필요한
용어를 찾아 참고로 전달합니다. 전체 사전을 GPU 문맥에 넣지 않으며 사전은 CPU/RAM에
캐시합니다. 추가 GPU 번역 모델은 없습니다. 큰 사전의 첫 로드·색인은 CPU 시간과 RAM을 씁니다.

지원 파일:

- 기존 VRS-INT 단보루 한국어 CSV: 헤더 없는 `tag,category_id,post_count,description`.
  `description`의 `키워드:` 뒤 검색어도 읽습니다. **단보루용**으로만 해석합니다.
- `rs-tag-knowledge/1` JSONL: 태그별 출처·분류·영어/일본어/한국어 검색어·뜻풀이·검수 상태.
  실제 데이터는 저장소 밖에 따로 보관합니다.

기존 영문 태그·일문 표기를 유지하고 `terms.ko`와 `definitions.ko`를 보강하는 흐름입니다.
`terms.en/ja`는 확보한 대응어가 있을 때만 넣습니다. 일본어 표기가 있다는 이유로 일본어
뜻풀이까지 갖췄다고 집계하지 않습니다. **검색어·관련어는 정확한 동의어라는 보장이 없습니다.**
다의어 후보, 제외어도 원래 요청과 함께 전달하며 자동으로 긍정 태그로 승격하지 않습니다.
이름 분류(작가·캐릭터·작품·기여자)는 정식 태그명과 별도 `name_aliases` 필드로만 검색합니다.
이름 항목의 `terms`는 관련 검색어로 보존하지만 생성 입력에 이름 후보를 넣는 데 쓰지 않습니다.
예를 들어 작가의 작업 분야인 `2D 애니메이션`은 작가 별명이 아닙니다. CSV에 있던 한국어·일본어
이름 검색어도 자동으로 확정 별칭으로 취급하지 않습니다. 실제 이름 대응을 확인해 JSONL의
`name_aliases.ko/ja/en`에 등록하세요. 영문 정식 태그는 그대로 조회할 수 있습니다.
모델이 새로 쓴 장면에서는 이름 분류를 조회하지 않습니다. 사용자 원문의 확인된 이름 후보는
별도로 전달합니다. 이 규칙은 사전 후보 선택용이며 모델 자체의 이름 추측까지 막지는 않습니다.
`cozy`처럼 분위기를 나타내는 표현은 자연어에 쓸 수 있습니다. 태그는 그 장면에서 이미
정한 조명·배경·소품의 표준 표현으로 옮기고, 분위기 하나 때문에 새 소품을 마음대로 더하지 않습니다.

아래는 파일 형식을 설명하는 직접 작성 예시입니다. 전체 사전이나 공식 번역 데이터가 아닙니다.

```json
{"schema":"rs-tag-knowledge/1","source":"danbooru","tag":"blue_eyes","category":"general","terms":{"en":["blue eyes"],"ja":["青い目"],"ko":["파란 눈","벽안"]},"definitions":{"ko":"눈의 홍채가 파란색인 표현."},"status":"unreviewed","provenance":{"note":"Manually authored format example; verify against your source before use."}}
```

이름 별칭의 형식 예시(실제 사전 항목이 아닌 가상 이름):

```json
{"schema":"rs-tag-knowledge/1","source":"danbooru","tag":"example_character (example_series)","category":"character","name_aliases":{"ko":["예시 캐릭터"],"ja":["架空のキャラクター"]},"status":"unreviewed","provenance":{"note":"Fictional format example only."}}
```

한 줄에 한 객체이며 `(source, tag)` 중복은 거부합니다. `status`는 `reviewed`,
`unreviewed`, `machine` 중 하나이고, `provenance`가 필요합니다. CSV를 불러온 항목은
`unreviewed`입니다. `reviewed`도 사용자가 붙인 기록이지 프로그램이 번역을 검증한 결과가 아닙니다.

```bash
# 저장소 루트에서 실행. 경로는 본인의 개인 데이터 경로로 변경합니다.
python -m tag_knowledge --file /path/to/personal/danbooru-kr.csv
python -m tag_knowledge --file /path/to/personal/danbooru-kr.csv --text "벽안은 유지하고 목도리는 제외"
python -m tag_knowledge --file /path/to/personal/danbooru-kr.csv --export-jsonl /path/to/personal/tag-knowledge.jsonl
python -m tag_knowledge --file /path/to/personal/tag-knowledge.jsonl --tag blue_eyes --source danbooru
```

내보내기는 기존 출력 파일을 덮어쓰지 않습니다. 내보낸 개인 JSONL을 편집해 한국어
뜻·검색어를 보강한 뒤 노드에 지정하세요. 파일이 바뀌면 다음 호출에서 다시 읽습니다.
각 호출의 `raw_output`에는 사전 해시·범위, 요청에서 찾은 후보, 원문 두 출력, 로컬 사전에
있는 태그와 `unverified` 태그, 장면에서 조회한 용어도 기록합니다. 미확인은 오답이라는 뜻이
아니며 임의로 삭제하지 않습니다.

### VRS 상태를 네이티브 Gemma에 전달하기

`vrs_state_path`에 `vrs-int-generation-state/1` 또는 `/2` JSON 파일을 지정하면,
VRS의 참고 재료를 자연어 작성과 태그 변환에 함께 전달합니다. 추가 학습·모델·pip 설치는
없습니다. 현재는 **이미 검색해 저장한 상태를 읽는 연결**이며, 노드가 이미지에서 자동 검색하거나
사전 전체를 학습하는 기능은 아닙니다. 파일은 개인 저장 경로에 보관하세요.

1. VRS-INT에서 [이미지 추천](vrs_int/README_KOR.md)을 실행해 추천 JSON을 저장합니다.
2. [상태 내보내기](vrs_int/GENERATION_STATE.md)로 **현재 `prompt`와 같은 요청**의 상태를 만듭니다.
3. 노드의 `vrs_state_path`에 상태 파일 경로, `tag_knowledge_path`에 선택적 한국어 사전을 지정합니다.
4. `both`로 실행하고 두 출력과 `raw_output.vrs_reference`를 확인합니다.

```bash
python -m vrs_int.generation_state \
  --recommendation /private/vrs/recommendation.json \
  --request '파란 머리는 유지하고 겨울 야시장에서 차를 들고 서 있게. 목도리는 빼줘.' \
  --keep 'blue hair' --target 'winter' --target 'night market' \
  --target 'standing' --target 'holding cup' --exclude 'scarf' \
  --output /private/vrs/winter-state.json
```

노드 `prompt`에도 위 `--request`와 같은 문장을 넣습니다. `--keep/target/exclude`는
사용자가 고른 명시 조건입니다. 모델의 한국어 이해 능력을 비교할 때 한쪽에만 영문 정답
조건을 추가하면 공정한 비교가 아니므로 양쪽 입력 조건을 맞추세요. `--tag-dictionary`와
`--resolve`로 사전 기반 재료 선별을 사용한 상태도 지원합니다.

**규칙과 한계:**

- 요청이 바뀌면 상태를 새로 내보냅니다. 오래된 요청의 조건이 섞이지 않도록 불일치는 오류로 처리합니다.
- `keep/target`와 `exclude/ban_tags`가 충돌하면 생성 전에 오류를 냅니다.
  `exclude`는 두 호출의 지시에 포함하고 태그 출력 필터에도 적용합니다. 자연어는 지시 준수에 의존합니다.
- 작가·캐릭터·작품 후보는 `keep/target`에서 명시적으로 선택한 경우에만 모델 참고에 포함합니다.
  전체 이름·근거는 원본 상태에 보존하며, 검색 후보를 원작자나 확정 정체성으로 승격하지 않습니다.
- 품질·연도 후보를 제거하고, 그룹당 5개·공출현 예제 3개·JSON 문맥 최대 6,000자로 제한합니다.
  명시 조건 자체가 한도를 넘으면 조용히 누락하지 않고 오류를 냅니다.
- 현재 상태의 태그 공간은 **단보루**입니다. `tag_set=e621`과 함께 지정하면 오류를 냅니다.
  e621 사전 입력과 e621 경험의 호환 계약은 별개입니다.
- 파일 상태는 별도 참고 자료이며 현재 `IMAGE`와 검색 이미지가 같은지 자동 검증하지 않습니다.
  네이티브 Recall에서 연결한 `vrs_state`는 RGB 픽셀 해시로 동일 이미지를 확인합니다.
  `raw_output`에는 이 한계, snapshot·파일 해시·전달된 재료·제외 집계를 기록합니다.
- 파일 변경 시 ComfyUI 출력 캐시를 무효화합니다. 저장된 상태를 읽는 데 추가 인코더는 없지만,
  문맥 토큰이 늘어나는 만큼 Gemma의 처리 시간과 메모리 사용이 늘 수 있습니다.

[가상 상태 예제](examples/vrs-native-state.json)는 형식 확인용입니다. 실제 검색 결과가 아니며,
예제의 `request`를 노드 `prompt`에 복사하고 파일의 절대 경로를 지정해 연결을 확인할 수 있습니다.

### ComfyUI 안에서 VRS 검색까지 연결

[VRS 네이티브 안내](vrs_int/COMFYUI_NATIVE.md)에 비전 로더, 경험 구축/추가, 경험 로더,
이미지 검색과 두 출력의 전체 연결을 설명했습니다. `vrs_state`와 `vrs_state_path`는 둘 중 하나만
사용하세요. 기존 PIL/NF4 경험은 별도의 네이티브 경험으로 재인코딩해야 합니다.

### 데이터와 품질의 경계

- 단보루와 e621의 이름 공간과 분류를 분리합니다. 단보루 사전을 e621 전체 지식으로
  취급하지 않습니다. 출처가 다른 같은 태그는 별개 항목이고 대응 관계를 자동으로 만들지 않습니다.
- 사전 조회는 전체 문장 번역·형태소 분석·이미지 판정이 아닙니다. 특히 공백 없이 이어진
  일본어 표현은 검색에서 놓칠 수 있습니다. 원문은 모델에 그대로 전달됩니다.
- 현재 기능은 **추론 시 참고 자료 공급**입니다. 전 태그의 모델 내재화, 한국어 번역 검수,
  e621 전체 데이터 확보, 이중 출력용 추가 학습 완료를 의미하지 않습니다.
- 자연어가 유창해도 태그 변환 시 행동·공간 관계가 빠지거나 비표준 태그가 나올 수 있습니다.
  이름 보존과 부정 처리도 모델 지시만으로 완벽히 보장하지 않습니다.
- 기존 태그 전용 LoRA 병합 모델은 문장을 요청해도 태그로 답할 수 있습니다. 두 형식의
  기준 비교는 원본 모델에서 시작하고, 학습 후보는 같은 문제로 따로 비교하세요.
- 자동 이미지 검색은 [VRS Native Recall](vrs_int/COMFYUI_NATIVE.md)에서 수행하고 `vrs_state`로 연결합니다.
  이전 CLI의 저장된 상태는 계속 `vrs_state_path`로 연결할 수 있습니다.
- 기본 서버에 새 노드를 표시하려면 설치 후 ComfyUI를 재시작해야 합니다.
  생성 큐가 비어 있는지 먼저 확인하세요.

출처별 분류 참고: [Danbooru 공식 TagCategory](https://github.com/danbooru/danbooru/blob/master/app/logical/tag_category.rb),
[e621 공식 TagCategory](https://github.com/e621ng/e621ng/blob/master/app/logical/tag_category.rb).

### 다음 개발 계획 — 2026-10-14

**사용자 방향 변경: 추가 LoRA 학습을 보류하고 VRS 보강을 우선합니다.** 기존 학습 후보와
자료는 보존합니다. Gemma는 원문과 이미지를 이해하고, 회수한 재료 중 어울리는 것을 골라
하나의 장면을 구성한 뒤 자연어/태그로 표현하는 역할입니다. 한국어, 2D 애니·일러스트와
3D 게임 그래픽을 우선합니다.

1. **저장된 VRS 상태 → 두 출력 연결:** `vrs_state_path`로 구현했습니다. 사전은 뜻·표현을,
   경험은 외형·의상·행동·배경 후보를 제공합니다. 명시 조건이 참고 재료보다 우선합니다.
   이미지 검색은 이제 VRS Native Recall에서 실행하고 `vrs_state`로 직접 연결합니다.
   [네이티브 구축·검색 안내](vrs_int/COMFYUI_NATIVE.md)에 설치·계약·예제와 한계를 설명했습니다.
2. **같은 원본 모델에서 비교:** 모델·입력·사전·제외 조건을 고정하고 VRS 상태 유무를
   비교합니다. 원문 두 출력, 입력 해시, 실제 전달된 재료, 시간·메모리를 함께 남깁니다.
   저장된 다른 참고 자료를 재사용한 경우 새 이미지 검색 결과로 표시하지 않습니다.
3. **한국어 의미 지식 보강:** 행동·의상·공간 관계와 확인된 이름 별칭을 먼저 채웁니다.
   단보루/e621 출처·버전·검수 상태를 분리하고, 사전 수와 모델이 실제 사용한 범위를
   따로 보고합니다. 한국어 제외·범위 지시를 기계적으로 긍정 태그로 바꾸지 않습니다.
4. **필요한 경험을 자동으로 찾기:** 요청/이미지 → 호환 인코더·기억 조회 → 제한된 상태
   → Gemma 순서로 연결할 대상입니다. 인코더 계약, 현재 이미지 대응, 중복 검색 방지,
   GPU 점유를 확인해야 하며 기존 상태 연결만으로 이 단계가 완료된 것은 아닙니다.
5. **의미와 창작 유용성을 검증:** 이름 보존, 유지/변경/제외, 인물별 행동·위치, 두 출력의
   의미 일치, 비표준 태그, 확장의 유용성을 나눠 봅니다. 이미지 원본/교체/없음과
   요청에 맞는 경험/맞지 않는 경험을 대조합니다. `indoors, rain`을 창밖의 비 표현에
   성공한 것으로 채점하지 않습니다. 원문 평가와 후처리 효과도 분리합니다.

**다음 작업 단위:** 저장된 상태를 연결한 소규모 실행 결과를 검토하고, 한국어 표현과
재료 선택의 실패를 보강합니다. 회수·선택·표현의 원인을 분리해도 반복되는 결함이 남을 때
추가 학습을 다시 판단합니다. 300건 학습 자료와 이중 출력 LoRA는 현재 실행 계획에서 보류합니다.
개인 사전·이미지·경험·가중치·결과는 저장소 밖에 보관합니다.

## English

Use **Gemma Prompt Expander Native (Tags / Natural Language)** with the same native Gemma 4 E2B
`Load CLIP` used by the existing TIPO node. Optional input: one RGB `IMAGE`. No extra dependency,
translation model, download, llama.cpp process, or independent model allocator is introduced.

- `both`: first expand one coherent scene into an English image-generation instruction, then render
  that scene into tags with the original constraints. Two sequential calls to the same model; the
  second call does not encode the image again.
- `tags` / `natural_language`: one call; the unused STRING output is empty.
- Outputs: `tags`, `natural_language`, `raw_output` (JSON audit, including both unfiltered responses).
  Prose never passes through the comma-based tag filter. `max_tokens` applies to each call.
- `tag_set`: `danbooru` or `e621`. `tag_knowledge_path`: optional local legacy Danbooru Korean CSV
  or `rs-tag-knowledge/1` JSONL. The dictionary stays in CPU/RAM and only bounded lexical references
  enter the model context. Its first load/indexing still has CPU/RAM costs.
- Keep canonical tags and available English/Japanese forms; augment Korean terms/definitions.
  JSONL supports explicit `en`, `ja`, `ko` fields, source-specific categories, provenance and review
  status. The format example and CLI commands above show how to inspect/export/edit personal data.
  Exports refuse to overwrite existing files. No private dictionary is bundled.
- Identity categories use canonical tags and explicit `name_aliases` (`en`/`ja`/`ko` lists), not
  broad descriptive `terms`. Existing CSV keywords remain available for review but are not automatically
  promoted to verified names. Generated scene prose never supplies new identity candidates.
- Source vocabularies are never silently merged. A Danbooru CSV cannot stand in for an e621 dataset.
  The API graph is [here](examples/gemma-prompt-expand-native.api.json); change the model filename.

Lexical retrieval is not complete translation, morphological parsing, instruction-following proof,
or training. Unknown output tags are reported against the loaded snapshot without being discarded.
An entry's existence or user-supplied review flag does not prove semantic correctness. The model can
still omit actions, alter names, mishandle negation, invent tags, or disagree between prose and tags.
Previously merged tag-only LoRAs may remain biased toward tags; evaluate a base-model reference first.
Korean, English and Japanese requests remain unchanged in the prompt; outputs are English. Automatic
image lookup is provided by [VRS Native Recall](vrs_int/COMFYUI_NATIVE.md), connected through `vrs_state`. The original native TIPO workflow and training template remain compatible.

### Saved VRS state → native Gemma

Set optional `vrs_state_path` to a local `vrs-int-generation-state/1` or `/2` JSON export.
The exporter command and fictional [state example](examples/vrs-native-state.json) above show the
format. Use the exact same request in the export and node: mismatches and contradictory explicit
controls fail before generation. Recommendation JSON and nested trial reports must be exported first.
Optional `--tag-dictionary` / `--resolve` selections are respected.

Only KEEP/TARGET-selected identity candidates enter the writer context; other names remain in the
original evidence. Quality/year candidates are removed. References are bounded to five candidates
per group, three co-occurrence examples and 6,000 JSON characters. Explicit controls are never silently
truncated. EXCLUDE also reaches tag postprocessing; prose compliance remains a model behavior.
Existing states are Danbooru-specific and are rejected with `tag_set=e621`.

`raw_output.vrs_reference` records hashes, snapshot, exact transmitted material, omissions and the
unverified status. The saved query image is not automatically bound to this node's optional IMAGE.
File changes invalidate ComfyUI result caching. No extra encoder/model is loaded by this bridge,
but additional prompt tokens can cost latency and memory. For retrieval inside ComfyUI, use the
[native vision, memory, build and recall nodes](vrs_int/COMFYUI_NATIVE.md). A connected native state
is bound to the supplied image by an RGB pixel hash. Full dictionaries, source images, and experiences
remain private external files.

### Next development plan — 2026-10-14

Additional LoRA training is deferred in favor of VRS-assisted inference. Preserve prior adapters and
datasets. Gemma composes and renders one scene; dictionaries supply meanings and canonical vocabulary,
and retrieved experience supplies optional creative references. The saved-state bridge is implemented.
Compare the same base model, request, dictionary and exclusions with/without the reference. Save raw
outputs, hashes, delivered references, time and memory; reused material is not fresh image retrieval.

Next, improve Korean contextual vocabulary and reference selection, then integrate automatic compatible
image/text retrieval with provenance, bounded context and resource accounting. Keep Danbooru/e621
sources separate. Evaluate requested names, semantic conditions, image-only evidence, cross-output
agreement and creative usefulness, with image-swapped/absent and mismatched-reference controls.
Do not count dictionary membership as meaning or creativity. Consider training again only for residual
failures after separating retrieval, selection and rendering errors. The previous 300-request training
and dual-output LoRA proposal is deferred, not a completed data collection or an active training run.
