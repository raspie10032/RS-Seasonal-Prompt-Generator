# Personal tag dictionary / 개인 태그 의미 사전

VRS-INT can read a local four-column CSV as a **base text knowledge pool and
translation vocabulary**, independently of image memory. Entries connect
English tags, Korean names/keywords, definitions and semantic categories. The
same pool can also explain and classify tags retrieved from image experience.

개인 태그 CSV를 **기본 텍스트 지식풀 + 번역 사전**으로 연결합니다.
이미지가 없어도 한국어 개념으로 영문 태그·뜻·분류를 찾고, 영문 태그를 한국어로
풀이할 수 있습니다. 이미지 경험과 함께 쓸 때는
**한국어 요청 → 용어 지식 → 이미지 경험의 참고 재료 → 생성 상태**로 연결합니다.
이미지를 다시 인코딩하지 않아도 작가·캐릭터·작품·일반 태그 분류와 한국어
설명을 붙일 수 있습니다. 사전에 이름이 있다고 이미지에서 그 인물을 새로
알아보게 되는 것은 아닙니다.

## Input and lookup / 입력과 검색

Use a UTF-8 CSV, optionally with a BOM, **without a header**:

```csv
example_artist,1,25,"[작가 > 예시] 가상의 작가. 키워드: <작가>, 예시화가"
example_character,4,15,"[캐릭터 > 예시] 가상의 인물. 키워드: <인물>, 예시인물"
scarf,0,100,"[패션 > 액세서리] 목도리. 키워드: <의상>, 목도리"
```

The fields are English tag, category ID, nonnegative post count, and Korean
description. Quote fields containing commas using normal CSV quoting.
Categories: `0` general, `1` artist, `3` copyright, `4` character, `5` meta.
Underscores/spaces and case normalize for lookup. Duplicate normalized tags,
unsupported categories, malformed rows and an empty dictionary are rejected.
The file is read once into process RAM; GPU memory is not used by the dictionary.

분류 번호는 `0` 일반, `1` 작가, `3` 작품, `4` 캐릭터, `5` 메타입니다.
`[인물 > 눈]`처럼 대괄호에 든 분류와 `키워드:` 뒤의 검색어를 읽습니다.
분류 표기의 공백과 일부 동의 표현을 정리하지만, 모르는 분류를 추측하지는
않습니다. 자료는 개인 경로에 두고 다음처럼 확인합니다.

```bash
python -m vrs_int.tag_dictionary --csv /path/to/personal-tags.csv
python -m vrs_int.tag_dictionary --csv /path/to/personal-tags.csv --query "목도리"
python -m vrs_int.tag_dictionary --csv /path/to/personal-tags.csv --query "blue eyes"
python -m vrs_int.tag_dictionary --csv /path/to/personal-tags.csv \
  --text "금발과 벽안은 유지하고 겨울에 코트를 입혀줘. 목도리는 제외."
```

Exact English tags and Korean keyword aliases rank before substring matches.
Post counts only break lookup ties; they never boost image-retrieval votes.
Multiple meanings remain alternatives. A Korean sentence is **not automatically
translated into generation controls**.

한국어 검색은 후보를 찾는 기능입니다. 같은 검색어에 여러 태그가 대응할 수
있으므로 자동 확정하지 않습니다. 아래 `--keep`·`--target`에는 선택한 영어
태그를 직접 지정합니다. 문장 전체의 의미를 판정하거나 자연스러운 영문 문장으로
번역하지는 않습니다.

`--text` retrieves the longest matching terms inside a sentence, including
English tags and Korean keyword aliases. Spaces/underscores normalize; basic
Korean particles are supported. It is a lightweight lexical matcher, not a
Korean morphological parser. The returned `vrs-int-text-knowledge/1` packet
preserves source spans, Korean definitions, all candidate counts, ambiguity,
unmatched text and dictionary provenance. `--limit` bounds alternatives per
mention, recording any truncation. Input is limited to 4096 characters.

`<우산>`처럼 꺾쇠에 든 분류어도 약한 검색 근거로 쓸 수 있지만, 정확한 번역
동의어로 확정하지 않습니다. 동명이인·같은 검색어는 후보를 남깁니다.
`목도리는 제외` 안의 `목도리`도 뜻을 찾되, **긍정 프롬프트에 넣을 태그로
자동 변환하지 않습니다.** 모든 언급의 역할은 `unresolved`이며 유지·추가·제외는
원문과 명시한 조건으로 판단해야 합니다. 미등록 문구와 부정 표현은 보존됩니다.

When `build_state` receives a Korean request and an explicit dictionary, it saves
this packet as `request_knowledge`. `messages_for` includes a bounded bilingual
vocabulary (up to 32 mentions, 3 meanings each, 240 description characters) in all
three trial modes. The original request and explicit controls remain intact.
Term knowledge is separate from neighbor votes and visual identity evidence.
Model translation quality and intent handling still require evaluation.

When the highest-ranked meaning is a general concept, its compact writer entry
omits character/artist/work matches sharing that descriptive keyword. All such
matches remain in the full packet, with omission counts in the writer input.
This limits unrelated-name contamination; it is not full contextual word-sense
disambiguation. Korean keyword lists contain related concepts, not only synonyms.

## Annotate image recommendations / 이미지 추천에 연결

Add the optional `--tag-dictionary` argument to an ordinary query:

```bash
python -m vrs_int recommend \
  --store /path/to/personal-memory --snapshot EXACT_SNAPSHOT_HASH \
  --encoder /path/to/encoder --image /path/to/query.png --gpu GPU-YOUR-UUID \
  --ratings u --neighbors 32 --limit 100 \
  --tag-dictionary /path/to/personal-tags.csv \
  --output /path/to/new-recommendation.json
```

Replace `EXACT_SNAPSHOT_HASH` with the stored 64-character snapshot ID. `u` is
appropriate for unrated personal generations; select your actual rating scope.
The output path must be new. Normal CUDA/encoder requirements still apply to
image retrieval. The CSV annotation itself needs only Python's standard library.

사전은 최종 후보 수를 자르기 전에 분류를 보완하므로, 이전에 `unknown`이던
작가 이름도 작가 후보로 분리할 수 있습니다. 원래 명시된 분류와 사전 분류가
다르면 원래 분류를 유지하고 `category_conflict`로 기록합니다.

Candidate `score`, `support`, observation keys, image similarities and
`visual_verdict: "U"` retain their original meaning. The CSV adds
`source_categories`, `dictionary` and file-hash provenance. It does not create
new image evidence. Queries without this option keep their previous behavior.

## Use existing results / 저장한 검색 결과로 상태 만들기

You can annotate a saved recommendation directly, with no image/GPU access:

```bash
python -m vrs_int.generation_state \
  --recommendation /path/to/recommendation.json \
  --tag-dictionary /path/to/personal-tags.csv \
  --request "붉은 머리는 유지하고 겨울 공원을 산책하는 장면으로 바꿔줘" \
  --keep "red hair" --target winter --target park --target walking \
  --target coat --exclude indoors --resolve \
  --output /path/to/new-state.json
```

`--resolve` produces a conservative **single-subject creative recipe**:

1. Keep all classified candidates and supporting observations in the state,
   after the usual quality/year filter and explicit exclusions.
2. Include identity candidates in the writer input only when named in explicit
   `keep` or `target` controls. Other artists, characters and works remain in
   the inspection data.
3. Choose one reference observation, preferring one supporting the selected
   character and then other `keep` traits. Use its co-occurring optional tags.
4. When a target changes clothing, action or setting, omit optional old tags in
   that whole group. Explicit `keep` tags remain required.
5. Choose at most one value for known conflicting slots: basic hair/eye colors,
   season, indoor/outdoor, time, pose and framing. Keep distinguishing hair and
   eye traits ahead of generic appearance tags, then apply group quotas.

`--resolve`를 빼면 사전 분류만 적용합니다. 선택 모드는 `writer_reference`에
TIPO용 재료를 따로 두고, 제외 이유와 참고 관측 ID를 기록합니다. 전체 후보는
상태의 `identity_candidates`와 `reference_groups`에 남습니다. 품질·연도 필터와
명시적인 `--exclude`는 선택 전에도 적용됩니다.

The selection rules are deliberately incomplete. Co-occurrence is not proof of
visual consistency. Group replacement can omit useful old accessories; list
essential ones in `--keep`. Multi-subject scenes should use candidate mode.
Conflicting explicit single-subject colors raise an error; explicit
`heterochromia` or `multicolored hair` controls relax their respective color slot.
Terms absent from the CSV remain valid user controls.

사전에 없는 사용자 표현은 삭제하지 않습니다. 선택 규칙은 색상 등 일부
충돌만 다루며 이미지 내용을 판정하지 않습니다. TIPO가 이후 새 모순을 만들거나
필수 태그를 빠뜨릴 수 있으므로 출력도 확인해야 합니다.

Dictionary-backed states use `vrs-int-generation-state/2`, recording the CSV's
SHA-256, byte size and row/category counts. Keep that CSV together with your
private run configuration to repeat future queries. A memory export contains
the image experience, **not this optional external CSV**. Changing the CSV
requires starting from the original recommendation, not stacking two different
dictionary annotations. Existing version-1 state generation remains available.

## Compare TIPO outputs / 실제 TIPO 비교

Use the [generation trial guide](GENERATION_STATE.md) and add
`--tag-dictionary /path/to/personal-tags.csv --resolve` to
`python -m experiments.run_generation_state_trial`. It still compares request
only, flat reference tags, and structured state with the same selected reference
tag set. Set `CUDA_VISIBLE_DEVICES=GPU-...` before launch. Dictionary-only and
resolved runs need distinct output directories and identical cases/seeds to
compare fairly. No image or model hidden state is sent to TIPO. It receives
selected tags, controls and text structure; Korean requests additionally include
the bounded bilingual vocabulary described above.

단순 사전 연결 효과와 재료 선택 효과를 구분하려면 같은 요청·시드로
`사전 없음 / 사전만 / 사전 + --resolve`를 각각 실행합니다. 후보가 더 많이
분류된 것과 최종 프롬프트가 좋아진 것은 별도로 평가합니다.
