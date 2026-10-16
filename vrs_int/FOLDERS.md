# VRS folder memory / 폴더에서 개인 경험 만들기

## 한국어

ComfyUI 화면의 **VRS 폴더 / VRS Folders** 버튼에서 폴더를 등록하면
생성 메타데이터가 있는 이미지를 읽고 개인 검색 기억에 추가합니다.
JSONL 파일이나 별도 실행기는 필요하지 않습니다. 이 기능은 수집·검색용이며 모델을 학습시키지 않습니다.

### 시작하기

1. [네이티브 VRS 설치 안내](COMFYUI_NATIVE.md)에 따라 의존성을 ComfyUI Python에 설치합니다.
   생성 큐를 마친 뒤 서버를 재시작하고 브라우저를 새로고침합니다.
2. `models/text_encoders`에 **네이티브 Gemma 4 E2B safetensors**를 준비합니다.
   현재 VRS는 `vision_model.*`이 있는 비양자화 네이티브 파일을 사용합니다.
   TIPO용 Q4 GGUF·mmproj와는 별도입니다. **비전 가중치만 FP16으로 읽으며 언어모델은
   적재하지 않습니다.** 이 패널에서는 모델을 자동 다운로드하지 않습니다.
3. 왼쪽 아래 **VRS 폴더**를 엽니다. 새 기억 이름과 **서버 측 절대 폴더 경로**를
   한 줄에 하나씩 넣습니다. 원격 ComfyUI라면 브라우저 PC의 경로가 아닙니다.
4. **ComfyUI output 폴더도 포함**을 선택하면 서버의 실제 output 경로도 등록합니다.
   `--output-directory` 설정을 반영하며 하위 폴더도 포함합니다. 중첩된 경로는 정리합니다.
5. 올바른 Gemma 모델을 선택하고 **시작 / 재개**를 누릅니다. 기본 묶음은 4장입니다.
   새 기억 생성 시 큐 작업 하나의 최대 이미지 수를 1~32장으로 선택할 수 있습니다.
6. 첫 묶음 완료 후 **조회 노드 추가**를 누르고 아래처럼 연결합니다.

```text
VRS Folder Memory.memory ────────────────┐
VRS Native Vision Loader.encoder ────────┤
Load Image.image ───────────────────────┤
                                       ▼
                               VRS Native Recall
                                       │ vrs_state
                                       ▼
                              Gemma Prompt Expander
```

Recall의 허용 등급 `ratings`에 **`u`**를 포함하세요. Recall과 Expander에는 같은
이미지·요청을 연결하며 **구축에 선택한 것과 같은 비전 모델**을 사용합니다.
세부 연결은 [네이티브 노드 안내](COMFYUI_NATIVE.md)를 참고하세요.

### 생성 큐와 함께 쓰기

- CPU 스캔 완료 후 30초 간격으로 새 파일을 확인합니다. 큰 폴더의 스캔 시간만큼 탐지가 늦어집니다.
- GPU 특징 추출은 **기존 ComfyUI 큐의 내부 작업**입니다. 생성 중이거나 대기 작업이 있으면
  넣지 않습니다. 새 일반 생성 요청은 아직 시작하지 않은 구축 묶음보다 우선합니다.
- 실행 중 새 생성이 들어오면 **현재 이미지**를 체크포인트에 저장한 뒤 양보합니다. 모델 적재·현재 이미지·저장 시간만큼 새 생성이
  기다릴 수 있습니다. 계속 생성하면 경험 갱신은 기다립니다. 자동 Run/재큐잉 기능은 필요에 맞게 끄세요.
- **중지**는 대기 중인 구축만 취소합니다. 실행 중이면 현재 이미지와 저장을 마친 뒤 멈춥니다.
  사용자의 생성에 전역 interrupt를 보내지 않습니다.
- 오류·구축 큐 항목 삭제·서버 재시작 뒤에는 같은 기억을 선택해 **시작 / 재개**를 누르세요.
  자동 재개하지 않으며 이미 완성한 경험과 체크포인트에 저장한 벡터는 재추출하지 않습니다.
- 한 번에 한 기억을 수집합니다. 한 기억에는 여러 폴더를 넣을 수 있습니다.
  기존 기억의 경로·모델·묶음 크기를 바꾸려면 새 기억을 만듭니다.

### 저장과 검색 반영

추출 결과는 먼저 SQLite 체크포인트에 저장합니다. 첫 결과는 바로 판본으로 발행하고,
이후에는 **128장 누적·60초 경과·작업 완료·정상 중지** 시 모아서 새 판본을 발행합니다.
화면의 **저장됨·발행 대기**는 벡터까지 저장했지만 아직 판본에 포함되지 않은 수입니다.
**경험 저장됨**과 **검색 준비 완료**는 따로 표시합니다. 큐가 비면 CPU 작업자 하나가
검색 캐시를 준비하며, 새 생성이 들어오면 처리 경계에서 양보합니다. 준비 도중에는 이전
검색 판본을 유지하고 완료 뒤 전환합니다. 준비 실패는 경험 저장을 취소하지 않습니다.
생성 요청이 들어오면 판본 저장은 CPU 수집 루프로 넘겨 생성 대기를 줄입니다.
60초는 발행을 판단하는 기준으로, 모델 처리·스캔·디스크 저장 시간까지 포함한 지연 상한은 아닙니다.
서버가 갑자기 종료되면 재개 시 체크포인트부터 이어갑니다. 중지 시 저장 오류가 나면
기존 검색 판본과 체크포인트를 유지하고 오류를 표시합니다.

### 저장 내용과 한계

PNG·WebP·JPEG의 NovelAI, 지원되는 ComfyUI API prompt 그래프, A1111 `parameters`를 읽습니다.
[메타데이터 지원 범위](GENERATED_IMAGES.md)를 참고하세요. 원래 포지티브·네거티브·캡션 출처와
허용한 생성 정보를 보존합니다. 원본 이미지는 수정·삭제하지 않습니다. 워크플로를 실행하지 않으며
임의 메타데이터 전체를 복사하지 않습니다. 생성 프롬프트가 실제 이미지의 정답인 것은 아닙니다.
ComfyUI 그래프에 실행 후 생성된 문장이 저장되지 않은 경우(TIPO·일부 와일드카드 등),
그 문장을 그래프만으로 복원할 수 없습니다. 읽을 수 있는 프롬프트만 사용하므로 모든
워크플로의 최종 출력문을 자동 수집한다는 뜻은 아닙니다.

- 같은 파일 바이트는 한 번만 저장하며 변경 없는 파일은 크기·수정 시각으로 건너뜁니다.
  최근 5초 안에 수정된 파일은 기다리고 인코딩 전후 해시를 확인합니다.
- 같은 경로의 내용이 바뀌면 새 경험으로 추가합니다. 원본 삭제는 경험 삭제로 동기화하지 않습니다.
  파일·디렉터리 심볼릭 링크는 따라가지 않습니다.
- **메타데이터가 없는 파일은 집계만 합니다. PixAI 자동 태깅은 이 패널에 포함하지 않습니다.**
  기존 [선택적 태깅 CLI](GENERATED_IMAGES.md)는 별도 경로입니다.
- 이름을 추측해 분류하지 않습니다. 명시적 작가 표기는 추출하고 분류를 모르는 캐릭터·작품명도
  원래 캡션·태그에 남깁니다. 이름 분류 사전의 화면 연결은 아직 없습니다.
- 등급은 `u`(미확인)입니다. 일반 등급 판정이 아닙니다.

저장 위치는 **`<ComfyUI user 디렉터리>/rs_vrs_folders/<기억 ID>/`**입니다.
`--user-directory`를 반영합니다. `config.json`은 설정, `catalog.sqlite3`은 수집 상태와
메타데이터, `memory/`는 완성된 VRS 판본입니다. 원본 픽셀은 없지만 **프롬프트·태그·경로가
남는 개인 자료**입니다. 백업 시 수집을 중지하고 기억 폴더 전체를 복사하세요. 공개하지 마세요.

**VRS Folder Memory**는 매 실행에서 최신 **검색 준비가 끝난** 판본을 읽고 `snapshot`에 그 해시를
출력합니다. 재현하려면 기존 **VRS Native Memory Loader**에 `memory/` 경로와 그 해시를
지정합니다. 옛 PIL/NF4 기억과는 섞지 않습니다. 모델·런타임·파서 규격이 바뀌면 새 기억을 만듭니다.
이전 판본과 변경된 압축 객체는 보존되므로 용량이 누적됩니다. 자동 판본 정리는 제공하지 않습니다.
검색 벡터와 태그 변환은 영구 캐시에서 재사용합니다. 재시작 후에도 검증 후 다시 열고,
추가·수정된 경험만 변환합니다. 첫 판본이나 이전 버전 자료에 준비 기록이 없거나 캐시를 정리한
경우에는 조회 중 준비할 수 있습니다. [검색 자원·용량·복구 안내](SEARCH.md)를 참고하세요.

## English

Open **VRS Folders** at the bottom left of ComfyUI. Register absolute folders on the
**server**, optionally include its configured output directory, select a native Gemma
4 E2B safetensors file and click **Start / Resume**. Subdirectories are included.
See [native setup](COMFYUI_NATIVE.md) for dependencies; restart after your queue finishes.
The panel does not download models. Its encoder needs unquantized `vision_model.*`
weights; TIPO's Q4 GGUF/mmproj is separate. Only vision tensors are loaded, in FP16.

After the first completed batch, **Add memory node** creates **VRS Folder Memory**.
Connect its `memory` to **VRS Native Recall** with the same vision encoder and a
reference image. Connect `vrs_state` to **Gemma Prompt Expander**, sharing the image
and request. Include `u` in allowed ratings. This is retrieval, not model training.

CPU scanning checks again 30 seconds after each scan. GPU work uses bounded jobs in
the **normal ComfyUI queue**, admitted only when idle. New generation jobs take priority
over a pending import; running imports checkpoint the current image and yield. Model
loading and an already-started save may still delay a new job. Default
batch size is 4, configurable from 1 to 32 when creating a collection. Stop cancels only
a queued import or finishes the current image and save; it never globally interrupts
generation. Continuous generation delays imports. Only one collection is watched at a
time. Errors, queue removal and server restart require explicit resume. To change a
collection's roots, model or batch size, create another collection.

Supported embedded NovelAI, ComfyUI and A1111 metadata is read without executing graphs.
Runtime-generated text absent from saved metadata (for example some TIPO or wildcard
workflows) cannot be reconstructed from the graph alone; only resolvable prompts are used.
Positives, negatives and provenance remain separate. Files missing readable metadata
are counted; **automatic PixAI tagging is not included**. Unclassified names remain in
original captions/tags; there is no category dictionary selector yet. Ratings are `u`
(unknown), not an automatic safe rating. Original files are never modified or deleted.
Identical bytes deduplicate, unchanged files skip, and recently modified files wait
5 seconds. Hashes are checked around encoding. Changed content becomes another record;
deleting a source does not delete experience. Symlinks are not followed.

Collections live under `<ComfyUI user directory>/rs_vrs_folders/<collection ID>/`.
They contain private captions, tags and paths, not image pixels or language weights.
Stop importing before backing up the entire collection. Encoded vectors are durably staged in SQLite. The first result publishes immediately;
subsequent results publish after 128 images, 60 seconds, completion, or a normal stop.
“Saved, awaiting publication” counts durable vectors not yet included in search.
60 seconds is a publication trigger, not a guaranteed latency bound. When a generation
arrives, snapshot writing moves to the CPU watcher. Resume after a crash reuses staged
vectors. Publication failure preserves the previous head and the staged data.
Old snapshots are retained;
disk usage grows and no automatic pruning runs. Folder Memory follows the latest
search-ready snapshot; pin its returned hash using **VRS Native Memory Loader** for
reproducibility. Incompatible legacy PIL/NF4 memories or changed encoder/parser
contracts require a separate collection.

## Local API

Uses ComfyUI's server access controls and server-side paths:

```text
GET  /rs-vrs/folders
POST /rs-vrs/folders/start
POST /rs-vrs/folders/stop
```

POST requires `Content-Type: application/json`. Resume an existing collection with
`{"collection_id":"<UUID from status>"}`, or create one:

```json
{
  "label": "My illustrations",
  "roots": ["/absolute/path/to/my-images"],
  "include_output": true,
  "model_name": "gemma4-e2b-native-bf16.safetensors",
  "batch_size": 4
}
```

Stop accepts `{}`. Status includes counts, errors and the completed snapshot.
`VRSFolderImportBatch` is internal; do not add it to generation workflows.

Search preparation runs on one CPU worker while the queue is idle and cooperatively yields to
new generation requests. Folder Memory retains the previous search-ready snapshot until the next
one is ready. The panel shows published experiences and search-prepared counts/heads separately.
Preparation errors preserve experience and the previous search head. Persistent vectors and voting
documents are reused after restart; only changed/new entries are converted. Initial/legacy collections
without readiness records, evicted caches or changed conversion code may still prepare on demand.
An exact-hash Native Memory Loader never substitutes a different snapshot.
See [search resource, quota and recovery details](SEARCH.md).
