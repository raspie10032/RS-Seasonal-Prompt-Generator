# VRS folder memory / 폴더에서 개인 경험 만들기

## 한국어

ComfyUI 화면의 **VRS 폴더 / VRS Folders** 버튼에서 폴더를 등록하면
생성 메타데이터가 있는 이미지를 읽고 개인 검색 기억에 추가합니다.
JSONL 파일이나 별도 실행기는 필요하지 않습니다. 이 기능은 수집·검색용이며 모델을 학습시키지 않습니다.

개인 자료를 외부 저장소로 업로드·동기화하지 않습니다. 본인 PC의 ComfyUI를 사용하세요.
원격 서버와 다운로드·Git 제외 보호의 범위는 [로컬 자료 안내](../LOCAL_DATA.md)를 참고하세요.

### 시작하기

1. [네이티브 VRS 설치 안내](COMFYUI_NATIVE.md)에 따라 의존성을 ComfyUI Python에 설치합니다.
   생성 큐를 마친 뒤 서버를 재시작하고 브라우저를 새로고침합니다.
2. 기본 모델 **`(auto: Gemma FP16 vision)`**은 첫 큐 작업에서 분리된 FP16 비전 약 **319.3MiB**를
   자동 다운로드합니다. TIPO가 받은 정상 mmproj와 GGUF 파서가 이미 있으면 재사용합니다. 비전 가중치 약 **319MiB**만
   읽고 언어·오디오는 적재하지 않습니다. 기존 `vision_model.*` 네이티브 Gemma safetensors도
   선택할 수 있습니다. 새 비전 파일에는 GGUF 파서·llama.cpp·언어모델이 필요 없습니다.
3. 왼쪽 아래 **VRS 폴더**를 엽니다. 새 기억 이름과 **서버 측 절대 폴더 경로**를
   한 줄에 하나씩 넣습니다. 원격 ComfyUI라면 브라우저 PC의 경로가 아닙니다.
4. **ComfyUI output 폴더도 포함**을 선택하면 서버의 실제 output 경로도 등록합니다.
   `--output-directory` 설정을 반영하며 하위 폴더도 포함합니다. 중첩된 경로는 정리합니다.
5. 올바른 Gemma 모델을 선택합니다. 메타데이터 없는 이미지도 넣으려면 **PixAI 보완 태깅**을 선택하고
   ComfyUI의 Python으로 `requirements-tagger.txt`를 설치한 뒤 서버를 재시작하세요.
   선택적으로 **서버 재시작 후 자동 재개**를 켜고 **시작 / 재개**를 누릅니다. 기본 묶음은 4장입니다.
   새 기억 생성 시 큐 작업 하나의 최대 이미지 수를 1~32장으로 선택할 수 있습니다.
6. 첫 묶음 완료 후 **VRS → TIPO 연결 추가**를 누르면 기억·비전·조회·확장 노드가 연결된 상태로 추가됩니다.
   Load Image를 Recall의 image에 연결하고 Recall에 요청을 입력하세요. **조회 노드 추가**는 기억 노드만 추가합니다.

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
- 오류·구축 큐 항목 삭제 뒤에는 같은 기억을 선택해 **시작 / 재개**를 누르세요.
  서버 재시작 후에는 새 기억에서 **자동 재개**를 선택했던 마지막 활성 수집만 재개합니다. 기본은 수동입니다.
  **중지**는 자동 재개 예약도 해제합니다. 서버 종료는 예약을 보존합니다.
  이미 완성한 경험과 체크포인트의 벡터·PixAI 예측은 재계산하지 않습니다.
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
- 메타데이터 없는 파일은 기본으로 집계만 하며, 새 기억에서 **PixAI 보완 태깅**을 선택하면 로컬 예측 태그로 경험을 만듭니다.
  원본 프롬프트와 구분하고 원본 이미지는 수정하지 않습니다. [기존 CLI](GENERATED_IMAGES.md)도 사용할 수 있습니다.
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
**server**, optionally include its configured output directory, select **`(auto: Gemma FP16 vision)`**
or an existing native Gemma 4 E2B safetensors file, then click **Start / Resume**. Subdirectories are included.
See [native setup](COMFYUI_NATIVE.md) for dependencies; restart after your queue finishes.
Auto downloads a pinned **319.3 MiB FP16 vision safetensors** inside the first queue job.
Only **~319 MiB vision tensors** are loaded; no language/audio weights are loaded.
No GGUF parser or llama.cpp is needed for the standalone file. A verified cached TIPO mmproj
is reused when the optional GGUF parser is available. Manual native files need
`vision_model.*` keys. Unrelated models and legacy extracted encoders are filtered out.

After the first completed batch, **Add VRS → TIPO workflow** adds connected memory, vision, recall and expansion nodes.
**Add memory node** creates only **VRS Folder Memory**.
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
time. Errors and queue removal require explicit resume. Restart resumes only the last active collection
that opted into auto-resume; Stop clears that reservation. The default is manual resume. To change a
collection's roots, model or batch size, create another collection.

Supported embedded NovelAI, ComfyUI and A1111 metadata is read without executing graphs.
Runtime-generated text absent from saved metadata (for example some TIPO or wildcard
workflows) cannot be reconstructed from the graph alone; only resolvable prompts are used.
Positives, negatives and provenance remain separate. Files missing readable metadata
are counted by default; optional **PixAI fallback** creates local predicted-tag experience instead. Unclassified names remain in
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

## Automatic vision download / 자동 비전 준비

Auto uses `models/text_encoders/rs_vrs_vision/vrs-gemma4-e2b-vision-fp16.safetensors`
from the [public vision-only repository](https://huggingface.co/raspie/vrs-int-gemma4-vision-fp16),
with pinned revision, size and SHA-256. An already installed standalone file takes priority;
otherwise a verified cached TIPO `rs_gemma_q4/mmproj-F16.gguf` is reused if GGUF support is installed.
Only public model files are requested; no image, metadata or experience is sent.
If a generation arrives during transfer, the current file read yields and the partial
file is removed. The next idle job retries from the beginning; HTTP range resume is
not implemented. Stop waits for an in-flight network read (up to the network timeout).
Set `RS_MODELS_OFFLINE=1` to reuse local verified assets without new downloads.

자동 비전은 분리된 공개 비전 가중치만 받으며 개인 입력을 전송하지 않습니다.
전송 중 생성 요청이 오면 읽기 경계에서 양보하고 임시 파일을 지웁니다. HTTP 이어받기는
없으므로 다음 큐에서 처음부터 다시 받습니다. 중지 시 진행 중 네트워크 읽기는 타임아웃
최대 60초까지 기다릴 수 있습니다. `RS_MODELS_OFFLINE=1`이면 새 다운로드를 거부합니다.

## PixAI fallback and runtime prompt saving / 보완 태깅과 최종 프롬프트 저장

### 한국어

- **PixAI는 선택 기능**입니다. ComfyUI Python으로 `python -m pip install -r requirements-tagger.txt`를 실행하세요.
  PyTorch를 교체하거나 실행 중 pip를 호출하지 않습니다. 첫 사용은 고정 리비전의 공개 모델 약 **1.81GiB**를 받습니다.
  기존의 검증 가능한 `models/rs_wd_tagger` 캐시를 재사용하며, 이후 로컬 캐시로 실행합니다.
  `RS_MODELS_OFFLINE=1`이면 새 다운로드를 막습니다. ComfyUI의 PyTorch 가속기를 사용하며 CPU 태깅은 자동 실행하지 않습니다. 추가 GPU 제품군의 전체 모델 호환성은 미검증입니다.
- 원래 프롬프트를 읽을 수 있는 이미지는 그 내용을 우선합니다. 읽을 수 없을 때만 PixAI를 호출합니다.
  예측은 `pixai-visual-tag-prediction` 출처로 구분하고 이미지마다 즉시 체크포인트에 남깁니다.
  추정 작가명은 스타일 후보이지 실제 작가 확인이 아닙니다. 무확신·오류는 상태에 표시하며 임의 캡션을 만들지 않습니다.
- 태거와 VRS 비전은 모두 기존 ComfyUI 큐와 메모리 관리자를 사용합니다. 생성 요청이 오면 현재 이미지 뒤에 양보합니다.
  **시험 1장의 태깅·인코딩 합산 PyTorch 최대 할당량 약 2.25GiB**는 최소 사양이 아닙니다.
  다른 모델과 더 큰 입력이 있으면 실제 사용량이 늘어납니다.
- **VRS Save Image (Final Prompts)** 노드에 생성기와 같은 최종 positive/negative 문자열을 연결하세요.
  TIPO·와일드카드·문자열 합성을 끝낸 실제 텍스트를 PNG에 저장하며, 폴더 수집은 이를 그래프의 기본값보다 우선합니다.
  일반 Save Image를 자동 수정하지 않습니다. `--disable-metadata`를 사용하면 이 메타데이터도 저장되지 않습니다.
  프롬프트는 저장 이미지 안에 읽을 수 있는 형태로 남으니 개인 문구가 있다면 공유 전에 제거하세요.
- TIPO 네이티브 노드는 입력 검사·모델 준비·태그/자연어 생성·완료/빈 입력/실패를 표시합니다.
  `both`는 순차 생성 2회이며, 오류 상세는 ComfyUI 실행 오류를 확인하세요.

### English

- **PixAI is opt-in.** Install `requirements-tagger.txt` with ComfyUI's Python and restart.
  No runtime pip or PyTorch replacement. First use downloads ~**1.81 GiB** of fixed-revision public model files.
  Verified `models/rs_wd_tagger` caches are reusable; `RS_MODELS_OFFLINE=1` prevents new downloads.
  Uses ComfyUI's PyTorch accelerator; there is no automatic CPU tagging fallback. Full-model compatibility on additional GPU backends is unverified.
- Readable original prompt metadata takes priority. Predictions are explicitly marked
  `pixai-visual-tag-prediction` and checkpointed after every image. Artist predictions are style candidates,
  not authorship identification. Low-confidence results and errors are reported rather than invented.
- Tagging and vision use ComfyUI's queue and memory manager, yielding after the current image.
  One synthetic-image probe used **~2.25 GiB peak PyTorch allocation** across tagging and encoding;
  this is not a minimum VRAM requirement or a bound for other workloads.
- Connect the generator's exact final positive/negative strings to **VRS Save Image (Final Prompts)**.
  Runtime TIPO/wildcard/concatenation text is stored in PNG metadata and preferred over graph defaults on import.
  Standard Save Image is unchanged. `--disable-metadata` disables these fields too.
  Saved prompts remain readable in the image; strip private metadata before sharing.
- **Add VRS → TIPO workflow** creates and connects memory, vision, recall, and expansion nodes.
  Attach Load Image to Recall and enter the request there. TIPO shows validation/preparation/generation/status.
  `both` makes two sequential calls.
- Optional **Resume after server restart** restores the last active opted-in collection. Default is manual.
  Stop clears the resume marker; server shutdown preserves it. Completed embeddings and tag predictions are reused.

기존 기억도 **중지 → PixAI/자동 재개 옵션 변경 → 시작 / 재개**로 바꿀 수 있습니다. 이미 저장한 벡터는 보존하고 이전에 건너뛴 파일만 다시 확인합니다.
Existing collections support **Stop → change PixAI/auto-resume options → Start / Resume**. Saved embeddings are preserved; previously skipped images are rechecked.

### CPU 선택 / CPU choice

**CPU 추론에 동의**는 기본 해제입니다. 선택하면 이 수집의 비전·PixAI만 CPU로 실행합니다.
GPU 실패 시 자동 전환하지 않으며, 다른 프로그램이 느려질 수 있습니다. 수집 중에는 변경할 수
없고 이미 인코딩된 기억의 CPU/GPU를 바꾸려면 새 수집을 만드세요. 저장된 선택은 별도로 동의한
자동 재개에도 유지됩니다. 자동 생성한 조회 그래프에도 `use_cpu`가 전달됩니다.

CPU inference is opt-in and defaults off. It applies to this collection's vision and PixAI only.
It may affect other work and is never an automatic fallback. Stop before changing options; after
encoding, create a new collection to change CPU/GPU. Saved consent also applies to separately enabled
auto-resume, and the generated recall workflow carries the same `use_cpu` choice.

PixAI를 끄면 이전에 남은 태깅 대기 작업도 실행하지 않습니다. 대기 목록은 보존하며 다시
선택하면 이어갑니다. CPU/GPU 변경 제한은 아직 공개 snapshot에 합치지 않은 벡터에도 적용합니다.
Disabling PixAI suspends existing pending tagging jobs without deleting them; re-enabling resumes them.
The CPU/GPU contract guard also covers checkpointed embeddings not yet published in a snapshot.
