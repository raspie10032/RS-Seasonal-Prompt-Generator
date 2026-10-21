# Lossless memory compression / 무손실 경험 압축

[Native nodes](COMFYUI_NATIVE.md) · [Search cache](SEARCH.md) · [Folder collection](FOLDERS.md)

## 한국어

### 검색 캐시는 자동으로 줄입니다

추천 태그·분류를 SQLite에 zlib로 압축해 저장합니다. 벡터는 기존 FP32 그대로이며
전체 비교·동점 처리·자기 이미지 제외 정책은 같습니다. 준비 중 항목별 DB 조회·쓰기를
묶고, 조회 때 최종 이웃의 문서만 한 번에 읽습니다. 작가·캐릭터·작품, 원본 캡션과
감사 기록은 원본 경험에 보존합니다. 추가 패키지·CUDA 휠·외부 서비스가 필요하지 않습니다.

새 캐시 규격은 `search-cache/2`입니다. 기존 규격을 잘못 읽거나 제자리에서 바꾸지 않고,
다음 조회 때 새 세대를 만듭니다. 처음 한 번 재준비할 수 있습니다. 이전 세대는 사용 중인
검색을 보호한 뒤 기존 용량 한도·디스크 여유 정책에 따라 정리합니다. 재생성되는 동안
두 세대의 공간이 동시에 필요할 수 있습니다. 캐시에는 개인 자료가 들어 있습니다.

### 원본 경험은 별도 압축 사본을 만듭니다

ComfyUI에서 **VRS Native Memory Loader** 또는 **VRS Folder Memory**의 `memory`를
**VRS Native Compact Memory (Lossless Copy)**에 연결합니다. `output_store`에 원본 밖의
새 폴더를 지정하고, 별도 일회성 워크플로로 실행하세요. 이미지·인코더·TIPO 모델이
필요하지 않으며 네트워크·다운로드·별도 프로그램을 사용하지 않습니다.

이 노드는 큐 안에서 CPU로 XZ 압축합니다. GPU 생성과 같이 계속 실행하는 노드가
아닙니다. 오래 걸릴 수 있고 압축 해제도 zlib보다 느릴 수 있으므로 용량 절약을
우선할 때 선택하세요. 취소는 객체 처리 경계에서 확인하며, 처리 중인 한 객체의
압축·파일 쓰기는 즉시 끊지 못할 수 있습니다.

- 모든 관측·태그·분류·캡션·감사 기록·tombstone과 벡터의 원래 값을 보존합니다.
  벡터를 FP16/INT8로 바꾸지 않습니다. 인코더 계약도 유지합니다.
- 각 객체를 다시 읽어 모든 필드를 비교하고, 전체 판본을 검사한 뒤에만 발행합니다.
- 별도 저장소에 선택한 판본의 새 기반을 만듭니다. 새 해시가 생기며 이전 판본 이력과
  `CURRENT` 포인터는 복사하지 않습니다. 기존 파일·원본 이미지는 삭제하지 않습니다.
- 결과의 `store`와 `snapshot`을 Memory Loader에 지정하면 새 사본을 사용합니다.
  Folder Memory의 수집 경로는 자동 변경하지 않으며, 수집 중인 원본 대신 이 사본에
  폴더 수집을 이어 붙이는 기능은 제공하지 않습니다. 명시적인 Build/Update는 지원합니다.
- **`vrs-int/3`은 이 버전 이상에서만 읽습니다.** 이전 코드에 새 사본을 전달하지 마세요.
  기존 `vrs-int/1`, `/2`는 그대로 읽고 씁니다. 새 형식을 지원하는 상대에게는 기존
  export/install을 사용할 수 있습니다. 이전 형식과의 증분 패키지 기반으로 사용하지 마세요.
- 원본도 유지하므로 실행 직후 총 디스크 사용량은 늘어납니다. 충분한 임시 공간이
  필요하며 20% 여유 기준을 검사합니다. 중단 시 완성된 판본은 발행하지 않지만,
  새 폴더에 미사용 객체가 남을 수 있습니다. 저장률은 자료마다 다르며 XZ가 더 클 수도 있습니다.

CLI에서도 같은 로컬 함수를 사용합니다:

```bash
python -m vrs_int repack --store ./personal-memory --snapshot EXACT_SHA256 \
  --output ./personal-memory-compact
python -m vrs_int verify --store ./personal-memory-compact --snapshot NEW_SHA256
```

모델을 적재하거나 네트워크에 접속하지 않습니다. 압축은 암호화가 아니며,
원본 태그는 복원 가능합니다.

## English

Search cache format `search-cache/2` stores zlib-compressed voting documents in SQLite,
batches revision lookups and writes, and fetches only winning neighbors in one query (chunked
for SQLite parameter limits). FP32 vectors, exact search, tie order, filters and source records
remain unchanged. No extra package, CUDA wheel or service is needed. A separate cache generation
is built on first use after upgrading; existing generations remain protected while leased and
are eligible for normal quota-based eviction. Rebuilding can temporarily need space for both.

For optional stronger source compression, connect a Memory Loader or Folder Memory `memory`
output to **VRS Native Compact Memory (Lossless Copy)**. Choose a new `output_store` outside
the source, and run a separate one-time workflow. This uses CPU XZ compression in ComfyUI's
queue, without images, models, downloads, subprocesses or external services. Compression and
decompression can be slower; this option prioritizes storage. Cancellation is checked between
objects, not inside a single compression/write.

The node preserves every entry field, source caption, typed name, audit item, tombstone and
vector value without changing the encoder contract. Every object is decoded and compared before
the verified snapshot is published. It creates a new independent base for the selected snapshot,
not the entire history or activation pointer. The source and all original images remain untouched.
Load the returned `store` and `snapshot` explicitly. Folder collection does not switch automatically;
resuming folder collection into this copied store is not supported. Explicit Build/Update is supported.

The new **`vrs-int/3` format requires this version or newer**. Existing `/1` and `/2` remain
readable/writable. Export/install works between supporting versions; do not use a legacy snapshot
as the package base for the new independent store. Standard-library `lzma` provides XZ; it is
imported only for XZ objects, so older zlib stores do not require it.

Because the original is retained, total disk use increases immediately. A 20% free-space reserve
is checked, but adequate working space is still necessary. Interrupted runs publish no finished
snapshot and may leave unused objects in the new directory. Compression savings depend on the
data and XZ can be larger. Compression does not encrypt or hide source tags.

The CLI commands above call the same local operation, without loading models or connecting
to a network.

## Local comparison / 로컬 비교

On 2026-10-21, a local store with 118,787 existing 768-dimensional vectors was compared
before/after using isolated caches, two CPU threads and the same three top-32 queries.
These are stored-vector/cache measurements, not image encoding, GPU or TIPO generation timings.
The historical encoder contract remains unchanged; this comparison does not establish native
encoder compatibility. OS file caches were not flushed. Timings are individual local runs.

| Measurement / 항목 | Before / 이전 | After / 변경 후 |
|---|---:|---:|
| Search cache / 검색 캐시 | 815.1 MiB | 518.8 MiB |
| Voting document DB / 추천 문서 DB | 467.1 MiB | 170.8 MiB |
| FP32 vector pool / 벡터 저장 | 348.0 MiB | 348.0 MiB |
| Initial cache build / 첫 캐시 구축 | 83.84 s | 86.26 s |
| Same-process reopen / 같은 프로세스 재열기 | 1.51 s | 1.15 s |
| Median CPU query probe / CPU 검색 시험 중앙값 | 0.079 s | 0.085 s |
| Selected source snapshot / 선택한 원본 판본 | 494.0 MiB (zlib) | 458.9 MiB (XZ copy) |

Cache storage fell 36.3%, voting documents 63.4%, and the source copy 7.1%.
The complete decoded cache (documents, vector bytes and row metadata) and all eight query
outputs matched. Source compression verifies every field and leaves the original in place.
Compression did **not** improve initial build or CPU query time in this run; the measured
speed benefit is reopen preparation. XZ additionally trades compression/decompression time
for storage. Do not extrapolate these figures to image recognition or another machine.

검색 캐시는 36.3%, 추천 문서 DB는 63.4%, 원본의 별도 압축 사본은 7.1% 줄었습니다.
전체 복원 문서·벡터 바이트·행 메타데이터와 검색 시험 8회의 결과가 일치했습니다.
원본 경험과 기존 인코더 계약은 보존합니다. 첫 구축과 CPU 검색 자체는 빨라지지 않았고,
이번 속도 이점은 재열기 준비에 한정됩니다. XZ는 저장 용량을 위해 압축·해제 시간을
추가로 쓰는 선택입니다. 이 수치를 이미지 인코딩·GPU 추론 속도로 해석하지 마세요.
