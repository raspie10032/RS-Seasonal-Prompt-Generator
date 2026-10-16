# VRS search resources / 검색 자원과 영구 캐시

[Native workflow](COMFYUI_NATIVE.md) · [Folder ingestion](FOLDERS.md)

## 한국어

VRS는 이미지 인코딩과 검색을 구분합니다. 아래 변경과 측정은 **이미 저장된 특징 벡터의
검색 준비**에 해당합니다. 비전 추출이나 TIPO 문장 생성의 속도·품질 개선 수치가 아닙니다.

- 처음 사용하는 경험은 원본을 검증하고 FP32 벡터와 추천 태그를 디스크에 저장합니다.
- 새 기억 객체나 서버 재시작 후에도 저장 결과를 재사용합니다. 원본 압축 객체와 캐시의
  해시는 매번 확인하지만, 변경 없는 객체의 해독·태그 해석·DB 쓰기는 반복하지 않습니다.
- 새 판본에서는 변경된 저장 묶음을 읽고, 경험별 내용 해시로 기존 변환 결과를 재사용합니다.
  추가·수정된 경험만 변환하며, 벡터와 태그 DB를 판본마다 통째로 복사하지 않습니다.
- 각 판본은 자기 행 목록을 사용합니다. 수정·삭제·학습 제외 항목은 새 판본의 검색에서 빠지고,
  이미 열린 이전 판본의 결과는 유지됩니다. 원본 경험과 인코더 계약은 변경하지 않습니다.
- 검색은 전체 벡터를 제한된 GPU 묶음으로 정확 비교하며 최종 이웃의 태그만 읽습니다.
  자기 이미지 제외·등급 필터 후 중복 제거·동점 순서를 유지합니다. 근사 검색은 아닙니다.

### 위치, 용량과 복구

기본 위치는 Linux의 `${XDG_CACHE_HOME:-~/.cache}/rs-vrs-int/search/`, Windows의
`%LOCALAPPDATA%\rs-vrs-int\search\`입니다. **ComfyUI 시작 전에 `RS_VRS_CACHE_DIR`**을
설정해 위치를 바꿀 수 있습니다. 변경하면 새 위치에 캐시를 만들며 이전 위치는 자동 이동하지 않습니다.
캐시에는 개인 태그가 있으므로 원본 경험과 마찬가지로 개인 자료로 취급하세요.

**`RS_VRS_CACHE_MAX_GIB` 기본값은 8GiB**입니다. 디스크 여유 20%도 확인합니다.
한도에 가까워지면 사용하지 않는 오래된 캐시 세대를 정리합니다. 실행 중인 검색에서 사용 중인
세대와 현재 확장 중인 세대는 삭제하지 않습니다. 정리로 공간을 확보하지 못하면 준비를 중단하고
한도·위치 변경을 안내합니다. 같은 세대에 축적된 과거 항목을 자동 압축·정리하지는 않습니다.
**원본 판본과 원본 이미지는 정리 대상이 아닙니다.** 캐시만 지우면 다음 사용 때 재생성합니다.
수동 삭제는 모든 ComfyUI/VRS 프로세스를 종료한 뒤 하세요. 이전 버전의 `session-*`도 이때 정리할 수 있습니다.

원본 내용·인코더 계약·변환 코드가 맞는 결과만 공유합니다. 프로세스 간 OS 잠금으로 준비를
직렬화하고 사용 중인 캐시는 별도 잠금으로 보호합니다. 정상 중단은 추가 쓰기를 되돌립니다.
강제 종료·손상으로 캐시의 해시가 맞지 않으면 별도 세대로 다시 만들며, 불완전한 결과는 사용하지 않습니다.
동일 환경에서 여러 번 열어도 공간을 중복 사용하지 않습니다. Windows 잠금 경로는 포함하지만
이번 실제 실행 검증 환경은 Linux입니다. Windows에서는 열린 메모리 매핑이 파일 크기 변경을
막으므로 벡터 풀을 상시 매핑하지 않고 요청한 행만 묶어 읽습니다. Linux/macOS에서는 읽기 전용
메모리 매핑을 사용합니다. Windows용 행 읽기의 결과·추가·롤백 검사는 Linux에서도 수행하지만,
이는 Windows 실기 실행이나 성능 검증을 대신하지 않습니다.

FP32 벡터는 `고유 변환 항목 수 × 차원 × 4바이트`이며 태그 DB·행 목록 등은 별도입니다.
INT8 원본은 검색용 FP32 배열이 더 큽니다. 변경·삭제 전 항목도 이전 판본 재현을 위해 캐시에
남을 수 있으므로, 현재 판본 건수만으로 디스크 크기를 계산하지 마세요.

### 폴더 자동 수집

큐가 비면 CPU 작업자 하나가 새 판본의 검색 자료를 미리 준비합니다. 중지나 새 생성 요청을
처리 경계에서 확인하고 양보합니다. 이미 진행 중인 파일/DB 처리와 무결성 기록은 끝날 때까지
기다릴 수 있으므로 즉시 중단이나 일정 지연 상한을 보장하지 않습니다.
**VRS Folder Memory**는 마지막으로 준비를 마친 판본을 사용하고, 준비가 끝난 뒤 전환합니다.
화면에서 경험 저장 수·검색 준비 완료 수·각 판본을 구분합니다. 첫 판본이나 이전 버전 자료에
준비 기록이 없으면 기존 방식대로 조회 중 준비할 수 있습니다. 캐시를 지우거나 파서가 바뀌면
준비 완료 기록이 있어도 다시 검증·생성합니다. 특정 해시를 지정한 **Native Memory Loader**는
언제나 그 해시를 사용하며 다른 판본으로 바꾸지 않습니다. 수집은 재시작 후 직접 재개합니다.

### 로컬 측정

118,787개 실제 저장 벡터(768차원, FP32), RTX 4070 Ti SUPER, CPU 스레드 2개에서
별도 프로세스로 같은 6회 조회를 비교했습니다. OS 파일 캐시를 비우거나 시스템을 재부팅하지는 않았습니다.
GPU 문맥 초기화 후 검색을 측정했으며, 후속 조회는 5회 중앙값입니다.

| 항목 | 이전 임시 캐시 | 영구 캐시 첫 생성 | 영구 캐시 새 프로세스 재사용 |
| --- | ---: | ---: | ---: |
| 첫 준비 포함 조회 | 63.80초 | 76.93초 | 1.63초 |
| 후속 조회 | 0.075초 | 0.080초 | 0.075초 |
| 검색용 디스크 | 약 815MiB | 약 815MiB | 같은 파일 재사용 |

첫 생성은 영구 기록 비용으로 느려졌습니다. 이 변경의 효과는 **재시작과 판본 추가 시 재사용**입니다.
6회 반환 결과의 해시는 모두 일치했고 원본 1,025개 파일의 해시·크기·수정 시각도 유지됐습니다.
GPU 텐서 최대 할당은 약 20.1MiB이고 프로세스 RSS는 약 1.14GiB로, 비전·언어 모델은 제외한 수치입니다.

같은 실자료의 분리된 복사본에 시험 항목을 추가한 경우, 검색 준비는 1건 **1.86초**,
4건 **2.10초**, 128건 **5.15초**였습니다. 원본 판본 발행 비용은 별도로 각각 0.06·0.21·5.76초입니다.
1→4→128건을 누적 재사용해 변환한 새 항목 수는 각각 1·3·124개이며, 128건까지의 검색 캐시 증가량은
약 **1.05MiB**였습니다. 추가 항목은 기존 벡터에 별도 시험 ID를 준 것이며 새 이미지 인코딩 시험은 아닙니다.

대규모 측정은 옛 PIL/NF4 계약의 저장 벡터를 직접 검색한 결과로 네이티브 인코더 호환성 증거가 아닙니다.
계약을 우회하지 마세요. 실제 네이티브 연결은 작은 별도 자료로 검사합니다. 수치는 디스크·OS 캐시·태그
길이·하드웨어에 따라 달라집니다. 전체 해시 검증과 정확 선형 검색 비용은 여전히 자료량에 비례합니다.

## English

Search persists a shared FP32 vector pool and SQLite voting documents across memory objects,
snapshots and process restarts. Unchanged compressed source objects and derived files are hashed
on every open, but no longer unpacked, decoded or reinterpreted. Changed objects reuse individual
entry revisions; only new/changed entries are converted. Each snapshot selects its own rows, so
edits, deletions and evaluation exclusions cannot leak older entries into a new snapshot. Already
open snapshots retain their results. Exact bounded GPU search, filters, deduplication and tie order remain.

Set **`RS_VRS_CACHE_DIR` before launching ComfyUI** to move the cache. Defaults are
`${XDG_CACHE_HOME:-~/.cache}/rs-vrs-int/search/` on Linux and
`%LOCALAPPDATA%\rs-vrs-int\search\` on Windows. The directory contains private tags.
**`RS_VRS_CACHE_MAX_GIB` defaults to 8GiB** and a 20% free disk reserve is checked. Unused old
cache generations can be evicted; live readers and the generation being extended are protected.
If enough space cannot be reclaimed, preparation stops with an actionable error. Historical rows
within the current generation are not automatically compacted. Original experiences are never evicted.
Close all VRS/ComfyUI processes before manually removing caches (including old `session-*` remnants).
The next use rebuilds them. Moving the cache path does not move/delete the old location.

OS locks serialize preparation across processes and protect active readers. Normal cancellation rolls
back appended rows; incomplete or damaged cache generations are rebuilt without accepting partial data.
Source, encoder contract and conversion-code identity control reuse. Budget `unique revisions × dimension × 4`
bytes for vectors plus metadata and voting documents. Superseded rows may remain for old snapshots.
The Windows locking path is implemented, but live measurements here were on Linux.
Windows reads requested vector rows in bounded contiguous groups instead of keeping
the growing pool memory-mapped: open mappings can block resizing during append rollback.
Linux/macOS retain read-only memory mapping. Row-read equivalence, append and rollback
tests also run on Linux; these do not establish Windows runtime performance.

Folder ingestion prewarms on one CPU worker while the queue is idle and cooperatively yields when a
normal generation arrives or collection stop is requested. Ongoing file/DB operations and sealing can
still delay yielding. Folder Memory selects the last search-ready snapshot until the next one is ready;
the panel distinguishes published and search-prepared counts/heads. An initial/legacy collection without
readiness records can prepare on demand. After cache eviction or code changes it may prepare again even
if readiness was recorded. A Memory Loader pinned to an exact hash never substitutes another snapshot.
Collection watching still requires explicit resume after a server restart.

On 118,787 real 768-dimensional vectors, RTX 4070 Ti SUPER and two CPU threads, initial lookup took
**63.80s** with the previous disposable cache, **76.93s** to create the persistent cache, and **1.63s**
in a fresh process reusing it. Median subsequent query latency stayed around **0.075–0.080s**. Disk
usage is about **815MiB**, shared across snapshots, and all six complete result hashes matched.
All 1,025 source files retained their hashes, sizes and mtimes. Search CUDA allocation was about
20.1MiB and process RSS about 1.14GiB; vision/language models are excluded. OS file caches were not flushed.

Adding 1, 4 and 128 synthetic identities to an isolated copy of the real store required **1.86s, 2.10s
and 5.15s** of preparation (snapshot publication separately took 0.06s, 0.21s and 5.76s). Sequential reuse
converted only 1, 3 and 124 new entries; total cache growth for 128 was **1.05MiB**. This uses copied
vectors, not freshly encoded images. Real-corpus measurement used a historical PIL/NF4 contract;
it does not make those vectors native-compatible. Native integration is tested separately on small fixtures.
This optimizes restart/delta preparation, not initial conversion, image recognition, generation quality,
or the O(N) exact search/hash checks. Performance depends on hardware, disk, OS cache and metadata.
