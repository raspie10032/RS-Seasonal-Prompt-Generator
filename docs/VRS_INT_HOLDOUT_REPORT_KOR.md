# VRS-INT 홀드아웃 추천 품질 평가 추가 기술 보고서

작성일: 2026-10-12 (한국 기준)
대상: `raspie10032/RS-Seasonal-Prompt-Generator`
기준 커밋: `3c78de102f5dc0a793ba006725dbf3a20d6b8625`
구현 위치: `vrs_int/evaluate.py`, `vrs_int/recommend.py`, `vrs_int/__main__.py`

## 1. 결과와 검증 범위

고정된 불변 스냅샷의 `val` 또는 `test` 특징 벡터를 질의로 사용하여 추천 품질을
계산하는 `evaluate` 명령을 추가했다. 기본 NumPy 모드에서는 이미지·인코더·CUDA를
사용하지 않으며, 실제 이미지 품질은 별도의 개인 환경 검증 대상으로 남긴다.

추천 결과는 기존과 같은 이웃 관측의 가중 투표다. 확률 보정, 작가 식별 확률,
SWEGCA의 승인/반려/기권 판정을 추가하거나 재해석하지 않았다.
기존 `visual_verdict: U` 필드는 그대로 유지되며 평가 판단으로 사용하지 않는다.

기존 31개와 신규 12개를 합한 오프라인 테스트 43개를 실행하여 통과했다.
이 수치는 실제 이미지 추천의 Precision 또는 Recall이 높다는 의미가 아니다.
실제로 검증한 것은 평가 계산, 검색 근거 격리, 저장된 합성 특징의 추천 계산,
기존 기억 동작의 회귀 보존이다.

## 2. 공개 기준 확인

- [저장소](https://github.com/raspie10032/RS-Seasonal-Prompt-Generator)
- [최초 공개 커밋](https://github.com/raspie10032/RS-Seasonal-Prompt-Generator/commit/74956083907b2d431fafd4d4963564d773a8864a)
- [기준 커밋의 성공 CI](https://github.com/raspie10032/RS-Seasonal-Prompt-Generator/actions/runs/38198348245)

GitHub 조회 및 로컬 checkout에서 기준 커밋과 성공한 기존 CI를 확인했다.
기존 `manifest.py`는 `train/val/test`를 허용하지만 CLI 평가 명령은 없었다.
기존 추천기는 벡터가 있는 `train` 관측만 검색하고 질의 이미지와 동일한 SHA-256은
제외했다. 기존 CI는 합성 입력을 사용하는 코드·저장·전송 검증이며 실제 코퍼스
품질을 측정하지 않았다.

기존 Windows 수정은 SQLite spool 연결을 임시 디렉터리 정리 전에 닫는 변경이다.
그 코드와 기억 저장 포맷을 변경하지 않았다. 이 보고서 작성 시점의 새 변경에 대한
원격 CI 결과는 기존 성공 기록과 별개이며, 리뷰 PR의 checks에서 확인할 수 있다.

## 3. 변경 사항

| 파일 | 변경 |
| --- | --- |
| `vrs_int/evaluate.py` | 평가 프로토콜, 지표, 희소도, 저장 보존율, 사례, 재현 메타데이터 |
| `vrs_int/recommend.py` | 기존 CUDA 경로와 명시적인 NumPy 오프라인 경로가 후보 필터·순위·투표·출력 구조를 공유 |
| `vrs_int/__main__.py` | `evaluate` 명령, 고정 스냅샷, GPU 명시 지정, 출력 덮어쓰기·기억 상태 경로 방지 |
| `tests/test_vrs_int_evaluate.py` | 합성 특징·가상 태그로 12개 새 테스트 |
| `vrs_int/EVALUATION.md` | 영어 실행 가이드, 지표 분모, 검증 단계와 한계 |
| `vrs_int/README.md`, `README_KOR.md` | 평가 가이드 연결 |
| `.gitignore` | `evaluation*.json` 개인 결과 제외 |

실제 `recommend`는 여전히 CUDA가 필수다. NumPy로 자동 재시도하지 않는다.
`recommend_offline`은 저장된 특징에 대한 명시적인 계산 경로이며 이미지 인코더의
CPU 대체 구현이 아니다. 기존 CUDA 벡터 정규화, 유사도 계산, 이웃 선택과 투표식은
유지했다. 평가에서 사용할 제외 목록만 선택적 인자로 추가했다.

저장/갱신/활성화 구현, 메모리 포맷, revision 비교, parent 계보, tombstone,
불변 버킷, 패키지 전송 로직은 변경하지 않았다. 평가가 스냅샷 또는 CURRENT를
쓰지 않는 것을 바이트 단위 비교로 검증했다. 활성화된 스냅샷과 관계없이 사용자가
명시한 해시만 평가한다.

## 4. 누출 방지 프로토콜

1. 먼저 데이터의 `train/val/test`를 정하고 이미지를 인코딩한다. 동일 작품의 변형,
   리사이즈·크롭·근접 중복은 가능하면 동일 그룹에 넣고 같은 split에 배정한다.
2. 평가 시작 시 스냅샷 해시·버킷·관측·벡터·counts를 `Store.verify`로 확인한다.
3. 검색 후보는 벡터가 있는 `train`이며 지정한 rating에 맞아야 한다.
4. 스냅샷에 있는 **모든 val/test 관측**의 이미지 SHA-256과 canonical ID 및
   `source_ids`를 제외 목록으로 만든다. 평가 대상 split 밖의 홀드아웃, rating
   필터 밖의 홀드아웃, 벡터가 없는 홀드아웃도 이 차단 목록에 들어간다.
5. 제외 해시와 같거나 원본 ID가 겹치는 train 관측은 검색에서 제거한다. 질의
   자신의 해시도 공유 추천 필터에서 제외한다. 같은 train 이미지 해시는 한 번만
   투표한다. 희소도는 이 규칙을 적용한 train 이미지로만 계산한다.
6. holdout 태그는 추천 호출의 인자로 전달하지 않는다. 추천 후에만 해당 관측의
   정답·범주를 읽어 지표와 오류를 계산한다. audit 태그는 보존율 계산에만 쓴다.
7. 평가 대상 split에서 동일 이미지 해시가 여러 번 나오면 key 사전순 첫 유효
   벡터만 측정한다. 결측 벡터와 중복 질의는 `coverage.skipped`에 기록한다.

기본 추천에는 평가용 전체 holdout 차단을 강제로 켜지 않는다. 평가 호출만 더
엄격한 제외 목록을 사용하므로 기존 일반 추천 동작을 보존한다.

`val`에서 이웃 수·k·rating·희소도 기준을 고른 뒤 동일 설정을 고정하고 `test`를
최종 평가하는 절차를 권장한다. 도구가 반복적인 test 튜닝을 기술적으로 금지하지는
않는다. 원본 태그를 변경하면 평가 스냅샷이 달라지며 이전 결과는 이전 해시에 남는다.

## 5. 측정 가능한 지표

평가 정답은 해당 스냅샷의 **보존된 태그**다. 명시적인 작가·캐릭터·작품명 범주만
사용하며 이름 문자열로 범주를 추론하지 않는다.

질의 i의 정답 집합을 G_i, 상위 k 추천 집합을 T_i(k)라 하면:

- Precision@k = `|G_i ∩ T_i(k)| / k`
- Recall@k = `|G_i ∩ T_i(k)| / |G_i|`
- micro: 전체 hits 합을 전체 k 슬롯 또는 정답 개수 합으로 나눈다.
- macro: 질의별 Precision 또는 Recall의 산술평균이다.

추천이 k개보다 적어도 Precision 분모는 k다. 빈 정답에 대한 macro recall은
제외하며 0인 분모는 JSON `null`이다. NaN 또는 임의의 성공 점수로 대체하지 않는다.

| 출력 | 의미 |
| --- | --- |
| `top_k` | 태그 P/R@k, macro/micro, hits·truth·슬롯·질의수 등 원시 집계 |
| `category_top_k.artist/character/copyright` | 전체 top-k 안에서 범주별 정답 회수율; 추천/정답 양쪽의 명시 범주 요구 |
| `identity_list_at_3` | 기존 추천기가 따로 내주는 범주별 최대 3개 목록의 회수율 |
| `rare.top_k` | 적격 train 이미지 빈도 `1..rare_max`인 알려진 희소 태그의 회수율 |
| `unseen_top_k` | 적격 train 빈도가 0인 정답의 회수율; 알려진 희소 태그와 분리 |
| `rare` 보존 통계 | 검색 가능한 train 이미지의 원본 audit 태그 대비 보존된 희소 어휘/출현 |
| `storage_preservation` | 벡터 결측·차단·중복도 포함한 live train 관측 전체의 희소 태그 보존 |
| `errors` | 오추천/미추천 태그수, 각각 발생한 질의수, 추천 없음, 추천은 있지만 적중 0 |
| `cases`, `per_tag` | 명시 지정 시 사례·오류 태그·이웃 key·태그별 빈도/회수율 |

희소 태그 기준은 기본 `rare_max=5`이며 사용자가 고정할 수 있다. 보존 통계에서는
원본 audit에만 있고 필터링으로 제거된 태그도 제거로 집계한다. 예를 들어 `highres`
제거는 보존율을 낮추지만 정책에 따른 정상적인 제거일 수 있다. 낮은 보존율을
바로 데이터 손실 버그로 판정하지 않는다.

audit가 없거나 읽을 수 있는 원본 태그가 없으면 저장된 태그를 기준으로 계산하고
원본 audit 확인 관측수를 따로 기록한다. 일부 audit만 존재하는 경우도 불완전할 수
있으므로 audit coverage가 낮은 100% 보존율은 무손실 입증이 아니다.

저장 보존 빈도는 train **관측**을 세고, 추천 희소도는 차단·중복 제거 후 **이미지**를
센다. 서로 다른 분모다. `--include-cases`의 `per_tag` 보존 상세는 적격 검색 집합을
기준으로 한다. train에 없는 태그는 이웃 투표로 생성할 수 없으므로 unseen 회수율은
원칙적으로 0이며, 이를 알려진 희소 태그의 실패와 분리해서 해석한다.

작가/캐릭터/작품명·희소·unseen 집합에 대한 precision 필드도 k 슬롯을 분모로
사용한다. 이 부분집합의 회수 성능 판단에는 recall을 사용한다. 별도 identity
목록의 precision은 3 슬롯을 사용한다.

오추천은 **주어진 정답 태그와의 불일치**다. 이미지에 실제로 존재하지만 annotation에
빠진 태그가 있을 수 있으므로 의미적으로 틀렸다는 확정 판정이 아니다.

## 6. 실행 명령과 출력

저장된 실제 또는 합성 특징 벡터의 CPU 평가:

```bash
python -m vrs_int evaluate --store ./vrs-int-data/memory --snapshot EXACT_SNAPSHOT_SHA256 --split val --ks 5,10,20 --neighbors 32 --rare-max 5 --backend numpy --output ./vrs-int-data/evaluation-val.json
python -m vrs_int evaluate --store ./vrs-int-data/memory --snapshot EXACT_SNAPSHOT_SHA256 --split test --ks 5,10,20 --neighbors 32 --rare-max 5 --backend numpy --include-cases --output ./vrs-int-data/evaluation-test.json
```

`EXACT_SNAPSHOT_SHA256`은 실제 64자 스냅샷 해시로 바꾼다. `--ratings g,s`를 주면
질의와 근거 양쪽에 적용된다. CLI는 기본적으로 모든 rating을 사용한다.
`--include-cases`를 생략하면 상세 태그명·오류 목록을 포함하지 않는다.
결측/중복의 observation key 등 메타데이터는 요약에도 있으므로 결과를 개인 파일로
취급한다. 기존 결과를 덮어쓰지 않으며 `objects/`, `snapshots/`, `CURRENT`에 평가
출력을 쓰는 경로도 거부한다.

결과에는 snapshot·parent·encoder contract·설정·구현 파일 SHA-256·Python/NumPy/
플랫폼·backend·평가 key 집합 digest가 포함된다. 결과 자체에도 canonical JSON의
`report_sha256`이 붙으며 이 필드 자체는 해시 계산에서 제외한다. 시각 또는 latest
별칭을 넣지 않아 같은 스냅샷·환경·설정의 반복 실행 결과를 비교할 수 있다.
CPU BLAS와 CUDA의 부동소수점 연산 차이로 가까운 점수의 순서가 달라질 수 있으므로
플랫폼을 넘는 비트 단위 일치를 주장하지 않는다.

## 7. 실행한 테스트와 계산 예시

로컬 환경: Linux, Python 3.12.14, NumPy 2.3.5. PyTorch 미설치.

```bash
python -m unittest discover -s tests -p "test_vrs_int*.py" -v
python scripts/check_code_only.py
python -m vrs_int evaluate --help
git diff --check
```

오프라인 43개 테스트 통과. 공개물 검사와 diff 형식 검사도 최종 stage에서 통과했다.
기존 CI는 이 파일명 패턴을 이미 사용하므로 신규 평가 테스트도 자동으로 들어간다.
OS/Python matrix는 Ubuntu/Windows/macOS × Python 3.11/3.13이며, 로컬 Linux 3.12
성공과 원격 matrix 성공은 별개다.

신규 12개 테스트의 검증 내용:

1. 수기로 계산한 top-k P/R, 범주 회수율, 희소/미관측 태그, 보존율, 오류 목록.
2. val 선택, 명시 범주가 없을 때의 빈 분모 처리, 상세 출력 비활성화.
3. test 이미지와 해시가 같은 train, 다른 holdout의 원본 ID alias, 중복 train vote 제외.
4. holdout 정답만 바꿔도 추천은 같고 지표만 바뀌는지 확인; 이전 계보 데이터 보존.
5. 두 번 평가의 결과/해시 일치와 store 바이트 불변, 모델 import 차단 상태에서도 동작.
6. 결측 벡터·중복 질의·빈 train 검색의 실패/기권 대신 0 hits 및 coverage 처리.
7. 기존 `exp((similarity - best_similarity)/0.07)` 투표 계산, batch 경계와 동률 key 정렬.
8. 잘못된 split·k·rating·limit·희소도 기준, 빈 평가, 손상된 버킷의 실패 처리.
9. i8 저장 특징과 audit 없는 관측의 coverage/fallback 처리.
10. 정답 수가 다른 여러 질의의 macro/micro 차이, rating 필터, 벡터 없는 train 보존.
11. 기본 production 추천은 CUDA 필수, CPU 자동 fallback 금지, 잘못된 질의 벡터 거부.
12. CLI 실행, 출력 덮어쓰기·기억 내부 출력 경로 금지, CUDA 선택 시 GPU 인자 필수.

단일 합성 질의에서 정답은 5개, top-5 추천은 5개 중 4개 정답이다.
이웃 1개를 사용하는 완전히 알려진 특징으로 아래 결과를 확인했다.

| k | hits | Precision@k | Recall@k |
| --- | --- | --- | --- |
| 3 | 3 | 1.0 | 0.6 |
| 5 | 4 | 0.8 | 0.8 |
| 10 | 4 | 0.4 | 0.8 |

이 가상 예에서 top-5 범주별 정답 및 알려진 희소 정답 회수율은 1.0, 미관측 정답
회수율은 0.0이다. 원본 희소 어휘 6개 중 정책상 제거된 `highres` 1개를 제외한
5개가 보존되어 어휘/출현 보존율은 5/6이다. **이 숫자는 테스트 fixture의 계산
확인값이며 실제 코퍼스 품질이 아니다.**

## 8. 아직 검증하지 못한 항목과 실제 품질 측정 단계

| 항목 | 상태 / 필요한 추가 증거 |
| --- | --- |
| 평가 수식·누출 방지·합성 저장 특징 | 로컬 오프라인 테스트로 검증 |
| 기존 기억 계보·스냅샷·갱신·전송 | 기존 회귀 테스트 통과 |
| 실제 인코더의 이미지 특징 추출 | 미검증; 개인 이미지와 로컬 인코더·CUDA 필요 |
| CUDA 검색의 실행 및 NumPy와 순위 비교 | 미검증; CUDA PyTorch·명시 GPU 필요 |
| 개인 코퍼스 실제 P/R·희소/범주 회수율 | 미측정; 품질이 확인된 annotation 및 그룹 분리 코퍼스 필요 |
| 작가 식별 정확도 또는 확률 보정 | 구현·주장하지 않음 |
| SWEGCA A/R/U | 해당 코어 호출/판정 없음 |
| 수백만 이미지의 평가 시간·RAM 한도 | 미측정; host cache·추가 메타데이터 pass가 존재 |
| 근접 중복 및 upstream 인코더 사전학습 누출 | 정확 해시/원본 ID만으로 입증 불가 |

실제 품질 검증은 다음 단계로 진행한다.

1. 개인 환경에서 train/val/test 및 중복 그룹을 먼저 고정한다. annotation의 범주와
   희소 태그 누락 여부를 확인한다.
2. 기존 `check-input`·`build`를 실제 인코더와 명시 GPU로 실행한다. 이 build는 모든
   split의 특징을 만들지만 추천 근거는 train만 사용한다. migrate로 만든 train-only
   특징에는 holdout 벡터가 없을 수 있으므로 별도 인코딩 단계가 필요하다.
3. 스냅샷을 verify하고 val 평가로 설정을 고정한다. 인코더·코퍼스·이미지는 개인
   저장 위치에 두고 공개 저장소에 업로드하지 않는다.
4. 동일 스냅샷의 test를 NumPy와 CUDA로 각각 평가한다. CUDA 예시:

```bash
python -m vrs_int evaluate --store ./vrs-int-data/memory --snapshot EXACT_SNAPSHOT_SHA256 --split test --ks 5,10,20 --neighbors 32 --rare-max 5 --backend cuda --gpu GPU-YOUR-UUID --include-cases --output ./vrs-int-data/evaluation-test-cuda.json
```

5. dtype/드라이버/PyTorch/encoder provenance를 기록하고 가까운 점수의 순위 차이를
   확인한다. 범주별·희소·unseen·오류 사례와 원시 counts를 함께 검토한다.
   자동 통과 임계값은 정하지 않았다. 특정 코퍼스에서 측정된 값만 해당 범위의
   결과로 보고하며 실이미지/CUDA 측정 전에는 추천 품질 개선을 주장하지 않는다.

## 9. 공개 배포 범위

추가 파일은 Python 코드와 Markdown 문서뿐이다. 합성 이름·벡터는 테스트 코드에서
임시 디렉터리로 생성하며 코퍼스·모델·이미지·특징 파일을 저장소에 추가하지 않았다.
기존 공개물 검사 규칙을 완화하지 않았다. JSON 결과는 이름과 무관하게 공개 검사에서
거부되며 표준 `evaluation*.json` 결과명은 Git ignore에도 추가했다.
개인 정답·추천·사례가 들어 있는 결과는 로컬에서 보관해야 한다.
