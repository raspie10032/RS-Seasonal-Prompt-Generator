# Local data and privacy / 로컬 자료와 개인정보

## 한국어

**개인 이미지·프롬프트·태그·VRS 경험은 본인 PC의 ComfyUI에서 처리합니다.**
이 확장에는 자료를 GitHub, Hugging Face, 개인/공용 클라우드 저장소로 업로드하거나
동기화하는 기능이 없습니다. VRS 구축·조회에는 외부 API나 계정이 필요하지 않습니다.

- **로컬의 기준은 ComfyUI 서버가 실행되는 컴퓨터입니다.** 브라우저만 본인 PC에서
  열고 원격/클라우드 ComfyUI를 사용하면 자료는 그 서버에서 처리됩니다. Comfy.ICU
  같은 외부 소개 페이지의 클라우드 실행 버튼은 이 확장의 로컬 사용 경로가 아닙니다.
- TIPO의 최초 모델·토크나이저·LoRA 준비는 **받기 전용 다운로드**입니다. 개인 입력을
  그 요청에 넣지 않습니다. 필요한 파일을 미리 준비하면 추론 중 다운로드가 필요하지
  않습니다. VRS 자동 비전 선택도 동일한 받기 전용 준비를 사용합니다.
  `RS_MODELS_OFFLINE=1` 또는 `HF_HUB_OFFLINE=1`이면 새 다운로드를 거부하고
  이미 받은 정상 파일만 사용합니다. 이 설정은 네이티브 Q4·LoRA·VRS 자동 비전에 적용됩니다.
  레거시 llama.cpp 실행기와 다른 확장 프로그램의 네트워크 접근까지 차단하지는 않습니다.
- VRS의 `export`/`install`은 지정한 **로컬 디렉터리 사이의 복사**입니다. 원격 저장소에
  게시하는 명령이 아닙니다. 백업에도 원본 태그·프롬프트·경로가 남으므로 개인 자료로
  취급하세요. 압축은 익명화가 아닙니다.
- 구축·갱신·로컬 백업·설치 시 데이터 디렉터리에 전체 내용을 제외하는 `.gitignore`를
  기록합니다. 폴더 수집 카탈로그와 검색 캐시에도 적용합니다. 기존 제외 규칙은 보존하고,
  원본 이미지 폴더는 변경하지 않습니다. 읽기만 하는 Store 로더는 파일을 수정하지 않습니다.
- 코드 배포 검사는 카탈로그·벡터·SQLite 임시 기록·이미지 등을 거부합니다. 레지스트리
  게시 작업도 패키지를 올리기 전에 이 검사를 통과해야 합니다.

이 보호는 일반적인 `git add`에 개인 자료가 섞이는 실수를 막습니다. **이미 Git에 추적된
파일, `git add -f`, 별도 백업/동기화 프로그램, 다른 커스텀 노드의 통신을 차단하는 기능은
아닙니다.** 개인 자료는 코드 저장소·클라우드 동기화 폴더 밖에 두세요. GitHub 검사는
push 이후 실행되므로 원격 전송을 취소하지 않습니다. push 전에도 아래 검사를 실행하고
스테이징 내용을 확인하세요. 기존 사용자 Git hook을 자동 설치하거나 교체하지 않습니다.

```bash
python scripts/check_code_only.py
git diff --cached --stat
```

## English

**Process personal images, prompts, tags and VRS memory in ComfyUI on your own PC.**
This extension has no uploader or synchronizer for GitHub, Hugging Face, or private/public
cloud storage. VRS building and retrieval need no external API or account.

- **Local means the computer running the ComfyUI server.** Opening a remote/cloud ComfyUI
  in a local browser still processes data on that server. Cloud-run buttons on third-party
  catalogue pages such as Comfy.ICU are not this extension's local execution path.
- First-use TIPO model, tokenizer and LoRA preparation makes **download-only requests**;
  personal inputs are not included. Prepare all selected files in advance to avoid downloads
  during inference. VRS automatic vision uses the same download-only preparation.
  `RS_MODELS_OFFLINE=1` or `HF_HUB_OFFLINE=1` rejects missing downloads and reuses valid
  cached files for native Q4, LoRA and VRS automatic vision. This is not a network sandbox
  for legacy llama.cpp runtimes or other extensions.
- VRS `export`/`install` copy between specified **local directories**; they do not publish
  to a remote repository. Backups retain tags, prompts and paths. Compression is not anonymization.
- Build, update, local export and install protect output directories with a `.gitignore`
  excluding everything inside. Folder catalogues and persistent search caches get the same
  protection. Existing ignore rules are preserved; original image folders are untouched.
  Read-only Store loading does not rewrite any files.
- The code-only publication check rejects catalogues, vectors, SQLite journals and images.
  Registry publication now runs this check before uploading the package.

These guards prevent accidental inclusion by ordinary `git add`. They **cannot block already
tracked files, `git add -f`, independent backup/sync software, or networking in other custom
nodes**. Keep personal material outside source repositories and cloud-synced folders. GitHub
checks run after a push and cannot undo that transfer: run the commands above and review the
index before pushing. Existing Git hooks are not installed or replaced automatically.
