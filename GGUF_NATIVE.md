# Gemma Q4 GGUF in ComfyUI PyTorch / 네이티브 Q4

**2026-10-16: [Image 30K 배포 / release guide](TIPO_IMAGE_30K.md)** —
`tipo_lora=image-30k`; 두 출력은 `tipo_lora_scope=tags_only` 권장.
High compute requirements; outputs may disappoint or contradict the input.
See the linked guide for hardware, setup, measured results and known failures.

[한국어 README](README_KOR.md) · [English README](README.md) · [두 출력 / Dual output](PROMPT_EXPANSION.md)

## 한국어

`Gemma TIPO Native`와 `Gemma Prompt Expander Native` 안에서 기존 Heretic ARA
계열의 Q4 모델을 받거나 로컬 파일을 재사용합니다. **별도 로더 노드와 llama.cpp는
필요 없습니다.** GGUF는 저장 형식이며 실행은 ComfyUI의 PyTorch와 모델 관리로 합니다.
기존 `Load CLIP` 연결은 `model_source=connected_clip`(기본값)으로 유지됩니다.

### 설치와 첫 실행

1. 이 저장소를 업데이트합니다. ComfyUI를 실행하는 Python으로 설치합니다.
   ```bash
   python -m pip install -r requirements.txt
   ```
   Windows portable에서는 ComfyUI 상위 폴더에서 실행합니다.
   ```powershell
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\RS-Seasonal-Prompt-Generator\requirements.txt
   ```
   Q4 경로는 Python 3.10 이상이 필요합니다. 추가 패키지는 순수 Python GGUF 파서
   `gguf>=0.19.0,<0.20`입니다. torch를 교체하거나
   CUDA 휠·llama.cpp·Transformers·bitsandbytes를 설치하지 않습니다.
2. 큐를 마친 뒤 ComfyUI를 재시작합니다.
3. `Gemma Prompt Expander Native` 또는 `Gemma TIPO Native`를 추가합니다.
4. `clip` 연결을 해제하고 **`model_source=heretic_ara_q4`**를 선택합니다.
   `gguf_path`, `mmproj_path`, `tokenizer_path`는 비워둡니다.
5. 요청을 쓰고 실행합니다. 최초 실행 때 아래 파일을 다운로드하고 SHA-256을 검사합니다.
   이미지 입력을 연결할 때만 비전 파일을 추가로 받습니다.

| 파일 | 다운로드 크기 | 언제 필요한가 |
| --- | ---: | --- |
| Heretic ARA Q4_K_M 텍스트 GGUF | 약 3.19GiB | 항상 |
| 대응 tokenizer.json | 약 30.7MiB | 항상 |
| 원본 FP16 mmproj | 약 0.92GiB | 이미지 입력 시 |
| pilot-001 / pilot-002 LoRA | 후보당 약 45.3MiB | `tipo_lora`에서 선택 시 |

텍스트만 약 **3.22GiB**, 이미지 포함 약 **4.14GiB**입니다. 파일 크기이며 VRAM 상한이
아닙니다. BF16 전체 모델 9.57GiB를 추가로 받을 필요는 없습니다. 기본 `tipo_lora=none`은
학습 후보를 적용하지 않습니다. 두 LoRA는 태그 전용 실험 후보이며 창작 품질 개선을
보장하지 않습니다. VRS 상태와 사전 입력은 기존 두 출력 노드와 같은 방식으로 연결합니다.

저장 위치는 `ComfyUI/models/text_encoders/rs_gemma_q4/`입니다. 선택한 모델의 최초 실행은
인터넷과 해당 용량이 필요합니다. 손상된 기존 파일은 덮어쓰지 않고 오류를 냅니다.
중단된 전송의 임시 파일은 제거하므로 다음 실행에서 다시 받습니다. 실행 중 pip 설치는 없습니다.

### 로컬 파일과 오프라인

`gguf_path`에 **Gemma 4 E2B Heretic ARA의 텍스트 GGUF**, `tokenizer_path`에 같은
모델의 `tokenizer.json`을 지정합니다. 이미지가 있으면 `mmproj_path`도 지정합니다.
세 경로를 모두 지정하면 모델 다운로드가 없습니다. 이미지가 없으면 mmproj는 읽지 않습니다.
빈 경로는 해당 공개 파일을 다운로드하므로 완전 오프라인에서는 필요한 경로를 모두 채우세요.
이름만 바꾼 safetensors, 다른 크기의 Gemma, 다른 모델용 mmproj는 지원하지 않습니다.

공개 기본 파일은 아래 출처·revision으로 고정되어 있습니다. 전체 SHA-256과 크기는
[`tipo_gguf.py`](tipo_gguf.py)에 있습니다.

| 대상 | 출처 | revision |
| --- | --- | --- |
| 텍스트 Q4 | [mradermacher/gemma-4-E2B-it-heretic-ara-GGUF](https://huggingface.co/mradermacher/gemma-4-E2B-it-heretic-ara-GGUF) | `c02d14dccce74f69566f8b815e648cc9eba243f7` |
| 토크나이저 | [p-e-w/gemma-4-E2B-it-heretic-ara](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara) | `c9a1d4c031981f14d86eeb0c7d87de7fafd34513` |
| FP16 비전 | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | `0314792d7f1f7e229411f620751375812bb9faf2` |

Heretic ARA는 Google 원본의 파생 모델입니다. 기본 비전은 원본 계열 FP16 가중치이며,
별도의 비전 튜닝은 아닙니다. 공개 mmproj에 포함된 오디오 가중치는 실행 모델에 적재하지
않습니다. 다른 비전 프로젝터를 학습한 모델이라면 해당 모델의 대응 파일이 필요합니다.

### 메모리와 제약

- 언어 가중치는 압축된 GGUF 텐서로 보관하고 각 연산 때 GPU에서 복원합니다. 행 단위로
  조회하는 PLE는 필요한 토큰 행만 복원합니다. 전체 PLE를 FP16/FP32로 펼치지 않습니다.
- 비전 가중치는 FP16으로 보관합니다. 기본 ComfyUI Gemma와 같은 FP32 계산 경로를
  쓰므로 FP16 저장과 FP16 전 연산은 다른 의미입니다.
- 출력 임베딩은 작은 묶음으로 복원합니다. 압축 가중치 외에 임시 계산 버퍼·KV 캐시·
  LoRA·이미지 처리 메모리가 필요합니다. ComfyUI가 모델의 적재·해제를 관리합니다.
- 현재 Gemma 4 E2B(35층, hidden 1536) 전용입니다. NVIDIA CUDA에서 검증하며 CPU나
  다른 GPU 제품군의 성능·호환성을 보장하지 않습니다. 지원하지 않는 양자화는 오류로
  알리고 CPU NumPy 계산으로 조용히 전환하지 않습니다.
- Q4는 손실 양자화입니다. BF16과 출력이 같거나 더 좋은 품질이라는 뜻이 아닙니다.
  가중치를 계속 복원하므로 최적화된 llama.cpp 커널과 속도도 다를 수 있습니다.
- `clip` 연결과 내부 Q4를 동시에 선택하면 오류를 냅니다. `connected_clip`을 쓰거나
  `clip`을 해제하세요. 이미 LoRA를 병합한 모델에는 같은 LoRA를 다시 적용하지 마세요.

[로더 없는 Q4 API 예제](examples/gemma-q4-native.api.json) ·
[연산부 출처와 라이선스](vendor/gguf_torch/NOTICE.md)

## English

Select **`model_source=heretic_ara_q4`** inside either native Gemma prompt node and
disconnect `clip`. Keep the three file paths empty for pinned, hash-checked first-use
downloads. No separate loader node or llama.cpp runtime is required. Existing workflows
continue with the default `connected_clip` and their original `Load CLIP` connection.

Q4 requires Python 3.10 or later. Install `requirements.txt` with **ComfyUI's own Python**, then restart after the queue
finishes. The added dependency is the pure Python `gguf` parser, not a CUDA wheel or a
replacement torch. The Windows portable command is shown above. No package installation
happens during generation. ComfyUI must already support Gemma 4 and `CLIP.generate`.

Text downloads total approximately **3.22GiB**, including the matching tokenizer.
An image adds the **0.92GiB FP16 mmproj**, for approximately **4.14GiB** total. Each
optional TIPO pilot adapter adds about 45.3MiB. These are disk/download sizes, not VRAM
limits; the full 9.57GiB native BF16 base is unnecessary for this path. The original
connected safetensors path remains supported. Keep `tipo_lora=none` for the base model.

Files live in `models/text_encoders/rs_gemma_q4`. For offline use, supply matching
`gguf_path`, `tokenizer_path`, and, only when using an image, `mmproj_path`. An empty
required path requests its pinned public asset. Existing invalid files are rejected
without overwrite; interrupted transfers are discarded and restarted on the next run.
The exact upstream revisions are listed above; hashes are in `tipo_gguf.py`.

Packed language weights stay compressed between operations. PLE gathers only requested
rows, and the tied output projection uses bounded chunks. Vision is stored in FP16;
computation follows ComfyUI Gemma's FP32 path. LoRA patches apply after dequantization
and use ComfyUI's clone/load/unload lifecycle. Audio tensors in the public mmproj are
not loaded. VRS and tag-dictionary inputs are unchanged.

This implementation targets Gemma 4 E2B and is validated on NVIDIA CUDA. Unsupported
architectures/quantization types fail explicitly. It does not silently fall back to
NumPy inference. Q4 is lossy; identical BF16 outputs, quality improvement, arbitrary
GGUF compatibility, or llama.cpp-level speed are not promised. Use a matching projector
for any separately vision-tuned model. The public base is a Heretic ARA derivative,
not an unchanged Google checkpoint. The two pilot LoRAs remain experimental.

## 검증 범위 / Validation scope (2026-10-15)

ComfyUI `15eb748b3ec5f8a0a2d470b7fb280e2d7579f916`, Linux, RTX 4070 Ti SUPER에서
별도 서버의 실제 `/prompt` 실행으로 다음을 확인했습니다.

- 빈 경로에서 공개 텍스트 Q4·토크나이저 다운로드 후 자연어·태그 두 출력.
- 이미지 연결 시 FP16 비전 추가 다운로드 후 두 출력과 태그 전용 TIPO 실행.
- 같은 ComfyUI 실행 환경의 직접 호출에서 LoRA `none → pilot-001 → pilot-002 → none`
  전환과 마지막 원본 출력의 일치.
- 공개 Q4 직접 호출 검사에서 PyTorch 최대 할당량 약 **3.78GiB**.
  짧은 생성·단일 이미지 기준이며 전체 GPU 사용량이나 메모리 상한은 아닙니다.

실행 경로에 대한 검증입니다. 화면 캔버스 조작이나 Windows 실행, BF16 대비 품질·속도
비교는 수행하지 않았습니다. 머리색을 구체적으로 기술하지 않는 출력과 비표준·모순 태그가
남아 있어 이미지 이해나 태그 정확도의 개선을 입증하지 않습니다.

Validated through a separate ComfyUI server's real `/prompt` execution on the commit
and hardware above: empty-path text/tokenizer downloads, two output formats, image-triggered
vision download, and the tag-only TIPO node. Direct calls in the same ComfyUI environment
also verified both LoRA switches and restoration of the original output. Peak PyTorch
allocation was about **3.78GiB** for the public-Q4 direct-call probe, with short generation
and one image; this is neither total device VRAM nor an upper bound. Canvas interaction,
Windows execution, and BF16 quality/speed comparisons were not tested. Image details and
tag correctness still have known limitations.
