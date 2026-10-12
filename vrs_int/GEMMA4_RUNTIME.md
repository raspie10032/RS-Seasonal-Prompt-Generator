# Gemma 4 TIPO checkpoint / Gemma 4 TIPO 실행 기준

This checkpoint keeps image understanding and every text layer. VRS-INT supplies
reference tags and Korean terminology as **text**; it does not replace neural
network weights or inject hidden states. Personal images, dictionaries, memory,
adapters and GGUF files remain outside the code distribution.

이번 기준은 **텍스트 층·어휘 유지 + 비전 FP16 중심 + PLE Q4_K**입니다.
VRS-INT는 참고 사례와 한국어 용어를 텍스트로 공급합니다. 내부 계산을 대신하는
기능은 실행 경로에 포함하지 않습니다. 기존 경험을 다시 구축할 필요도 없습니다.

## Choose the model / 모델 선택

| Choice / 선택 | Meaning / 의미 |
|---|---|
| Gemma TIPO automatic text model | Existing Korean TIPO v2 fine-tune / 기존 한국어 TIPO v2 |
| Gemma TIPO automatic vision model | Existing vision TIPO and its paired projector / 기존 비전 TIPO와 대응 mmproj |
| Local mixed-precision candidate | Gemma 4 E2B base derivative, text Q4_K_M with PLE Q4_K, vision F16/F32; **not a newly completed TIPO fine-tune** / 원본 계열의 혼합 정밀도 후보이며 새 TIPO 학습 완성본은 아님 |

The mixed-precision candidate is an explicit local selection. Automatic downloads
continue to select the established TIPO files. Do not pair an arbitrary text
checkpoint with a projector from a differently trained model.

혼합 정밀도 후보는 사용자가 직접 지정합니다. 자동 다운로드 기본 모델을 바꾸지
않습니다. 비전 모델과 mmproj는 같은 원본에서 준비한 대응 파일을 사용하세요.

## ComfyUI / 노드 사용

1. Put the language GGUF in `ComfyUI/models/gguf/`, then refresh the model list
   or restart ComfyUI when your current work allows it.
2. Select that filename in **Gemma TIPO → model**. Set `gpu_layers=-1` for GPU
   offload; saved workflows with `0` request CPU execution.
3. For image input, connect **Load Image → Gemma TIPO.image** and set
   `mmproj_path` to the absolute path of its corresponding vision projector.
4. Connect the text output to **Preview as Text** or the image prompt encoder.

텍스트만 넣으면 텍스트 확장, 이미지를 연결하면 비전 태그 생성입니다.
기존 Gemma TIPO 노드에서는 이미지가 우선이며 `prompt` 문장과 자동으로 결합하지
않습니다. 이미지 경험과 요청 조건의 결합은 아래 생성 상태 CLI에서 다룹니다.

Both text and vision use two CPU helper threads by default. The legacy vision
node can retry on CPU if GPU execution fails; inspect its log when verifying
the selected backend. Its process exits after each request. The legacy text
engine caches its model, so its VRAM can remain allocated until process exit.
The separate RS AI Prompt Writer uses a different model and lifecycle; it is
not the Gemma candidate described here.

텍스트·비전의 CPU 보조 스레드는 기본 2개입니다. 기존 비전 노드는 GPU 실패 시
CPU 재시도가 가능하므로 실제 GPU 사용 여부는 로그에서 확인해야 합니다.
비전은 호출별 프로세스 종료로 해제되지만, 기존 텍스트 TIPO는 모델을 캐시합니다.
다른 노드인 RS AI Prompt Writer의 모델·메모리 동작과 혼동하지 마세요.

## VRS and Korean input / 경험과 한국어 입력

The complete commands and input formats are maintained in these guides:

- [Personal memory](README.md): image/tag pairs, incremental updates and recommendation.
- [Generated images](GENERATED_IMAGES.md): read original generation metadata;
  optionally tag unreadable images with a separately prepared local tagger.
- [Korean text knowledge](TAG_DICTIONARY.md): Korean terms → English tag candidates,
  meaning and artist/character/copyright categories.
- [Generation state](GENERATION_STATE.md): export selected identities, keep/change/
  exclude conditions and compatible reference tags; compare TIPO inputs on GPU.
- [Tuning pilot](GEMMA4_TUNING.md): optional language-only LoRA experiment and its limits.

기본 흐름은 **이미지 경험 검색 → 사전으로 의미 해석 → 조건과 참고 재료 정리 →
Gemma로 확장**입니다. 사전 조회와 상태 내보내기는 모델 없이도 가능합니다.
상태를 내보냈다는 사실만으로 한국어 부정문이나 장면 변경을 모델이 정확하게
수행한다고 볼 수는 없습니다. 작가·캐릭터·작품 이름은 참고 후보와 사용자 지정값을
구분하고, 품질·연도 태그는 경험 추천에서 제외합니다.

## Inspect a local candidate / 혼합 정밀도 검사

These tools require Python 3.11+, NumPy and the `gguf` package from the matching
llama.cpp checkout. They operate on local files and do not download weights.
Use new output filenames; keep original weights unchanged.

```bash
python -m experiments.gemma4_gguf_audit inspect /path/to/text-Q4_K_M-PLE-Q4_K.gguf --report /path/to/new-text-report.json
python -m experiments.gemma4_gguf_audit vision-only /path/to/mmproj-vision-audio-f16.gguf /path/to/new-mmproj-vision-f16.gguf --report /path/to/new-vision-report.json
python -m experiments.gemma4_gguf_audit compare-ple /path/to/text-Q4_K_M-default.gguf /path/to/text-Q4_K_M-PLE-Q4_K.gguf --report /path/to/new-ple-report.json
```

`vision-only` removes audio tensors and audio metadata while checking that
every retained vision/projector tensor is byte-identical. `compare-ple` requires
PLE to be the only changed tensor, with unchanged dimensions and Q4_K precision.
It does not create the quantized text file or prove equal model capability.

## Recorded evidence / 기존 측정 범위

The private 2026-10-12 comparison used eight fixed text/image cases on a local
llama-server. The default-PLE candidate passed 8/8 explicit tag checks; the
PLE-Q4 candidate passed 7/8, with unwanted quality tags in one output. These are
keyword checks on a small sample, not prompt-quality or image-recognition scores.

The mixed pair occupied about **2.98 GiB on disk**, including approximately
**356 MiB** of vision/projector tensors and metadata. Sampled server VRAM was
**2,853 MiB for both candidates**, while peak process RSS was approximately
3,547 MiB versus 2,971 MiB. Disk/RAM savings therefore did not establish VRAM
savings in that run. Model weights, private cases and raw outputs are not shipped.

8개 고정 사례에서 PLE Q4는 7개 조건 검사를 통과했고, 1개에는 원하지 않은 품질
태그가 남았습니다. 파일 감소와 VRAM 감소는 별개입니다. 현재 조합은 다시 쓸 수
있는 실행 기준이며, 프롬프트 품질이나 한국어 지시 수행의 완성을 뜻하지 않습니다.
