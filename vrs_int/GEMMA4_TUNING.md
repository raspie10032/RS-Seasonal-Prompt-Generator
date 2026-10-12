# Gemma 4 TIPO tuning pilot / Gemma 4 TIPO 조정 실험

The goal is to write a new prompt from Korean instructions and VRS reference state,
while preserving selected names and attributes, changing the requested scene, and
omitting excluded elements. The first output format is comma-separated English tags.
English prose generation and direct numerical-state injection are separate tasks.

한국어 요청과 VRS 참고 상태를 받아 새 프롬프트를 쓰는 Gemma 4 조정 실험입니다.
사용자가 선택한 작가·캐릭터·작품·외형 보존, 장면 변경, 제외 조건 준수를 다룹니다.
이번 출력은 영문 태그이며 자연어 프롬프트 학습까지 완료했다는 뜻은 아닙니다.

## Model and evidence boundary / 모델과 검증 범위

- Use a local **Gemma 4 E2B Safetensors** checkpoint. A GGUF file is not accepted.
- The existing project's older training configuration uses
  [`p-e-w/gemma-4-E2B-it-heretic-ara`](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara),
  a derivative of `google/gemma-4-E2B-it`. Pin its revision when acquiring it.
- This creates a **new language-only LoRA** on that base. It does not continue the
  TIPO v2 GGUF's learned weights. Keep the base-model comparison separate from a
  comparison against the existing TIPO model.
- The base is loaded with NF4/double quantization and BF16 computation. Frozen
  per-layer embeddings remain BF16; only language-projection LoRA weights train.
- Vision/audio modules are removed from the training process after loading. Their
  original files and VRS search vectors are not changed. Images are not training inputs.
- Raw generations are scored **without a constraint-repair postprocessor**. Exact
  required/excluded tag matching and a small known-conflict table are diagnostics,
  not human preference, general translation accuracy, or generated-image quality.

기존 GGUF에 이어서 학습하는 것이 아니라 같은 Gemma 4 계보의 원본에서 시작합니다.
학습 전·후에는 동일한 원본 모델을 비교합니다. 이미지 인코더는 학습하지 않으므로
기존 VRS 경험을 재구축할 필요가 없습니다. 모델 교체·ComfyUI 배포는 자동 실행하지 않습니다.

## Environment / 환경

Use a separate Python environment with a CUDA-capable PyTorch build supporting
your selected GPU, Transformers with `Gemma4ForConditionalGeneration`, PEFT,
bitsandbytes and Accelerate. Do not replace a running ComfyUI environment to install
training dependencies. The pilot saves exact library versions in `run-config.json`.

The local test environment can reuse compatible installed packages through a local
`.pth` file; this shares their existing versions without installing into ComfyUI.
This is a local convenience, not an isolated reproducible dependency lock.

GPU UUIDs prevent device-index changes from redirecting a run. CUDA is mandatory,
with no CPU fallback. The default CPU thread count is two. The runner requires at
least 11 GiB free VRAM at startup; this is a preflight check, not a peak-memory guarantee.
Do not start a run on a GPU already needed by another active workload.

## Construct the pilot data / 시험 자료 만들기

Run from the repository root. Keep data, model weights and results outside the public
checkout. Output directories must not already exist.

```bash
python -m experiments.gemma4_tuning_data --output /private/gemma4-pilot/data
```

This generates 240 training and 116 evaluation examples. Six scene families train;
three different scene families evaluate. Korean accessory inclusion/exclusion pairs
also have disjoint held-out concepts. All records have source labels and required/
excluded tags; the manifest pins hashes. The three input modes are request, flat
reference tags and structured VRS-style state.

These are **constructed targets**, using fictional names and a small fixed vocabulary.
They are not image-caption truth, actual VRS retrievals, human-rated preferences or
a complete TIPO training corpus. The evaluation shares templates and much vocabulary
with training, so even a strong result establishes only narrow instruction adaptation.
Use real, reviewed requests and separate images/scene families for subsequent evaluation.

합성 정답은 규칙을 지킨 예시이며 실제 이미지 캡션이나 선호도 정답이 아닙니다.
평가는 장면별로 나뉘지만 입력 틀과 일부 어휘를 공유하므로 일반화 주장은 제한됩니다.
개인 이미지의 원래 프롬프트도 생성 의도이지 실제 시각 내용 전체의 정답은 아닙니다.

## Train and compare / 학습과 비교

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m experiments.train_gemma4_tipo \
  --base /private/models/gemma4-e2b-safetensors \
  --data /private/gemma4-pilot/data \
  --output /private/gemma4-pilot/run-001 \
  --gpu-uuid GPU-your-device-uuid \
  --steps 32 --grad-accum 2 --rank 8 --lr 0.0001 \
  --eval-count 20 --max-new-tokens 192
```

Optionally append `--states /private/barbara-state.json /private/bookshop-state.json`
to compare actual VRS states without adding them to training. Both a state object
and the earlier trial's `{"state": ...}` wrapper are supported. These outputs are
saved separately as `external-before.json` and `external-after.json`.

The runner hashes the original checkpoint, validates assistant-only loss masking,
rejects silent truncation, evaluates the original base, trains only the LoRA, evaluates
the adapter with the same greedy decoding, and checks a saved/reloaded adapter gives
the same first output. A generation cut off by the token limit cannot pass.

Default training sees 64 records across 32 optimizer steps: this is less than one
epoch. It is a bounded feasibility pilot, not a completed production model. The
evaluation subset rotates through scene/mode groups; all remaining held-out records
stay available for a larger assessment. Checkpoints contain adapter weights only;
optimizer/random states are not saved, so they are not exact-resume checkpoints.

Important outputs:

| File | Meaning |
|---|---|
| `progress.json` | Latest stage and completed step |
| `run-config.json` | Versions, settings, weight hashes, trainable modules and data manifest |
| `before.json`, `after.json` | Raw responses, exact checks and per-request times |
| `loss.json` | Loss and gradient norm by step |
| `adapter/` | LoRA weights/config and tokenizer; original base still required |
| `report.json` | Final before/after totals, training VRAM allocation and serialization check |

PyTorch peak allocation covers the training process's allocated CUDA tensors, not
total device VRAM. Other applications, display allocations and allocator reservations
need separate observation. No claim about image-generation VRAM follows from this run.

## Next data stage / 다음 학습 자료

After checking the pilot, construct reviewed examples from actual VRS states and the
Korean dictionary: correct-name copying, conflicting neighbor selection, scene-only
editing, negation, missing dictionary terms and varied useful expansions. Split on
source image/near-duplicate group, character and scene as appropriate before making
variations. Keep provenance and original captions in private storage. Preserve prompt
intent separately from machine-predicted visual tags.

다음에는 실제 한국어 요청과 개인 VRS 상태에서 검토한 정답을 늘립니다.
반복 생성물·유사 이미지가 평가에 섞이지 않도록 먼저 나누고, 작가·캐릭터·작품을
잘 복사하는지와 새로운 장면의 유용성은 별도로 평가합니다.

See [generation-state experiments](GENERATION_STATE.md) and
[text knowledge/dictionary](TAG_DICTIONARY.md).
