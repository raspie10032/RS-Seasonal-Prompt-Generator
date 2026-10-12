# RS-Seasonal-Prompt-Generator

*English | [한국어](README_KOR.md)*

 Generates season-specific fashion prompts by randomly combining fashion items, background settings, weather, time, and additional situational details from CSV data.

> **v2.9.0** adds RS AI Prompt Writer for Korean/English descriptions.
> It also fixes seasonal seed reproducibility, independent clothing
> toggles, CSV categories, and text/vision tag cleanup. Vision uses the paired
> trained model + mmproj and supports CUDA / Vulkan / Metal with CPU fallback.
> **`gpu_layers=0` stays CPU; `-1` enables GPU auto/all layers.** Model and
> runtime downloads are published only after a complete transfer/install.
> See [the changelog](Update_log_ENG.txt).

This pack provides **three independent nodes** (category: `prompt`):
1) Seasonal Fashion Prompt Generator (CSV, zero-dep), 2) Gemma TIPO
(Prompt / Image → Tags), 3) RS AI Prompt Writer (Korean / English → Prompt). Install by cloning/copying this folder into
`ComfyUI/custom_nodes/` and restarting ComfyUI. Each outputs a `STRING`;
they're chainable.

## AI prompt writing

[VRS-INT](vrs_int/README.md) is an experimental standalone memory builder and
image-reference tag recommender. Distribution is code-only: users select their
own Danbooru images and tags, acquire the encoder separately, and build/update
personal collections in batches. Corpus data and weights are not bundled.
Artist candidates are creative references, not verified image attribution.
See the [image/text pair examples (Korean)](examples/vrs-int/README.md) for JSONL
templates, a local sidecar-to-manifest script, model-free input checks, and batch
build/update/recommend commands. Optional captions are preserved as metadata;
recommendations currently use the paired tag arrays, not caption embeddings.

A separate [VRS/Danbooru GPU experiment](experiments/README.md) records local
tag associations and evidence IDs. The [full dataset build](experiments/FULL_DATASET.md)
and [128 held-out image evaluation](experiments/IMAGE_RETRIEVAL.md) are documented
separately. The image test found a historical pixel-casting defect in the feature
cache and includes a matched repair diagnostic. The separate
[image-inspired prompt experiment](experiments/IMAGE_PROMPT.md) connects the corrected
index, VRS reference tags, seasonal CSV scenes, and the Writer to create new prompts.
It targets creative generation rather than exact tag reconstruction, and is not yet
integrated with the ComfyUI nodes. The experiment guides and measured limits are in Korean.

For descriptions such as “a woman in a white shirt and blue skirt standing in a
park,” use **RS AI Prompt Writer (Korean / English → Prompt)**. It uses an
instruction-tuned model to preserve the requested scene while writing a coherent
prompt. The older TIPO model remains available for tag expansion and images;
it can omit or change requested details, even with stronger instructions.

1. Add **RS AI Prompt Writer** from the `prompt` category.
2. Enter a Korean/English description in `prompt`, or connect Seasonal Fashion
   Prompt Generator's `STRING` output to this input.
3. Keep `model` at automatic. First use downloads approximately **2.50 GB** of
   Qwen3-4B-Instruct-2507 Q4_K_M from Unsloth. The revision is pinned and its
   SHA-256 is checked. It runs locally without an API key or a paid API call.
4. Choose `tags` for a comma-separated English prompt, or `natural_language`
   for one English paragraph. The output connects to **Preview as Text** or the
   positive prompt input of your image model's text encoder.
5. Start with `creativity=balanced`, `length=medium`, `temperature=0.6`.

| Setting | Behavior |
| --- | --- |
| `faithful` | Translate/reorganize supplied details; instruct the model not to invent details. |
| `balanced` | Preserve specified details and add compatible lighting, atmosphere and composition. |
| `creative` | Elaborate unspecified details while retaining explicit requirements. |
| `length` | Short / medium / long target; detail preservation has priority. |
| `seed` | Repeatable sampling on the same model/backend; hardware/builds can differ. |
| `gpu_layers` | `-1` (default): all GPU layers, requiring a compatible llama-cpp-python GPU build. `0`: explicitly select CPU. No automatic CPU retry. |
| `cpu_threads` | `2` (default): CPU helper threads for both generation and prompt processing. Keep low to preserve desktop responsiveness. |

These are model instructions, not a guarantee of semantic correctness. Review
important requirements in the output. Additional GPU checks found that balanced
mode can still add unspecified weather (for example, rain when an umbrella was
mentioned), and prose can contain fragments or minor grammatical errors.
This node accepts text; image input remains
in Gemma TIPO. Custom models belong in `ComfyUI/models/gguf/`; use an instruction-
tuned language model with a chat template. Importing the nodes never downloads
models. Generation errors stop the writer node with a visible error instead of
silently returning the input. If an installed CUDA wheel cannot load its runtime,
use a matching GPU runtime in the **same Python environment as ComfyUI**.
`RS_TIPO_FORCE_CPU=1` also applies to the writer, but cannot repair a broken wheel.
CPU-only builds are rejected before downloading the writer model when GPU is
requested. Select `gpu_layers=0` explicitly if CPU execution is intended.
With multiple GPUs, set `CUDA_VISIBLE_DEVICES` to the intended GPU's UUID before
starting the Python process. This applies to the entire ComfyUI process.

Default model provenance: [Qwen original model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[Unsloth GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF), Apache-2.0;
revision `a06e946bb6b655725eafa393f4a9745d460374c9`, file
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, SHA-256
`3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`.

## Showcase

![Gemma TIPO node in ComfyUI](assets/gemma-tipo-example.png)

Korean natural language in → clean English Danbooru tags out:

> **In:** `소녀가 하얀 세일러복과 스카프를 하고 있음.`
>
> **Out:** `white sailor collar, white skirt, black hairband, blue eyes,
> blush stickers, long sleeves, looking at viewer, open mouth, red scarf,
> school uniform, serafuku, sitting, solo focus, sparkle sticker, striped
> clothes, sweatdrop, collared shirt, white background`

And **vision mode** — connect an image, get a plausibly-expanded
Danbooru tag set (TIPO-style; not exact labeling):

![Gemma TIPO vision mode in ComfyUI](assets/gemma-tipo-vision-example.png)

## Node 1 — Seasonal Fashion Prompt Generator

The original generator. Pick a `season` and toggle the fashion / background /
weather / pose / situation options; it returns a comma-separated prompt
randomly combined from the bundled CSV data. **Zero dependencies** — nothing
is installed and nothing runs a model.

The same inputs and `seed` reproduce the same CSV prompt, including `seed=0`.
Each request has its own random generator. Turn ComfyUI's seed behavior to
**fixed** when reproducing results; changing a CSV or this generator's version
can change the selected items. Top and bottom toggles also work independently;
an enabled one-piece is an alternative to separates, never combined with them.
Additional situation CSV conditions use `|` for alternatives and `&` for
requirements within an alternative (e.g. `misty&forest`).

## Node 2 — Gemma TIPO (Prompt / Image → Tags)

One unified node. **Connect an `image` → vision mode** (the image is
captioned via Gemma-4 vision = llama.cpp `mtmd` with the auto-downloaded
**co-trained vision model + mmproj**; set `gpu_layers=-1` to use an available GPU).
**No image → text mode** (the `prompt`, Korean/English NL or tags, is
expanded in-process via `llama-cpp-python`). Either way the output runs
through the same `kgen.formatter` post-processing (sort / dedup /
drop-redundant / count-conflict).

> ⚠️ **Vision mode is TIPO-style, not a strict 1:1 labeler.** With the
> co-trained vision model + mmproj it draws the image's main content from
> the picture, but it then **plausibly expands** that into a fuller
> Danbooru tag set (TIPO creative upsampling): expect some added tags
> that aren't literally in the image, and run-to-run variation with the
> `seed`. Great for turning a reference image into varied text-to-image
> prompts — for exact, faithful labeling use a dedicated tagger instead.

Text mode accepts **Korean or English natural language**, not just tags:

| Input | Output (example) |
| --- | --- |
| `spring, white blouse, pleated skirt, park` | `spring, white blouse, pleated skirt, park, outdoors, school uniform, long sleeves, sitting, solo, …` |
| `빨간 원피스, 밀짚모자, 해변` (Korean) | `straw hat, red one-piece swimsuit, sun hat, beach, ocean, swimsuit, outdoors, …` |

Inputs:
- `prompt`: text to convert (Korean/English NL or tags).
- `model`: a dropdown of the `*.gguf` files found in
  **`ComfyUI/models/gguf/`**. Pick `(auto: download default)` to fetch the
  default model (`gemma4-tipo-ko-v2-Q4_K_M.gguf` from
  [`raspie/gemma4-tipo-ko-gguf`](https://huggingface.co/raspie/gemma4-tipo-ko-gguf))
  into that folder on first use — a **progress bar shows on the node** while
  it downloads (reused afterward, never re-downloaded); or drop your own
  `.gguf` into `ComfyUI/models/gguf/` and select it (reload the node list to
  see new files).
- `tag_length`: target verbosity (`very_short` … `very_long`).
- `sort`: prompt-ordering preset applied by `kgen.formatter` (tags are
  categorized special/characters/copyrights/artist/general/quality/meta/rating):
  - `danbooru` (default): artist, rating, special, characters, copyrights,
    general, quality, meta.
  - `quality_first`: quality tags lead.
  - `artist_first`: artist/character lead.
  - `general_only`: keep only special + characters + general tags.
  - `simple`: only special, rating, general.
- `temperature`: sampling temperature.
- `ban_tags`: comma-separated tags to strip from the result.
- `seed`: changes the expansion for variation.
- *(optional)* `image`: connect a ComfyUI IMAGE to switch to **vision
  mode** (image → tags). **Fully automatic** — first vision use
  auto-downloads the prebuilt llama.cpp mtmd binary for your OS/arch and
  GPU (Windows/NVIDIA → CUDA + cudart; Linux/NVIDIA and AMD/Intel →
  Vulkan; macOS → Metal; otherwise CPU — no build) **and** the trained
  vision model + mmproj. Nothing to install or configure by hand.
- *(optional, vision only)* `mmproj_path`: **leave empty** — the trained
  vision mmproj is auto-downloaded (paired with the vision model). Only
  set this to point at your own mmproj.
- *(optional, both modes)* `gpu_layers`: **`0` = CPU** (default and saved
  workflow compatibility), **`-1` = GPU auto/all layers**, positive values =
  layer count. Text mode requires a GPU-enabled `llama-cpp-python` build.
  Vision retries on CPU if GPU initialization or inference fails.
  `RS_TIPO_FORCE_CPU=1` forces CPU in both modes; `LLAMA_MTMD_CLI` selects a
  local binary. CPU mode also disables vision-projector offload.

Notes:
- Longer / more detailed descriptions yield more and more accurate tags.
- Korean descriptions are not mixed into successful tag output. Existing
  comma-separated English input tags are preserved before cleanup.
- Ban matching ignores case and space/underscore differences. Ban filtering
  and deduplication still work if the optional category formatter is unavailable.
- Redundant general tags are dropped to save tokens: if a more specific tag
  with the same head noun exists (e.g. `white skirt`), the bare `skirt` is
  removed. Distinct tags like `dress`/`dress shirt` are kept.
- Conflicting person-count tags are collapsed per group: with `1girl`
  present, `4girls` / `multiple girls` are removed (`1girl` + `1boy` is
  kept — groups are independent).
- On model/dependency/inference failure, text returns the original prompt;
  vision returns an empty string. Check the `[TIPO]` / `[TIPO-V]` console log
  to distinguish fallback from generated output. Vision accepts one image;
  select one batch item before connecting a batch.
- Downloads use temporary files and reject empty/truncated transfers. This
  checks transfer completeness, not cryptographic model integrity.

### Supported environment & requirements

| | |
| --- | --- |
| **OS** | Windows / Linux / macOS — anywhere ComfyUI runs and a `llama-cpp-python` wheel exists (x86-64, Apple Silicon/arm64) |
| **Node 1 (Seasonal)** | Any ComfyUI install. Zero dependencies, no model. |
| **Node 2 (Gemma TIPO)** | Vision: optional GPU auto-selection (`gpu_layers=-1`, CUDA / Vulkan / Metal) with CPU fallback. Text: CPU by default (GPU via a CUDA `llama-cpp-python` build). |
| **RAM** | min ~8 GB (≈5 GB used for the 4-bit model) · 16 GB+ recommended |
| **Disk** | text ~4 GB (3.2 GB GGUF + deps); vision adds ~3.2 GB vision GGUF + ~1 GB mmproj + the GPU prebuilt (fetched once on first vision use) |
| **Speed** | **Vision** can offload to GPU with `gpu_layers=-1`; speed depends on hardware. **Text** runs in-process via `llama-cpp-python` (CPU wheel by default, so slow; install a GPU build for speed). |
| **Python** | ComfyUI's embedded env; `llama-cpp-python >= 0.3.23` (auto-installed/upgraded, one restart prompt) |

### Install (Node 2 only)

**No manual install is required.** The first time you actually run Node 2,
its dependencies are auto-installed into the active environment as needed:
`tipo-kgen` for sorting; **text mode only** also installs `llama-cpp-python`
**>= 0.3.23** from prebuilt wheels. Automatic installation never starts a
llama-cpp-python source build; if no matching wheel exists, the original
prompt is returned with an installation message. Older builds cannot
load the gemma4 architecture, so an existing too-old `llama-cpp-python` is
upgraded automatically — if that happens mid-session ComfyUI will ask you to
**restart once** (a C-extension can't be hot-reloaded). Node 1 never installs
anything.

To pre-install manually instead (optional):

```
pip install -r requirements.txt
```

If the prebuilt `llama-cpp-python` wheel install can't find a wheel for your
platform/Python, install it yourself once:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

## Examples and validation

Four [API request examples](examples/) use ComfyUI's built-in **Preview as Text**
(`PreviewAny`) output node:

- `examples/korean-ai-writer.api.json`: Korean description → AI writer → text preview.
- `examples/seasonal-to-writer.api.json`: seasonal CSV output → AI writer → English prose.
- `examples/seasonal-prompt.api.json`: seasonal CSV prompt, no model needed.
- `examples/seasonal-to-tipo.api.json`: seasonal output feeds Gemma TIPO text
  mode. First execution may install dependencies and download the model.

These are `/prompt` request bodies, not canvas workflow JSON. With this pack
installed in a running ComfyUI, submit from the repository directory:

```bash
curl -H 'Content-Type: application/json' --data-binary @examples/seasonal-prompt.api.json http://127.0.0.1:8188/prompt
```

For vision, connect **Load Image → Gemma TIPO.image → Preview as Text** and
leave `model` at automatic and `mmproj_path` empty. Keep `gpu_layers=0` for
CPU or choose `-1` to enable GPU detection.

Run the offline regression suite from the repository directory:

```bash
python -m unittest discover -s tests -v
```

NumPy + Pillow enable the PNG conversion test; it is skipped if they are absent.
The suite covers all 512 season/clothing-toggle combinations, reproducibility,
node discovery, tag processing, download interruption and mocked GPU fallback.
It neither downloads models nor proves real inference quality, speed, or driver
compatibility. CI is configured for Linux, Windows and macOS with Python 3.9
and 3.13 (the macOS 3.9 job uses an Intel runner).

For an **opt-in real inference check**, use the Python environment intended for
ComfyUI. First use may install the documented dependencies and download several
GB of models. The writer check defaults to GPU and two CPU helper threads;
the legacy TIPO checks default to CPU. They return a failing exit code on
fallback or empty output and can save a local JSON report:

```bash
python tests/smoke_writer.py --report models/writer-smoke.json
python tests/smoke_inference.py --mode text --report models/text-smoke.json
python tests/smoke_inference.py --mode vision --image /path/to/image.png --report models/vision-smoke.json
```

The writer script checks six fixed inputs in both output formats against explicit
scene-detail keywords. Read the outputs too: keyword checks do not establish
semantic correctness. Each legacy smoke command checks one input; these are not
general quality benchmarks. The writer report records the backend, requested
thread count and its own NVIDIA process allocation when available. Set
`RS_TIPO_VERBOSE=1` to capture actual layer-offload messages. Legacy vision
`--gpu-layers -1` permits GPU detection and CPU fallback, so its passing report
alone does not establish GPU use. Offline CI never runs these scripts.

## References & Credits

| Project | Use | License |
| --- | --- | --- |
| [KGen / TIPO](https://github.com/KohakuBlueleaf/KGen) by KohakuBlueleaf | `kgen.formatter` for tag categorization & prompt sorting | Apache-2.0 |
| [z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension) by KohakuBlueleaf | Reference implementation of the TIPO node flow; training-data lineage | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference engine; prebuilt `llama-mtmd-cli` (CUDA / Vulkan / Metal / CPU) for vision mode | MIT |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | Python bindings for in-process GGUF inference (text mode) | MIT |
| [Gemma](https://ai.google.dev/gemma) by Google ([`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)) | Base model family the TIPO GGUF derives from | Gemma Terms of Use |
| [`p-e-w/gemma-4-E2B-it-heretic-ara`](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara) | Decensored gemma-4-E2B the TIPO GGUFs were fine-tuned from (used so over-censoring doesn't corrupt explicit Danbooru tag output) | Gemma Terms of Use |
| [`unsloth/gemma-4-E2B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) by Unsloth | Base gemma-4-E2B `mmproj` (vision-tower lineage; optional fallback — no longer the default) | Gemma Terms of Use |
| [`raspie/gemma4-tipo-ko-gguf`](https://huggingface.co/raspie/gemma4-tipo-ko-gguf) | Default Korean→tags TIPO text model **and** the co-trained vision model + mmproj auto-downloaded by Node 2 | Gemma Terms of Use |

The TIPO expansion concept ("upsample"/expand short prompts into detailed
Danbooru tag sets) originates from KohakuBlueleaf's KGen/TIPO and the
[z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension); the
default model's fine-tune training data follows that same lineage.
