# RS-Seasonal-Prompt-Generator

English | [한국어](README_KOR.md)

Create image-generation prompt ideas from seasonal CSVs, descriptions, or your
own reference images and tags. This repository contains three **ComfyUI nodes**
and the standalone **VRS-INT** command-line tool.

**VRS-INT ships code, documentation, and fictional examples only.** No starter
memory, collected Danbooru images/tags, feature vectors, recommendation results,
or model weights are included. You choose the material and build your own memory.
Artist, character, and series names are retained as core information.

## Choose a starting point

| What you want | Use | Requirements |
| --- | --- | --- |
| Random seasonal clothing and scene prompts | Seasonal Fashion Prompt Generator | ComfyUI; no model or extra packages |
| Write English tags or prose from Korean/English requests | RS AI Prompt Writer | ComfyUI + a separate Qwen Writer, distinct from Gemma TIPO |
| Expand Korean/English descriptions or tags | Gemma TIPO, text mode | ComfyUI + GGUF runtime/model; GPU-capable runtime plus `gpu_layers=-1` enables GPU offload |
| Generate prompt tags inspired by an image | Gemma TIPO, image mode | ComfyUI + GGUF + matching mmproj + llama.cpp vision runtime |
| Retrieve tags and names from your own reference collection | VRS-INT | Python 3.11+, NVIDIA CUDA environment, separate vision encoder, personal image/tag pairs |

VRS-INT retrieves creative reference material. It does not train the encoder,
reliably identify an image's author, or write a finished natural-language prompt.
Use its tags in your image prompt or a separate prompt writer. This public
version has no VRS-INT ComfyUI node or automatic Writer connection.

- **ComfyUI:** follow [installation](#install-for-comfyui), then the node guides below.
- **VRS-INT only:** start with the [complete VRS-INT guide](vrs_int/README.md). ComfyUI and the full Gemma language model are unnecessary.
- **Data preparation:** follow the [image/text pair walkthrough](examples/vrs-int/README.md).
- **한국어:** [전체 사용법](README_KOR.md) · [VRS-INT](vrs_int/README_KOR.md) · [쌍 구성 예제](examples/vrs-int/README_KOR.md).

## Install for ComfyUI

### 1. Install under `custom_nodes`

From your ComfyUI directory:

```bash
git clone https://github.com/raspie10032/RS-Seasonal-Prompt-Generator.git custom_nodes/RS-Seasonal-Prompt-Generator
```

Alternatively, extract the repository ZIP with this layout. Avoid an extra nested folder.

```text
ComfyUI/
  custom_nodes/
    RS-Seasonal-Prompt-Generator/
      __init__.py
      seasonal_fashion_prompt_node.py
      gemma_tipo_node.py
      data/
```

Restart ComfyUI. Find the nodes under **`prompt`** or search by display name.
All return `STRING`. To connect one to a text encoder, convert the encoder's
prompt/text widget to an input when required by your ComfyUI version. A text
preview node lets you inspect the string before generating an image.

### 2. Prepare optional TIPO dependencies

The seasonal node needs only Python's standard library. TIPO attempts dependency
installation and model downloads on its first actual generation, not at import.
For manual installation, run this from the repository using **ComfyUI's Python**:

```bash
python -m pip install -r requirements.txt
```

Windows portable, from `ComfyUI_windows_portable`:

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\RS-Seasonal-Prompt-Generator\requirements.txt
```

The dependencies are `tipo-kgen` and `llama-cpp-python>=0.3.23`. Restart ComfyUI
after upgrading an already loaded runtime. These are **TIPO requirements**, not
VRS-INT's CUDA/Transformers setup. Downloads need network access and disk space;
no paid inference API is used.

## Seasonal Fashion Prompt Generator

1. Add **Seasonal Fashion Prompt Generator**.
2. Choose a season, leave clothing at `random`, and set `background_toggle=on`.
3. Connect the output to a text preview or your image model's positive text input.
4. Run the workflow. Optionally connect it to **Gemma TIPO.prompt** for expansion.

| Input | Meaning |
| --- | --- |
| `season` | `random`, `spring`, `summer`, `autumn`, `winter` |
| `background_toggle` | `on`: seasonal scene; `off`: omit background block; `simple`: plain white/grey/black background |
| `weather_toggle`, `time_toggle` | `random`, `off`, or listed value; used when background is `on` |
| `top_toggle`, `bottom_toggle`, `one_piece_toggle` | Clothing choices; `random` or `off` |
| `accessory_toggle`, `hat_toggle`, `shoes_toggle`, `socks_toggle` | Random accessory/hat/shoes/socks or turn off |
| `composition_toggle`, `gaze_toggle`, `pose_toggle`, `body_direction_toggle` | `random`, `off`, or listed value |
| `additional_situation_toggle` | Add CSV situation details when background/weather/time conditions match |
| `wanna_shot` | `Yes`: special café/coffee/tumbler scene instead of the normal scene path |
| `seed` | Reproducible selection, including seed 0 |

The same seed and inputs reproduce the selection, including seed `0`, without
changing the global random state. Top and bottom toggles work independently.
When both separates and a one-piece are available, one outfit type is selected.

### Customize CSVs

Edit UTF-8 files in [`data/`](data/), preserve headers/columns, and restart
ComfyUI to reload cached data. Back up personal changes before updating code.

| Files | Contents |
| --- | --- |
| `seasonal_fashion.csv` | Season, category, item. Categories: `top`, `bottom`, `one_piece`, `accessory`, `hat`, `shoes`, `socks` |
| `seasonal_backgrounds.csv` | Season, background classification, description |
| `seasonal_weather.csv`, `seasonal_times.csv` | Weather and time choices by season |
| `general_composition.csv`, `gaze_direction.csv`, `poses.csv`, `body_directions.csv` | Composition, gaze, pose, and body-direction choices |
| `additional_situations.csv` | Situation, `\|`-separated matching alternatives, description |

Quote CSV fields containing commas. Situation conditions support `|` for OR
and `&` for AND, for example `misty & lakeside|foggy & lakeside`.

## AI prompt writing

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

## Gemma TIPO: text or image to prompt tags

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

## VRS-INT: personal reference memory

```text
Your images + categorized tags → existing vision encoder → versioned personal memory
New reference image           → same vision encoder     → neighbor tags + name candidates
```

1. [Set up Python, CUDA, and the vision-only encoder](vrs_int/README.md#setup).
2. [Prepare JSONL or image-sidecar pairs](examples/vrs-int/README.md).
3. [Check, build, and recommend](vrs_int/README.md#first-build-and-recommendation).
4. [Append or correct a batch](vrs_int/README.md#update-a-collection).
5. [Back up, transfer, and roll back](vrs_int/README.md#backup-transfer-and-rollback).

VRS-INT uses a separate Gemma E2B vision tower, not the language model or
`mmproj.gguf`. Explicit artist/character/series names survive even with one
occurrence. Captions and original tags remain audit metadata; a caption alone
cannot build a searchable tag memory.

Keep personal material outside the repository or in ignored `vrs-int-data/` and
`vrs-int-models/`. Compressed memories retain tags and source metadata after
decoding; compression is not anonymization. No automatic collection, upload, or
shared-memory synchronization is performed.

## Gemma checkpoint and experience inputs

See [Gemma 4 TIPO runtime](vrs_int/GEMMA4_RUNTIME.md) for the **vision FP16 / PLE Q4**
candidate, explicit model selection and measured limits. Text layers and vocabulary
remain intact. This candidate is not a newly completed TIPO fine-tune. Internal
FFN computation replay is not part of this checkpoint.

- [Generated image metadata and optional tagging](vrs_int/GENERATED_IMAGES.md)
- [Korean text knowledge and translation dictionary](vrs_int/TAG_DICTIONARY.md)
- [VRS reference state → TIPO expansion](vrs_int/GENERATION_STATE.md)

The state connection is a CLI workflow. Negation and scene constraints still need
output review. Vision TIPO exits its process after each call; legacy text TIPO
caches its model. Both use two CPU helper threads by default.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Nodes missing | Folder nesting, restart, startup import errors; search `prompt`. VRS-INT is a CLI, not a node. |
| TIPO returns original text | Dependency/model/inference failure can cause fallback. Read `[TIPO]` console messages. |
| Vision output empty | Read `[TIPO-V]` logs. Check model/projector pairing, executable libraries, input image, and 600-second inference timeout. |
| `unknown model architecture: gemma4` | Check Gemma4-capable llama-cpp-python ≥0.3.23 in ComfyUI's interpreter; restart after upgrading. |
| CPU remains busy | Check TIPO `gpu_layers=-1` and a GPU-capable runtime. Legacy vision can retry CPU after GPU failure; inspect logs. VRS-INT settings are separate. |
| Installed package still missing | Use ComfyUI's Python, not an unrelated system/venv interpreter. |
| Model download failure | Check URL, network, disk, and console logs. Downloads reject empty or truncated transfers; verify custom files separately. |
| VRS-INT errors | See [VRS-INT troubleshooting](vrs_int/README.md#troubleshooting). |

## Updates and validation

For an unmodified Git installation, from the repository directory:

```bash
git pull --ff-only
```

Restart ComfyUI if using nodes. Preserve local CSV/code edits before updating.
Code updates do not fetch a corpus or migrate/activate personal memories. Record
your exact snapshot hash and encoder provenance.

The public offline suite uses fictional fixtures and substitute encoders:

```bash
python -m pip install numpy Pillow
python -m unittest discover -s tests -v
python scripts/check_code_only.py
```

It checks inputs, metadata, snapshots, updates, packages, and publication rules.
CI covers Linux/Windows/macOS and Python 3.11/3.13. This is not GPU inference or
recommendation-quality validation. The publication checker reads indexed Git
blobs, including staged changes; review the diff too. Historical node changelogs:
[English](Update_log_ENG.txt) / [한국어](Update_log_KOR.txt).

## License and credits

Code: [Apache-2.0](LICENSE). Model weights, datasets, and images retain their own
terms; the code license does not relicense them.

| Project | Contribution |
| --- | --- |
| [KGen / TIPO](https://github.com/KohakuBlueleaf/KGen), [z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension) | Prompt expansion concept, formatter, TIPO training-data lineage |
| [llama.cpp](https://github.com/ggml-org/llama.cpp), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | GGUF inference |
| [Gemma](https://ai.google.dev/gemma) | Base family; separate model terms |
| [raspie/gemma4-tipo-ko-gguf](https://huggingface.co/raspie/gemma4-tipo-ko-gguf) | Default Korean TIPO model |
| [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | Earlier base projector source; current vision defaults use the paired trained projector |
| [p-e-w/gemma-4-E2B-it-heretic-ara](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara) | TIPO base lineage and source of VRS-INT's separately prepared vision tower |

Originally created with ChatGPT, then developed with Claude Code and Codex.
