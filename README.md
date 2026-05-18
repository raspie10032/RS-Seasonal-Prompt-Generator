# RS-Seasonal-Prompt-Generator
 Generates season-specific fashion prompts by randomly combining fashion items, background settings, weather, time, and additional situational details from CSV data.

> **v2.6.0** adds a third node: **Gemma TIPO Vision (Image → Tags)**. See
> the changelog (`Update_log_ENG.txt` / `Update_log_KOR.txt`).

This pack provides **three independent nodes** (category: `prompt`):
1) Seasonal Fashion Prompt Generator (CSV, zero-dep), 2) Gemma TIPO
Prompt → Tags (text → tags), 3) Gemma TIPO Vision (image → tags). Install
by cloning/copying this folder into `ComfyUI/custom_nodes/` and
restarting ComfyUI. Each outputs a `STRING`; they're chainable.

## Showcase

![Gemma TIPO node in ComfyUI](assets/gemma-tipo-example.png)

Korean natural language in → clean English Danbooru tags out:

> **In:** `소녀가 하얀 세일러복과 스카프를 하고 있음.`
>
> **Out:** `white sailor collar, white skirt, black hairband, blue eyes,
> blush stickers, long sleeves, looking at viewer, open mouth, red scarf,
> school uniform, serafuku, sitting, solo focus, sparkle sticker, striped
> clothes, sweatdrop, collared shirt, white background`

## Node 1 — Seasonal Fashion Prompt Generator

The original generator. Pick a `season` and toggle the fashion / background /
weather / pose / situation options; it returns a comma-separated prompt
randomly combined from the bundled CSV data. **Zero dependencies** — nothing
is installed and nothing runs a model.

## Node 2 — Gemma TIPO Prompt → Tags

Standalone converter: feed it any text (typed in the `prompt` box, or wired
from Node 1 or any text node) and it expands it into a richer, category-sorted
Danbooru tag set using a local GGUF model run **in-process** via
`llama-cpp-python`. Post-processing/sorting uses the lightweight pure-Python
formatter from [`tipo-kgen`](https://github.com/KohakuBlueleaf/KGen)
(`kgen.formatter` only — no torch/transformers loaded).

It accepts **Korean or English natural language**, not just tags:

| Input | Output (example) |
| --- | --- |
| `spring, white blouse, pleated skirt, park` | `spring, white blouse, pleated skirt, park, outdoors, school uniform, long sleeves, sitting, solo, …` |
| `빨간 원피스, 밀짚모자, 해변` (Korean) | `straw hat, red one-piece swimsuit, sun hat, beach, ocean, swimsuit, outdoors, …` |

Inputs:
- `prompt`: text to convert (Korean/English NL or tags).
- `model`: a dropdown of the `*.gguf` files found in
  **`ComfyUI/models/gguf/`**. Pick `(auto: download default)` to fetch the
  default model
  ([`raspie/gemma4-tipo-ko-gguf`](https://huggingface.co/raspie/gemma4-tipo-ko-gguf))
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
- `temperature`: sampling temperature.
- `ban_tags`: comma-separated tags to strip from the result.
- `seed`: changes the expansion for variation.

Notes:
- Longer / more detailed descriptions yield more and more accurate tags.
- The raw input text (incl. Korean) is **not** mixed into the tag output.
- Redundant general tags are dropped to save tokens: if a more specific tag
  with the same head noun exists (e.g. `white skirt`), the bare `skirt` is
  removed. Distinct tags like `dress`/`dress shirt` are kept.
- Conflicting person-count tags are collapsed per group: with `1girl`
  present, `4girls` / `multiple girls` are removed (`1girl` + `1boy` is
  kept — groups are independent).
- On any failure (missing deps, model unavailable, inference error) it falls
  back to the input text unchanged — it never hard-errors.

### Supported environment & requirements

| | |
| --- | --- |
| **OS** | Windows / Linux / macOS — anywhere ComfyUI runs and a `llama-cpp-python` wheel exists (x86-64, Apple Silicon/arm64) |
| **Node 1 (Seasonal)** | Any ComfyUI install. Zero dependencies, no model. |
| **Node 2 (Gemma TIPO)** | CPU-only works (no GPU required); GPU optional for speed |
| **RAM** | min ~8 GB (≈5 GB used for the 4-bit model) · 16 GB+ recommended |
| **Disk** | ~4 GB free (3.2 GB GGUF + deps) |
| **Speed** | CPU inference works but is slow (a single prompt can take minutes); a GPU-accelerated `llama-cpp-python` build is recommended for speed |
| **Python** | ComfyUI's embedded env; `llama-cpp-python >= 0.3.23` (auto-installed/upgraded, one restart prompt) |

### Install (Node 2 only)

**No manual install is required.** The first time you actually run Node 2,
its dependencies are auto-installed into the active environment once:
`tipo-kgen` (pure-Python) and `llama-cpp-python` **>= 0.3.23**
(from the prebuilt CPU wheel index, so no native build). Older builds cannot
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

## Node 3 — Gemma TIPO Vision (Image → Tags)

Feed a ComfyUI `IMAGE` and it captions it into a rich, category-sorted
Danbooru tag set using **Gemma-4 vision** (llama.cpp `mtmd`) + the base
**gemma-4-E2B `mmproj`**. TIPO-style — it captures the image *and*
plausibly expands it (creative upsampling, not exact labeling), so it
pairs well with text-to-image workflows.

- `image`: the image to caption.
- `model`: dropdown of `*.gguf` in `ComfyUI/models/gguf` (or
  `(auto: download default)`). Any gemma-4-E2B text/merged TIPO model
  works — it's paired with the auto-downloaded base mmproj.
- `tag_length` / `sort` / `temperature` / `ban_tags` / `seed`: same as
  Node 2; output runs through the same kgen post-processing.
- optional `mmproj_path` (empty = auto-download base gemma-4-E2B mmproj),
  `gpu_layers` (default 0 = CPU, won't fight other GPU work).

On first use it auto-downloads a **prebuilt llama.cpp mtmd binary**
(no build) and the **mmproj** into `ComfyUI/models/`. Needs no extra pip
packages beyond Node 2's (`tipo-kgen`; PIL/numpy come with ComfyUI). Any
failure (binary/mmproj/model unavailable, inference error) returns an
empty string — it never hard-errors. CPU inference is slow; raise
`gpu_layers` to offload if you have spare VRAM.

> Note: Gemma-4 tends to emit an internal reasoning block; the node
> strips it and salvages the tag tokens, so output is tag-only but may
> vary run to run (that's the intended TIPO behavior).

## References & Credits

| Project | Use | License |
| --- | --- | --- |
| [KGen / TIPO](https://github.com/KohakuBlueleaf/KGen) by KohakuBlueleaf | `kgen.formatter` for tag categorization & prompt sorting | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference engine | MIT |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | Python bindings for in-process GGUF inference | MIT |
| [Gemma](https://ai.google.dev/gemma) by Google | Base model family for the GGUF used in TIPO expansion | Gemma Terms of Use |

The TIPO expansion concept ("upsample"/expand short prompts into detailed
Danbooru tag sets) originates from KohakuBlueleaf's KGen/TIPO and the
[z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension).
