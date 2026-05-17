# RS-Seasonal-Prompt-Generator
 Generates season-specific fashion prompts by randomly combining fashion items, background settings, weather, time, and additional situational details from CSV data.

> **v2.1.0** splits the project into **two independent nodes**: the original
> CSV random generator, and a separate Gemma TIPO text→tags node. See the
> changelog (`Update_log_ENG.txt` / `Update_log_KOR.txt`).

This pack provides **two separate nodes** (category: `prompt`). Install by
cloning/copying this folder into `ComfyUI/custom_nodes/` and restarting
ComfyUI. Each outputs a single `STRING`; you can chain node 1 → node 2.

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
- `tipo_gguf_path`: leave **empty** to auto-download the default model
  ([`raspie/gemma4-tipo-ko-gguf`](https://huggingface.co/raspie/gemma4-tipo-ko-gguf))
  on first use into **`ComfyUI/models/gguf/`** (reused on later runs, never
  re-downloaded). Or set a local `.gguf` file/folder path to use your own
  (a `*q4*` file is preferred when several are present).
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
- On any failure (missing deps, model unavailable, inference error) it falls
  back to the input text unchanged — it never hard-errors.

### Install (Node 2 only)

**No manual install is required.** The first time you actually run Node 2,
its dependencies are auto-installed into the active environment once:
`tipo-kgen` + `huggingface_hub` (pure-Python) and `llama-cpp-python` (from the
prebuilt CPU wheel index, so no native build). Node 1 never installs anything.

To pre-install manually instead (optional):

```
pip install -r requirements.txt
```

If the prebuilt `llama-cpp-python` wheel install can't find a wheel for your
platform/Python, install it yourself once:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

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
