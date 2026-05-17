# RS-Seasonal-Prompt-Generator
 Generates season-specific fashion prompts by randomly combining fashion items, background settings, weather, time, and additional situational details from CSV data.

> **v2.0.0** adds optional TIPO prompt expansion (local GGUF, opt-in). See the changelog (`Update_log_ENG.txt` / `Update_log_KOR.txt`).

## Usage

1. **Install the node**: clone or copy this folder into
   `ComfyUI/custom_nodes/` and restart ComfyUI.
2. In ComfyUI, add the node **`Seasonal Fashion Prompt Generator`**
   (category: `prompt`). It outputs a single `STRING` you can wire into any
   text/CLIP encode node.
3. **Base use (no extra dependencies):** pick a `season` and toggle the
   fashion / background / weather / pose options as desired, then queue.
   The node returns a comma-separated prompt. This is the original behavior
   and needs nothing installed.
4. **TIPO expansion (optional):** set `tipo_toggle` to `on` to expand the
   generated prompt into a richer Danbooru tag set. On the first run it
   auto-installs its dependencies and (if `tipo_gguf_path` is empty)
   downloads the default model — see *Install* below.

### TIPO examples

The TIPO path also accepts free natural-language text (Korean or English),
not just tags — useful if you feed it from a text box instead of the
seasonal generator:

| Input | Output (example) |
| --- | --- |
| `spring, white blouse, pleated skirt, park` | `spring, white blouse, pleated skirt, park, outdoors, school uniform, long sleeves, sitting, solo, …` |
| `빨간 원피스, 밀짚모자, 해변` (Korean) | `straw hat, red one-piece swimsuit, sun hat, beach, ocean, swimsuit, outdoors, …` |

Notes:
- Longer / more detailed descriptions yield more and more accurate tags.
- The raw input text (incl. Korean) is **not** mixed into the tag output.
- Use `tipo_ban_tags` to strip unwanted tags, and `tipo_sort` to control
  category ordering.
- Output quality depends on the GGUF model; on any failure the node falls
  back to the un-expanded prompt (it never hard-errors).

## TIPO prompt expansion (optional)

The node can optionally expand its generated prompt into a richer Danbooru-style
tag set. Inference runs **in-process from a local GGUF model** via
`llama-cpp-python`, so you can use your own custom GGUF (e.g. a Gemma-4-E2B
fine-tune) **as-is**. The output is then cleaned up with the lightweight,
pure-Python tag formatter from [`tipo-kgen`](https://github.com/KohakuBlueleaf/KGen)
(`kgen.formatter` only — no torch/transformers is loaded).

- Set `tipo_toggle` to `on` to enable expansion.
- `tipo_gguf_path`: leave **empty** to auto-download the default model
  ([`raspie/gemma4-tipo-ko-gguf`](https://huggingface.co/raspie/gemma4-tipo-ko-gguf))
  from HuggingFace on first use (cached by `huggingface_hub`). Or set a local
  `.gguf` file / folder path to use your own (a `*q4*` file is preferred when
  several are present).
- `tipo_tag_length`: target verbosity (`very_short` … `very_long`).
- `tipo_sort`: prompt-ordering preset applied by `kgen.formatter` — tags are
  categorized (special/characters/copyrights/artist/general/quality/meta/rating)
  and emitted in the preset's order:
  - `danbooru` (default): artist, rating, special, characters, copyrights,
    general, quality, meta.
  - `quality_first`: quality tags lead.
  - `artist_first`: artist/character lead.
  - `general_only`: keep only special + characters + general tags.
- `tipo_temperature`: sampling temperature.
- `tipo_ban_tags`: comma-separated tags to strip from the result.

When `tipo_toggle` is `off` (default) the node behaves exactly as before and
needs no extra dependencies.

### Install

**No manual install is required.** The first time you actually run the node
with `tipo_toggle` set to `on`, the missing dependencies are auto-installed
into the active environment: `tipo-kgen` (pure-Python) and `llama-cpp-python`
(from the prebuilt CPU wheel index, so no native build). This happens once and
only when TIPO is first used — with `tipo_toggle` off the node never installs
anything.

To pre-install manually instead (optional):

```
pip install -r requirements.txt
```

If the prebuilt `llama-cpp-python` wheel install fails to find a wheel for your
platform/Python, install it yourself once:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

If `tipo_gguf_path` is empty the default model is downloaded from HuggingFace
on first use (cached); otherwise your local path is used. If
`llama-cpp-python` is missing, the model can't be obtained, or inference
fails, the node falls back to the un-expanded prompt instead of erroring.

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
