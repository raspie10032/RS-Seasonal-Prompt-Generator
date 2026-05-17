# RS-Seasonal-Prompt-Generator
 Generates season-specific fashion prompts by randomly combining fashion items, background settings, weather, time, and additional situational details from CSV data.

## TIPO prompt expansion (optional)

The node can optionally expand its generated prompt into a richer Danbooru-style
tag set. Inference runs **in-process from a local GGUF model** via
`llama-cpp-python`, so you can use your own custom GGUF (e.g. a Gemma-4-E2B
fine-tune) **as-is**. The output is then cleaned up with the lightweight,
pure-Python tag formatter from [`tipo-kgen`](https://github.com/KohakuBlueleaf/KGen)
(`kgen.formatter` only — no torch/transformers is loaded).

- Set `tipo_toggle` to `on` to enable expansion.
- `tipo_gguf_path`: path to your `.gguf` file, or a folder containing one
  (a `*q4*` file is preferred when several are present).
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

```
pip install -r requirements.txt
```

`tipo-kgen` is pure-Python. `llama-cpp-python` is native — if the standard
install fails to build, use the prebuilt CPU wheel index:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

The GGUF model is loaded from the local `tipo_gguf_path` you provide (nothing
is downloaded automatically). If `llama-cpp-python` is missing, the path is
empty/invalid, or inference fails, the node falls back to the un-expanded
prompt instead of erroring.

## References & Credits

| Project | Use | License |
| --- | --- | --- |
| [KGen / TIPO](https://github.com/KohakuBlueleaf/KGen) by KohakuBlueleaf | `kgen.formatter` for tag categorization & prompt sorting | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference engine | MIT |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | Python bindings for in-process GGUF inference | MIT |
| [Gemma](https://ai.google.dev/gemma) by Google | Base model family for the GGUF used in TIPO expansion | Gemma Terms of Use |
| [NAI-FaceDetailer](https://github.com/raspie10032/NAI-FaceDetailer) | Origin of the ported TIPO chat-completion expansion approach | — |

The TIPO expansion concept ("upsample"/expand short prompts into detailed
Danbooru tag sets) originates from KohakuBlueleaf's KGen/TIPO and the
[z-tipo-extension](https://github.com/KohakuBlueleaf/z-tipo-extension).
