# RS-Seasonal-Prompt-Generator

English | [한국어](README_KOR.md)

Create image-generation prompt ideas from seasonal CSVs, descriptions, or your
own reference images and tags. This repository contains two **ComfyUI nodes**
and the standalone **VRS-INT** command-line tool.

**VRS-INT ships code, documentation, and fictional examples only.** No starter
memory, collected Danbooru images/tags, feature vectors, recommendation results,
or model weights are included. You choose the material and build your own memory.
Artist, character, and series names are retained as core information.

## Choose a starting point

| What you want | Use | Requirements |
| --- | --- | --- |
| Random seasonal clothing and scene prompts | Seasonal Fashion Prompt Generator | ComfyUI; no model or extra packages |
| Expand Korean/English descriptions or tags | Gemma TIPO, text mode | ComfyUI + GGUF runtime/model; public text path uses CPU |
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

Restart ComfyUI. Find both nodes under **`prompt`** or search by display name.
Both return `STRING`. To connect one to a text encoder, convert the encoder's
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
| `seed` | Selection seed; see the public-version behavior below |

For repeatable selection, use a fixed **16-digit seed**, e.g. `1234567890123456`,
and keep ComfyUI's seed control fixed. Smaller seeds, including `0`, involve
random normalization in this public version. Top/bottom selection requires both
items; enabling just one can omit it. For separates, enable both and turn off
the one-piece. If all three are available, the node chooses between them.

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

Quote CSV fields containing commas. Situation conditions are exact alternatives
separated by `|`; this public version does not interpret `&` as an AND expression.

## Gemma TIPO: text or image to prompt tags

![Gemma TIPO in ComfyUI](assets/gemma-tipo-example.png)

### Text mode

1. Add **Gemma TIPO (Prompt / Image → Tags)** without connecting `image`.
2. Enter Korean/English text or comma-separated tags in `prompt`, e.g. `spring, white blouse, pleated skirt, park`.
3. Leave `model=(auto: download default)`. First use downloads `gemma4-tipo-ko-v2-Q4_K_M.gguf` into `ComfyUI/models/gguf/`.
4. Start with `tag_length=long`, `sort=danbooru`, `temperature=0.5`.
5. Inspect the output. The model can add, omit, or change details; expansion does not guarantee faithful translation.

**The public text engine sets `n_gpu_layers=0`.** Installing a GPU wheel or
changing the widget's `gpu_layers` does not move this text path to GPU. It can
be slow and CPU intensive. VRS-INT uses a separate CUDA path.

### Image mode

![Gemma TIPO image mode](assets/gemma-tipo-vision-example.png)

Connect **Load Image → Gemma TIPO.image**, then send the string to a preview or
text encoder. An image selects vision mode; `prompt` is ignored. Use one image
per request.

With automatic `model` and empty `mmproj_path`, the public defaults are the TIPO
GGUF above and base Gemma E2B `mmproj-F16.gguf`. First use also obtains a
platform-specific llama.cpp `b9209` executable under `ComfyUI/models/tipo_bin/`.
These are **not VRS-INT's separate vision encoder**.

The public resolver selects generic/CPU release assets; automatic CUDA runtime
selection is not implemented. To use a prepared compatible GPU executable, set
`LLAMA_MTMD_CLI` to its full path before starting ComfyUI, keep its runtime
libraries available, and request positive `gpu_layers`. Actual offload depends
on the executable/hardware. The public widget accepts `0`–`100`, not `-1`.

Image mode expands plausible prompt ideas, including details absent from the
picture. Output tags are not verified labels of the input image.

### TIPO controls

| Input | Meaning |
| --- | --- |
| `prompt` | Korean/English description or tags; text mode only |
| `model` | Automatic default or local `.gguf` in `ComfyUI/models/gguf/`; refresh/restart after adding files |
| `tag_length` | `very_short`, `short`, `long`, `very_long`; approximate target, not exact count |
| `sort` | `danbooru`, `quality_first`, `artist_first`, `general_only`, `simple` |
| `temperature`, `seed` | Sampling variation; seeds do not promise identical output across runtimes |
| `ban_tags` | Comma-separated tags removed during successful category formatting |
| `image` | Optional image; connecting it selects vision mode |
| `mmproj_path` | Vision only; empty for default, otherwise supply a compatible projector |
| `gpu_layers` | Vision only in public code; `0` defaults to CPU, positive values request offload |

`general_only` retains special/character/general categories and drops artists
and series. `simple` retains special/rating/general. Use `danbooru` or
`artist_first` to retain artist, character, and series candidates during formatting.
TIPO requires several GB for its model and runtime; vision adds a projector and
executable. Its requirements are separate from the smaller VRS-INT encoder path.

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

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Nodes missing | Folder nesting, restart, startup import errors; search `prompt`. VRS-INT is a CLI, not a node. |
| TIPO returns original text | Dependency/model/inference failure can cause fallback. Read `[TIPO]` console messages. |
| Vision output empty | Read `[TIPO-V]` logs. Check model/projector pairing, executable libraries, input image, and 600-second inference timeout. |
| `unknown model architecture: gemma4` | Check Gemma4-capable llama-cpp-python ≥0.3.23 in ComfyUI's interpreter; restart after upgrading. |
| CPU remains busy | Public TIPO text is CPU-only; vision is not automatic CUDA setup. VRS-INT GPU settings do not configure TIPO. |
| Installed package still missing | Use ComfyUI's Python, not an unrelated system/venv interpreter. |
| Model download failure | Check URL, network, disk, and console logs. A present file does not prove a complete download. |
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
python -m pip install numpy
python -m unittest discover -s tests -p "test_vrs_int*.py" -v
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
| [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | Public TIPO default projector |
| [p-e-w/gemma-4-E2B-it-heretic-ara](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara) | TIPO base lineage and source of VRS-INT's separately prepared vision tower |

Originally created with ChatGPT, then developed with Claude Code and Codex.
