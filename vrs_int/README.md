# VRS-INT: personal image/tag memory

English | [한국어](README_KOR.md) | [Project overview](../README.md) | [Pair examples](../examples/vrs-int/README.md)

Build a versioned memory from your own images and categorized tags, then use a
new reference image to retrieve material for a new prompt. The existing Gemma
E2B vision encoder extracts features; VRS-INT stores observations and retrieves
similar examples. Building memory **does not train encoder weights**.

**Code and fictional examples only:** no corpus, starter memory, vectors, or
weights are distributed. This standalone storage/retrieval path does not invoke
the native VRS judgment core. It is not a WD14 classification model or an author
identification service. A separate writer is needed for natural-language prompts.

## More inputs and Gemma integration

- [Generated image experience](GENERATED_IMAGES.md): original generation metadata with optional local PixAI tagging.
- [Korean text knowledge](TAG_DICTIONARY.md): meaning, English tag candidates and names from a separate personal CSV.
- [Generation state](GENERATION_STATE.md): keep/change/exclude conditions and references passed as text.
- [Gemma checkpoint](GEMMA4_RUNTIME.md): vision FP16 / PLE Q4 selection and measured limitations.

Generated-image memories support `vrs-int/2` and `u` (unrated). Existing Danbooru
memories retain `vrs-int/1`; older code is not assumed to read the new format.
The Danbooru examples below remain valid. Artist/character/copyright categories
are retained; quality/year tags are excluded from recommendations. This does not
guarantee recognition or instruction following.


## Contents

- [Setup](#setup)
- [First build and recommendation](#first-build-and-recommendation)
- [Understand recommendation output](#understand-recommendation-output)
- [Update a collection](#update-a-collection)
- [Backup, transfer, and rollback](#backup-transfer-and-rollback)
- [Storage, resources, and compatibility](#storage-resources-and-compatibility)
- [Command reference](#command-reference)
- [Troubleshooting](#troubleshooting)

## Setup

### 1. Clone and use a separate Python environment

Run commands from the **repository root**, not from `vrs_int/`. No ComfyUI
installation or package installation of this repository is required.

```bash
git clone https://github.com/raspie10032/RS-Seasonal-Prompt-Generator.git
cd RS-Seasonal-Prompt-Generator
```

Linux, using Python 3.11 or later:

```bash
python3 -m venv .venv-vrs
source .venv-vrs/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell, with Python 3.11 installed:

```powershell
py -3.11 -m venv .venv-vrs
.\.venv-vrs\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is unavailable, use `.venv-vrs/bin/python` or
`.\.venv-vrs\Scripts\python.exe` instead of `python`; no execution-policy
change is required. All workflow commands below are single lines and work in
Bash or PowerShell with the documented placeholders replaced.

| Operation | Dependencies |
| --- | --- |
| `check-input`, sidecar preparation, `inspect`, explicit encoder download | Python standard library |
| Verify/store/export/install/activate memory containing vectors; offline tests | NumPy; no GPU or model |
| `build`, `recommend` | CUDA PyTorch, NumPy, Pillow, Transformers with Gemma4 PIL support, Accelerate, bitsandbytes, safetensors; compatible NVIDIA GPU |

Install the CUDA-enabled PyTorch build appropriate for your OS and driver using
the [official PyTorch selector](https://pytorch.org/get-started/locally/). Use
this virtual environment's `python -m pip` when applying its command. Then:

```bash
python -m pip install numpy pillow transformers accelerate bitsandbytes safetensors
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from transformers import Gemma4VisionModel; from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil; print('Gemma4 PIL API available')"
```

The second check must report `CUDA: True`; the third must import both classes.
Use a Transformers release providing these APIs; root `requirements.txt` is for
TIPO and is insufficient here. See the official [Gemma4 documentation](https://huggingface.co/docs/transformers/model_doc/gemma4)
and [bitsandbytes compatibility guide](https://huggingface.co/docs/bitsandbytes/installation)
if your environment lacks them. Offline CI on macOS does **not** establish GPU
support there: this adapter has no CPU, MPS, or alternative GPU backend fallback.

### 2. Prepare the separate vision-only encoder

This is the one explicit model-download step. It does not run inference:

```bash
python -m vrs_int.prepare_encoder ./vrs-int-models/gemma4-e2b-vision
```

Choose a destination that does **not** already exist. The helper requests only
the contiguous vision tensors using HTTP byte ranges, not the full language
model. Its weight source is `p-e-w/gemma-4-E2B-it-heretic-ara` at
`c9a1d4c031981f14d86eeb0c7d87de7fafd34513`. Processor configuration is obtained
from `unsloth/gemma-4-e2b-it`; the revision resolved at preparation time is recorded.

Keep this entire directory together:

```text
vrs-int-models/gemma4-e2b-vision/
  model.safetensors
  config.json
  preprocessor_config.json
  provenance.json
```

This is a separate vision tower, **not `mmproj.gguf`**. Runtime uses NF4 linear
layers, BF16 computation, PIL preprocessing, and mean-pooling plus normalization
to produce 768-dimensional vectors. `build`/`recommend` verify local files and
do not download a missing model. Model terms remain separate from the code license.

Preserve the exact directory for later updates and queries. Re-running preparation
can resolve a different processor revision, so the resulting encoder is not
automatically interchangeable even when the base weights are unchanged.

### 3. Select the GPU explicitly

```bash
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
```

Copy your chosen GPU's complete `GPU-...` UUID. Replace **`GPU-YOUR-UUID`** in
the following commands. With multiple cards this prevents selecting an unintended
device. `build` and `recommend` fail if CUDA is unavailable; no CPU inference retry.

## First build and recommendation

### 1. Prepare actual images and metadata

Follow the [pair walkthrough](../examples/vrs-int/README.md). Example local layout:

```text
vrs-int-data/
  pairs/
    1001.png
    1002.jpg
  batch-01.jsonl
  query.png
```

The repository's examples use fictional names and `g` ratings; **image files are
not bundled**. Use your own files and matching metadata. The manifest uses one
JSON object per line; image paths are relative to `--images`, not to the JSONL.
The future `query.png` can be outside the collection and needs no tags.

### 2. Validate before loading the model

```bash
python -m vrs_int check-input --manifest ./vrs-int-data/batch-01.jsonl --images ./vrs-int-data/pairs
```

This checks IDs, tag fields, ratings, paths, file hashes, and retained tags. It
does **not** decode pixels or prove that the images are readable or well labeled.
No model/GPU is loaded. Fix reported line-number errors before building.

### 3. Build a snapshot

```bash
python -m vrs_int build --manifest ./vrs-int-data/batch-01.jsonl --images ./vrs-int-data/pairs --encoder ./vrs-int-models/gemma4-e2b-vision --store ./vrs-int-data/memory --gpu GPU-YOUR-UUID --label personal-01
```

Read `snapshot` in the JSON output: it is a **64-character SHA-256 content ID**.
`observations`, `vectors`, `source_records`, and `snapshot_bytes` describe the
built snapshot. Save the output and hash. In commands below, **replace
`SNAPSHOT_1` with that exact hash**, without brackets or a label such as `personal-01`.

```bash
python -m vrs_int verify --store ./vrs-int-data/memory --snapshot SNAPSHOT_1
python -m vrs_int inspect --store ./vrs-int-data/memory --snapshot SNAPSHOT_1
```

`verify` reads all referenced objects and checks hashes/counts. `inspect` displays
the manifest, including the encoder contract; it does not list every source tag
and is not a replacement for `verify`.

### 4. Recommend from a new image

```bash
python -m vrs_int recommend --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --encoder ./vrs-int-models/gemma4-e2b-vision --image ./vrs-int-data/query.png --gpu GPU-YOUR-UUID --ratings g,s --neighbors 32 --limit 20 --output ./vrs-int-data/recommendation-01.json
```

Open the UTF-8 JSON result in a text editor. The output path must be new; existing
files are never overwritten. Use a new filename for each query.

`--ratings g,s` restricts the **stored neighbors** considered. All four `g,s,q,e`
ratings are supported and the CLI default includes all four if omitted. This
filter does not classify the query or sanitize its tags. Write comma-separated
codes without spaces. `--neighbors` accepts 1–256; `--limit` accepts 1–100.

## Understand recommendation output

| Field | Meaning |
| --- | --- |
| `snapshot`, `query_image_sha256` | Exact memory and input file used |
| `contract_sha256` | Encoder/storage contract hash when neighbors are found |
| `candidates` | Top tags, up to `--limit`, including any categories that win votes |
| `identities.artist`, `.character`, `.copyright` | Separate top-three name candidates for each explicitly known category |
| `score` | Similarity-weighted neighbor vote, **not** accuracy or identity probability |
| `support` | Number of retrieved neighbors carrying the tag |
| `observation_keys`, `image_ids` | Stored examples supporting that tag |
| `neighbors` | Matching observation IDs, cosine similarities, ratings, image hashes |
| `visual_verdict: "U"` | Visual attribution remains unproved; it is intentionally retained even with candidates |

Only entries with a vector and `split=train` can vote. The exact query file is
excluded by SHA-256, and identical neighbor files are counted once. Resized or
re-encoded copies have different hashes and are not detected as near-duplicates.
Missing typed fields cannot be reconstructed into author/character/series labels
by guessing names. If no eligible neighbors remain, lists are empty and verdict
is `U`; even nonempty results can be weak because there is no semantic confidence
threshold. A memory of a few pictures is a workflow demonstration, not coverage.

For prompt creation, review candidates, keep your specified artist/character/series,
then combine useful clothing, setting, and composition tags. Retrieved names are
creative references, not claims about the source image. Copy your selection into
ComfyUI or another writer; automated prompt composition is not connected here.

## Update a collection

Put **only new or corrected complete records** in `batch-02.jsonl`. Prepare the
actual images for those records and check the batch, then:

```bash
python -m vrs_int check-input --manifest ./vrs-int-data/batch-02.jsonl --images ./vrs-int-data/pairs
python -m vrs_int build --manifest ./vrs-int-data/batch-02.jsonl --images ./vrs-int-data/pairs --encoder ./vrs-int-models/gemma4-e2b-vision --store ./vrs-int-data/memory --gpu GPU-YOUR-UUID --base SNAPSHOT_1 --label personal-02
```

Save the returned hash as **`SNAPSHOT_2`** and pass it explicitly to subsequent
queries. A label is descriptive, not a moving alias. Building never activates a
snapshot or overwrites the old one.

- Same ID (`danbooru:<positive integer>`) replaces that observation; new ID adds one.
- A correction is a **full replacement**, not a tag merge. Include every wanted artist, character, series, general tag, and caption again.
- The CLI re-encodes each submitted image, even for a tags-only correction. Unchanged records' images are unnecessary, but corrected/new records' images must exist.
- Resubmitting the same ID with identical content does not add an extra observation/vote; an update with unchanged stored objects returns its parent. Paths/metadata changes also count as content changes. Different IDs can remain separate observations even when their images match; search deduplicates identical image hashes.
- Keep original IDs stable and unique. Do not combine unrelated sources by reusing IDs: the current key namespace is `danbooru:`.
- Keep encoder, codec, tag policy, and preprocessing compatible. `--buckets` chooses the initial layout; an update inherits its parent's layout.

The [two-batch example](../examples/vrs-int/README.md#add-images-and-correct-tags)
shows two initial images, one corrected record, and one new image. Tag correction
keeps the earlier source record in the older snapshot, not as an appended history
inside the replacement record.

## Backup, transfer, and rollback

### Personal backup

Preserve the whole `memory/` directory and exact encoder directory; keep original
image/metadata backups separately. A memory contains vectors and metadata, **not
image bytes**. Normal recommendation needs only memory, matching encoder, and the
new query image. Re-encoding after an encoder change needs original images.

### Full package and incremental package

These commands create **directories**, not ZIP files. Output directories must be new.

```bash
python -m vrs_int export --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --output ./vrs-int-data/package-01
python -m vrs_int export --store ./vrs-int-data/memory --snapshot SNAPSHOT_2 --base SNAPSHOT_1 --output ./vrs-int-data/package-02-delta
python -m vrs_int install --store ./vrs-int-data/restored --package ./vrs-int-data/package-01
python -m vrs_int install --store ./vrs-int-data/restored --package ./vrs-int-data/package-02-delta
python -m vrs_int verify --store ./vrs-int-data/restored --snapshot SNAPSHOT_2
```

A delta requires its **exact immediate parent**, installed first. Arbitrary version
jumps or automatic merges of other people's collections are not supported.
A full export of `SNAPSHOT_2` works without that parent, but does not include the
parent's history. Objects shared between versions are stored once in a store.

`export` includes original tags, captions, and any source metadata in the records;
it does not include original images or encoder weights. It is **not a readable
JSONL tag export or an anonymized package**. Code-only publication should omit
personal packages too. SHA-256 checks detect content changes; they do not verify
the identity of a publisher. `install` verifies but never activates a version.

### Optional activation and rollback

The CLI always queries an explicit `--snapshot`; it does not resolve `CURRENT`.
Activation is an optional pointer for external tooling:

```bash
python -m vrs_int activate --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --expected-current none
python -m vrs_int activate --store ./vrs-int-data/memory --snapshot SNAPSHOT_2 --expected-current SNAPSHOT_1
python -m vrs_int activate --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --expected-current SNAPSHOT_2
```

The final command rolls the pointer back; for CLI queries also choose
`--snapshot SNAPSHOT_1`. `none` is valid only when no pointer exists. A changed
pointer causes a conflict rather than silently replacing another activation.
There is no garbage collector; old objects/history remain until managed separately.

## Storage, resources, and compatibility

```text
memory/
  objects/<sha256>.z       compressed tag dictionaries, records, vectors, audit data
  snapshots/<sha256>.json immutable manifest, contract, parent, counts
  CURRENT                 optional explicitly activated snapshot hash
```

Names are dictionary-encoded per bucket and objects use zlib compression by default.
An optional [lossless XZ copy](COMPRESSION.md) uses the new `vrs-int/3` format;
older readers do not support it. Typed
artist/character/series tags are protected even at frequency one; general noise
such as quality/resolution/watermark tags is filtered with reasons preserved.
Stored lookup tags are lowercase, spaces replace underscores, and duplicates are
removed. Raw records retain original spelling and `caption`; captions are not embedded.

`inspect` exposes snapshot metadata. There is currently no CLI raw-tag viewer or
human-readable JSONL export; keep your original manifest/sidecars for direct review.
Compressed objects remain reversible records, not a way to conceal `q/e` tags.

| Representation | Purpose and limits |
| --- | --- |
| `f32le` (default) | Preserves the produced FP32 vectors. 768 × 4 = 3,072 bytes/image before serialization, compression, and metadata |
| `i8` (opt-in at build) | Lossy vector quantization: 768 codes plus scale/image before overhead. Can alter neighbors/name candidates; not a quality-equivalent default |
| GPU search | Decodes both formats to FP32 in bounded batches; `i8` storage does not imply 4× lower search VRAM |

NF4 encoder weight quantization and `i8` stored-vector quantization are separate.
To compare codecs, build separate stores with `--codec f32le` and `--codec i8`;
do not append a different codec to an existing snapshot. PQ/ANN indexing is not implemented.

The CLI uses two CPU math threads, one PyTorch interop thread, and on supported
systems two CPU affinity slots and lower process priority. I/O, decoding, packing,
and hashing still use CPU. It requests a **20% PyTorch CUDA allocator limit**;
this is not a GPU reservation or a hard cap on all driver/library allocations.
Build/migration refuses to start below 20% free disk; that initial check does not
predict final free space. Old snapshots and full exports require additional storage.

Search streams bounded vector blocks to GPU and scans them exactly. A persistent disk
cache holds FP32 vectors and compressed voting documents; only winning documents are read.
The first preparation decodes source objects; unchanged files are verified and reused
across restarts. Row metadata and larger collections still increase RAM and latency.
See [cache limits and measurements](SEARCH.md). Million-image performance is not established.
No constant-VRAM/constant-RAM or accuracy guarantee follows from the small encoder.

The contract records encoder provenance hash, preprocessing, PIL backend, NF4,
pooling, dimension, codec, and tag policy. Changing these requires a separate
compatible base/re-encoding, not an automatic merge. Matching contracts do not
promise bit-identical output across all library/driver versions. Preserve package
versions alongside important memories (`python -m pip freeze`).

## Command reference

Use `python -m vrs_int --help` and `python -m vrs_int COMMAND --help` for exact syntax.

| Command | Purpose |
| --- | --- |
| `check-input --manifest FILE --images DIR` | Model-free input/path/hash checks |
| `build --manifest FILE --images DIR --encoder DIR --store DIR --gpu UUID` | Build; optional `--base HASH`, `--codec f32le/i8`, `--label TEXT`, `--buckets N` |
| `recommend --store DIR --snapshot HASH --encoder DIR --image FILE --gpu UUID --output FILE` | Query; optional `--ratings`, `--neighbors`, `--limit` |
| `inspect --store DIR --snapshot HASH` | Show manifest |
| `verify --store DIR --snapshot HASH` | Check all referenced data and counts |
| `export --store DIR --snapshot HASH --output DIR` | Full package; add `--base PARENT_HASH` for a delta |
| `install --store DIR --package DIR` | Verify and install package without activation |
| `activate --store DIR --snapshot HASH --expected-current HASH_OR_none` | Explicit pointer change / rollback |
| `update --store DIR --snapshot HASH --operations FILE --contract FILE --label TEXT` | Advanced pre-encoded operations with expected revisions |
| `migrate --memory DIR --corrected DIR --encoder DIR --store DIR` | Developer import of the earlier experiment layout; not an arbitrary dataset importer |

For normal additions/corrections use **`build --base`**, not `update`. Advanced
`update` reads UTF-8 JSONL operations: `upsert` requires `key`, a complete encoded
`entry`, and `expected_revision` (null only for a new key); `delete` requires an
existing key, its revision, and a reason. Revisions come from `store.revision(entry)`.
`--contract` is JSON containing the producer's full matching contract, not the
whole snapshot manifest. Conflicts abort before a new snapshot is published.
Use [store.py](store.py) and [storage tests](../tests/test_vrs_int.py) as the API reference.
`migrate` expects the internal layout in [import_snapshot.py](import_snapshot.py);
those experiment data are intentionally absent from the distribution.

## Troubleshooting

| Error or result | Action |
| --- | --- |
| `No module named vrs_int` | Run from repository root using the intended Python environment. |
| `CUDA required; no CPU fallback` | Check CUDA PyTorch, driver, and the complete selected UUID. Changing to CPU is not supported. |
| Missing Gemma4/PIL import or NF4 error | Verify both documented APIs and bitsandbytes compatibility; root TIPO requirements are insufficient. |
| `Encoder file or digest mismatch` | Restore the unchanged matching encoder directory. Do not edit provenance hashes to bypass the check. |
| `Query encoder incompatible` / `Incompatible update` | Use the original encoder/contract/codec, or build a separate base from original images. |
| `FileExistsError` during prepare/export/recommend/sidecar preparation | Pick a fresh destination. Failed preparation can leave a partial directory; preserve/inspect it and retry elsewhere. |
| HTTP range error in encoder preparation | Server/proxy must honor byte ranges. No full-language-model fallback is attempted. |
| Invalid path / missing image / duplicate ID | Use `/`-separated paths inside `--images`; one complete record per positive integer ID per batch. |
| Empty recommendations | Check `train` vectors, rating filters, and query self-exclusion. More tags without visual vectors do not add searchable examples. |
| Names absent or weak | Supply explicit typed name fields and relevant varied examples. Frequency or a vote score is not identity accuracy. |
| `Delta requires its exact immediate parent` | Export from the actual parent, or use a full package. Install that exact parent before a delta. |
| `Active snapshot changed` | Read `CURRENT`, then choose the intended exact hash; do not overwrite a concurrent change blindly. |
| `Keep at least 20% disk free` | Choose a volume with enough free space; account for temporary data and retained versions. |
| `Bucket too large` | Rebuild a separate base with more `--buckets`; object decompression is bounded at 64 MiB. |
| CUDA out of memory / growing RAM | Check other GPU processes, allocator allowance, and collection size. The current CLI has no encoder batch-size or ANN tuning option. |

## Development checks

```bash
python -m pip install numpy
python -m unittest discover -s tests -p "test_vrs_int*.py" -v
python scripts/check_code_only.py
```

Tests use fictional data and substitute encoders, with no GPU/model download.
They validate mechanics, not visual quality. The publication checker reads Git's
index; inspect staged content as well. Personal memory/packages/weights are excluded
from this repository; keep them out of pull requests and releases.

ComfyUI 네이티브 / Native build, recall and Gemma integration: [guide](COMFYUI_NATIVE.md).
