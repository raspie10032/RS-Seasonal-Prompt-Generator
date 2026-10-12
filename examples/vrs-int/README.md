# Image/text pairs for VRS-INT

English | [한국어](README_KOR.md) | [VRS-INT guide](../../vrs_int/README.md) | [Project overview](../../README.md)

This walkthrough prepares **your own images and tags** for personal reference
memory. Examples contain fictional artist/character/series names and `g` ratings.
**No images or initial memory are included.** Replace the fictional metadata with
labels that describe your actual files; the examples are a schema demonstration.

One image corresponds to one record. Structured tags are used for recommendation;
an optional free-text caption is preserved for review. Captions are not embedded,
converted into tags, or used to retrain the encoder.

## Before you begin

- Use Python 3.11 or later and run commands from the repository root.
- Input preparation/checking uses only the standard library. Building/querying needs the [separate CUDA/encoder setup](../../vrs_int/README.md#setup).
- Keep private files outside the repository, or under the ignored `vrs-int-data/` folder.
- Choose either **JSONL directly** or **image-sidecar JSON** below. Both produce the same input format.

## Option A: write a JSONL manifest

Prepare your images under one root, for example:

```text
vrs-int-data/
  pairs/
    1001.png
    1002.jpg
    1003.webp
  batch-01.jsonl
  batch-02.jsonl
  query.png
```

The tracked [`batch-01.jsonl`](batch-01.jsonl) contains the first two records.
Copy it into your private folder and adapt the filenames/tags to your images.
[`batch-02.jsonl`](batch-02.jsonl) demonstrates a later correction and new record.

For readability, here is one record expanded over several lines. **Inside a
`.jsonl` file, put the entire object on one line**, with no outer array or commas
between records. Save UTF-8; UTF-8 BOM is also accepted by the reader.

```json
{
  "id": 1001,
  "image": "1001.png",
  "rating": "g",
  "tags_general": ["1girl", "solo", "blue_hair", "long_hair", "white_dress", "standing", "flower_field"],
  "tags_artist": ["example_artist_a"],
  "tags_character": ["example_character_a"],
  "tags_copyright": ["example_series_a"],
  "tags_meta": ["highres", "signature"],
  "caption": "A blue-haired character in a white dress stands in a flower field.",
  "split": "train"
}
```

### Field reference

| Field | Required? | Rules and use |
| --- | --- | --- |
| `id` | Yes | Positive JSON integer, unique within a batch. Stable key becomes `danbooru:1001`. A JSON string such as `"1001"` is invalid. |
| `image` | Yes in JSONL | File path relative to `--images`, using `/`, e.g. `spring/1001.png`. No absolute paths or escape outside the image root. |
| `rating` | Yes | One of `g` general, `s` sensitive, `q` questionable, `e` explicit. The program does not infer it. |
| `tags_general` | Optional field | General tags: clothing, color, pose, background, etc. Array of nonempty strings. |
| `tags_artist` | Optional field | Artist names as an array. Explicitly typed names are protected, including single occurrences. |
| `tags_character` | Optional field | Character names as an array; preserve them when correcting the record. |
| `tags_copyright` | Optional field | Series/work names as an array. This is the source tag category, not license metadata. |
| `tags_meta` | Optional field | Metadata tags; unnecessary file/quality/watermark tags may be excluded from recommendations with reasons recorded. |
| `tags` | Optional field | Untyped tag array, unioned with typed fields. Typed fields are strongly recommended for names. |
| `caption` | Optional | Any-language string, retained in the original record; not embedded or automatically parsed into tags. |
| `split` | Optional | `train` default, or `val`/`test`. Only `train` entries can be recommendation neighbors. All supplied images are encoded. |
| `image_sha256` | Optional | Exact SHA-256 of the image file; if supplied it must match. Otherwise computed. |
| Other JSON fields | Optional | Retained in original metadata; not automatically interpreted. Avoid private values in anything you intend to export. |

At least **one retained tag** across the supported tag arrays is required.
Caption-only records, empty tag arrays everywhere, or records whose tags are all
filtered are rejected. A string such as `"tags": "blue_hair long_hair"` is invalid:
use `"tags": ["blue_hair", "long_hair"]`. Danbooru's `tag_string*` fields are not
accepted as substitutes; convert them into these arrays yourself.

The current ID namespace is `danbooru:` even for locally assigned IDs. Keep IDs
stable and do not mix conflicting numeric IDs from different sources. Filenames
need not be the numeric ID; they simply must point to the intended file.

Tags are normalized for lookup: lowercase, underscores changed to spaces,
duplicates removed. Thus `blue_hair` becomes `blue hair`; the raw original string
is still preserved. Names in typed fields are protected by category. Names placed
only in `tags` cannot reliably appear in the artist/character/series result groups.

## Option B: put a JSON file beside each image

This avoids hand-packing multiple records onto JSONL lines. Use **the entire
image filename plus `.json`**, e.g. `1001.png.json`, not `1001.json`.

```text
vrs-int-data/pairs/
  1001.png
  1001.png.json
  1002.jpg
  1002.jpg.json
```

Save the expanded object above as `1001.png.json`. You may omit its `image` field:
the helper derives it from the sidecar filename. If present, it must match that
relative filename exactly. For `1002.jpg.json`, provide the second image's ID and
complete tags. The JSON object may span multiple lines in a sidecar.

Then run:

```bash
python examples/vrs-int/make_manifest.py --pairs ./vrs-int-data/pairs --output ./vrs-int-data/batch-01.jsonl
```

The helper recursively reads **all `*.json`** under `--pairs`, checks the result,
and publishes a new `.jsonl`. Keep unrelated JSON files out of that directory.
It never modifies images, calls a model, or downloads anything. Images without
sidecars are not included. The output must not already exist. The helper uses a
same-directory hard link to publish atomically; use a filesystem supporting hard
links (such as local NTFS/ext4/Btrfs) if a network/FAT volume rejects publication.

When preparing later batches, use a sidecar folder containing only the records
for that batch, or write a separate JSONL. Re-running against the full sidecar
collection resubmits every record and therefore re-encodes every image.

## Check the input

```bash
python -m vrs_int check-input --manifest ./vrs-int-data/batch-01.jsonl --images ./vrs-int-data/pairs
```

If you use the tracked example directly, change `--manifest` to
`./examples/vrs-int/batch-01.jsonl`; you must still supply actual `1001.png` and
`1002.jpg` files under your `--images` folder.

With the unmodified first batch, successful checking reports:

- `records`: 2; `ratings`: `g` = 2; `splits`: `train` = 2.
- `typed_unique_tags`: artist = 2, character = 1, copyright = 1.
- `removed_tag_occurrences`: 3 (`highres` twice, `signature` once).
- `input_manifest_sha256`: the hash of the exact JSONL bytes, so editing formatting also changes it.

This checks metadata, file existence, paths, and hashes. It does not decode the
images or verify semantic labels. An arbitrary file renamed `.png` may pass this
step and still fail image encoding.

## Build and query

Finish [encoder/CUDA setup](../../vrs_int/README.md#setup), select a GPU UUID, then
replace `GPU-YOUR-UUID` below with the complete UUID:

```bash
python -m vrs_int build --manifest ./vrs-int-data/batch-01.jsonl --images ./vrs-int-data/pairs --encoder ./vrs-int-models/gemma4-e2b-vision --store ./vrs-int-data/memory --gpu GPU-YOUR-UUID --label personal-01
```

Save the returned 64-character `snapshot` hash. Replace `SNAPSHOT_1` below with
that hash, and supply your new reference image as `query.png`:

```bash
python -m vrs_int verify --store ./vrs-int-data/memory --snapshot SNAPSHOT_1
python -m vrs_int recommend --store ./vrs-int-data/memory --snapshot SNAPSHOT_1 --encoder ./vrs-int-models/gemma4-e2b-vision --image ./vrs-int-data/query.png --gpu GPU-YOUR-UUID --ratings g,s --neighbors 32 --limit 20 --output ./vrs-int-data/recommendation-01.json
```

No tagging of `query.png` is necessary. An exact copy of a memory image excludes
itself, so use a separate reference image to exercise normal retrieval. Two images
demonstrate the workflow but cannot establish recommendation quality. Results are
reference candidates with neighbor votes, not verified author probabilities.

## Add images and correct tags

The supplied second batch shows:

| ID | Change | Images needed for this batch |
| --- | --- | --- |
| `1001` | Replace its complete record, adding `blue_sky` while retaining artist/character/series | `1001.png` |
| `1003` | Add a new image and different fictional names | `1003.webp` |
| `1002` | Unchanged; omit from the batch | Its original image is not needed for this update |

Copy/adapt [`batch-02.jsonl`](batch-02.jsonl), then:

```bash
python -m vrs_int check-input --manifest ./vrs-int-data/batch-02.jsonl --images ./vrs-int-data/pairs
python -m vrs_int build --manifest ./vrs-int-data/batch-02.jsonl --images ./vrs-int-data/pairs --encoder ./vrs-int-models/gemma4-e2b-vision --store ./vrs-int-data/memory --gpu GPU-YOUR-UUID --base SNAPSHOT_1 --label personal-02
```

The new snapshot has **three observations**, not four. Existing `1001` is revised,
not added as another vote. Save the new hash as `SNAPSHOT_2` and use it explicitly
for future queries. The older snapshot remains readable.

**Updates replace whole records.** Sending only `blue_sky` would remove the old
tags and typed names from the new record. Include its full current tag lists,
caption, rating, and split. The CLI currently re-encodes submitted images even
for tags-only changes, so retain the image for the corrected record.

You may use a different image folder per batch: `--images` only needs the current
batch's files. Stored relative paths are source metadata; recommendation does
not reopen those old files. Preserve originals elsewhere if you want to re-encode
under a changed encoder later.

## What to retain and what to publish

| Material | Retain for | Included in memory export? |
| --- | --- | --- |
| Original images | Re-encoding, direct visual review | No |
| JSONL/sidecars | Human-readable original tags, captions, editing | Raw records are preserved inside compressed memory; separate files are not copied |
| Memory directory | Queries, exact snapshot history, local updates | Export packages only the selected snapshot's needed objects; full history needs separate backups |
| Exact encoder directory | Compatible future builds/queries | No |
| New query image | Current query / later replay | Not automatically added to memory |

The public repository includes only code, docs, and these fictional templates.
Do not commit personal image pairs, generated JSONL, memory packages, query results,
or weights. Memory compression does not remove source tags or private metadata.
For transfer/rollback and error handling, see the [full VRS-INT guide](../../vrs_int/README.md).
