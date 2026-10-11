"""Shared, model-free checks for local image/tag pairs (UTF-8 JSONL)."""
from collections import Counter
import json
from pathlib import Path, PureWindowsPath

from .import_snapshot import file_digest
from .tag_policy import FIELDS, canonical, filter_tags, typed_tags


def read_pairs(manifest, images):
    """Yield (original record, resolved image, SHA-256); never modify inputs."""
    images = Path(images).resolve()
    if not images.is_dir():
        raise ValueError("--images must be an existing image root directory")
    seen = set()
    with Path(manifest).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each line must be a JSON object")
                json.dumps(row, allow_nan=False)
                post_id = row.get("id")
                if type(post_id) is not int or post_id <= 0:
                    raise ValueError("positive integer Danbooru id required")
                if post_id in seen:
                    raise ValueError(f"Duplicate id {post_id}; use one complete record per batch")
                seen.add(post_id)
                if row.get("rating") not in ("g", "s", "q", "e"):
                    raise ValueError("explicit g/s/q/e rating required")
                if row.get("split", "train") not in ("train", "val", "test"):
                    raise ValueError("invalid split")
                for field in ("tags", *FIELDS.values()):
                    tags = row.get(field, [])
                    if not isinstance(tags, list) or any(not isinstance(tag, str) or not canonical(tag) for tag in tags):
                        raise ValueError(f"{field} must be a list of nonempty strings")
                if "caption" in row and not isinstance(row["caption"], str):
                    raise ValueError("caption must be a string (audit only, not embedded)")
                if not filter_tags(row)[0]:
                    raise ValueError("at least one retained tag required; caption alone is not sufficient")
                relative = row.get("image")
                if not isinstance(relative, str) or not relative.strip():
                    raise ValueError("image must be a nonempty relative path")
                path = Path(relative)
                if path.is_absolute() or PureWindowsPath(relative).anchor or "\\" in relative:
                    raise ValueError("image must be relative to --images, using '/' separators")
                image = (images / path).resolve()
                if images not in image.parents:
                    raise ValueError("Image path escapes --images")
                if not image.is_file():
                    raise ValueError(f"missing image file: {relative}")
                sha = file_digest(image)
                if row.get("image_sha256", sha) != sha:
                    raise ValueError("Source image hash mismatch")
            except (ValueError, OSError) as exc:
                raise ValueError(f"Line {number}: {exc}") from exc
            yield row, image, sha
    if not seen:
        raise ValueError("Empty input manifest")


def check_manifest(manifest, images):
    """Read files and metadata, without decoding pixels or importing a model."""
    before = file_digest(manifest)
    ratings, splits = Counter(), Counter()
    categories = {kind: set() for kind in FIELDS}
    count = removed_count = 0
    for row, _, _ in read_pairs(manifest, images):
        count += 1
        ratings[row["rating"]] += 1
        splits[row.get("split", "train")] += 1
        removed_count += len(filter_tags(row)[1])
        for kind, tags in typed_tags(row).items():
            categories[kind].update(tags)
    if file_digest(manifest) != before:
        raise ValueError("Input manifest changed during check")
    return {"records": count, "ratings": dict(ratings), "splits": dict(splits),
            "typed_unique_tags": {kind: len(tags) for kind, tags in categories.items()},
            "removed_tag_occurrences": removed_count, "input_manifest_sha256": before,
            "scope": "Metadata, paths and file hashes only; no image decoding, encoder or GPU."}
