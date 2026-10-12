"""Pack local image.ext + image.ext.json pairs into a checked VRS-INT JSONL.

Uses the standard library only. Does not download, encode or alter images.
Run from a source checkout; paths in the output are relative to --pairs.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vrs_int.manifest import check_manifest


def make_manifest(pairs, output):
    pairs, output = Path(pairs).resolve(), Path(output).absolute()
    if not pairs.is_dir():
        raise ValueError("--pairs must be an existing image/sidecar directory")
    if output.suffix.lower() != ".jsonl":
        raise ValueError("--output must end in .jsonl")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".vrs-int-pairs-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for sidecar in sorted(pairs.rglob("*.json")):
                # Portrait.png.json describes Portrait.png; folders are kept.
                image = sidecar.with_suffix("").relative_to(pairs).as_posix()
                with sidecar.open(encoding="utf-8-sig") as source:
                    row = json.load(source)
                if not isinstance(row, dict):
                    raise ValueError(f"{sidecar}: expected one JSON object")
                if row.get("image", image) != image:
                    raise ValueError(f"{sidecar}: image field disagrees with sidecar filename")
                row["image"] = image
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        report = check_manifest(temporary, pairs)
        # Publish only a fully checked file, and never overwrite an existing one.
        os.link(temporary, output)
        return dict(report, manifest=str(output), image_root=str(pairs))
    finally:
        os.unlink(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_manifest(args.pairs, args.output), ensure_ascii=False, indent=2))
