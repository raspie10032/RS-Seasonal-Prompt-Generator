"""Export a separate Gemma 4 text checkpoint without loading weights into RAM.

The default removes only unused modalities. --drop-ple is a destructive *model
ablation* in the output copy, not a claim that PLE is unnecessary. Never edits
the source checkpoint, tokenizer vocabulary, or VRS image encoder.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct


PLE_PARTS = {"embed_tokens_per_layer", "per_layer_model_projection",
             "per_layer_projection_norm", "per_layer_input_gate",
             "per_layer_projection", "post_per_layer_input_norm"}
SUPPORT_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                 "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
                 "generation_config.json", "README.md", "LICENSE", "LICENSE.txt")


def read_header(path):
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated Safetensors: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > min(path.stat().st_size - 8, 64 * 1024**2):
            raise ValueError(f"Invalid Safetensors header: {path}")
        return json.loads(stream.read(size)), size + 8


def text_key(name, drop_ple=False):
    if name.startswith("model.language_model."):
        name = "model." + name.removeprefix("model.language_model.")
    elif name != "lm_head.weight":
        return None
    if drop_ple and PLE_PARTS.intersection(name.split(".")):
        return None
    return name


def export(base, output, *, drop_ple=False):
    base, output = Path(base), Path(output)
    config = json.loads((base / "config.json").read_text())
    if config.get("model_type") != "gemma4" or "text_config" not in config:
        raise ValueError("Expected a multimodal Gemma 4 Safetensors checkpoint")
    if output.exists() or output.resolve().is_relative_to(base.resolve()):
        raise ValueError("Output must be a new directory outside the base checkpoint")
    entries, discarded, names = [], [], set()
    for path in sorted(base.glob("*.safetensors")):
        header, start = read_header(path)
        for name, value in header.items():
            if name == "__metadata__":
                continue
            lo, hi = value["data_offsets"]
            if lo < 0 or hi < lo or start + hi > path.stat().st_size:
                raise ValueError(f"Invalid tensor bounds: {name}")
            new_name = text_key(name, drop_ple)
            record = {"source_file": path.name, "source_key": name, "output_key": new_name,
                      "shape": value["shape"], "dtype": value["dtype"],
                      "bytes": hi - lo, "parameters": math.prod(value["shape"])}
            if new_name is None:
                discarded.append(record)
            else:
                if new_name in names:
                    raise ValueError(f"Duplicate tensor: {new_name}")
                names.add(new_name)
                entries.append((record, path, start + lo))
    if not entries or "model.embed_tokens.weight" not in names:
        raise ValueError("Missing text embedding weights")
    if not drop_ple and "model.embed_tokens_per_layer.weight" not in names:
        raise ValueError("Expected Gemma 4 E2B/E4B PLE weights")
    total = sum(r[0]["bytes"] for r in entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total + 256 * 1024**2:
        raise ValueError("Insufficient free disk space")
    output.mkdir()
    offset, out_header = 0, {"__metadata__": {"format": "pt"}}
    for record, _, _ in entries:
        out_header[record["output_key"]] = {"dtype": record["dtype"], "shape": record["shape"],
                                          "data_offsets": [offset, offset + record["bytes"]]}
        offset += record["bytes"]
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    file_hash = hashlib.sha256()
    temporary = output / "model.safetensors.partial"
    with ExitStack() as stack, temporary.open("xb") as dest:
        streams = {p: stack.enter_context(p.open("rb")) for _, p, _ in entries}
        prefix = struct.pack("<Q", len(blob)) + blob
        dest.write(prefix)
        file_hash.update(prefix)
        for record, path, start in entries:
            stream = streams[path]
            stream.seek(start)
            remaining, sha = record["bytes"], hashlib.sha256()
            while remaining:
                chunk = stream.read(min(remaining, 8 * 1024**2))
                if not chunk:
                    raise ValueError(f"Truncated tensor: {record['source_key']}")
                dest.write(chunk)
                file_hash.update(chunk)
                sha.update(chunk)
                remaining -= len(chunk)
            record["source_copied_payload_sha256"] = sha.hexdigest()
    temporary.rename(output / "model.safetensors")
    text_config = dict(config["text_config"])
    text_config.update(architectures=["Gemma4ForCausalLM"], model_type="gemma4_text")
    if drop_ple:
        text_config["hidden_size_per_layer_input"] = 0
    (output / "config.json").write_text(json.dumps(text_config, indent=2))
    for name in SUPPORT_FILES:
        if (base / name).is_file():
            shutil.copyfile(base / name, output / name)
    report = {"source": str(base.resolve()), "output": str(output.resolve()),
              "drop_ple": drop_ple, "kept": [r[0] for r in entries], "removed": discarded,
              "source_weight_bytes": total + sum(r["bytes"] for r in discarded),
              "output_weight_bytes": total, "output_parameters": sum(r[0]["parameters"] for r in entries),
              "output_file_sha256": file_hash.hexdigest(),
              "scope": "Byte-preserving text export; PLE deletion requires separate capability evaluation."}
    (output / "slimming-manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-ple", action="store_true")
    args = parser.parse_args()
    report = export(args.base, args.output, drop_ple=args.drop_ple)
    print(json.dumps({k: v for k, v in report.items() if k not in ("kept", "removed")}, indent=2))


if __name__ == "__main__":
    main()
