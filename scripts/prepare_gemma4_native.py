"""Repackage full HF Gemma 4 E2B safetensors for ComfyUI, without training.

Pure standard-library streaming copy: tensor payloads and precision are unchanged.
No model download, pickle execution, GGUF conversion, or adapter merging.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile


PREFIXES = {
    "model.language_model.": "model.",
    "model.vision_tower.": "vision_model.",
    "model.embed_vision.": "multi_modal_projector.",
    "model.audio_tower.": "audio_model.",
    "model.embed_audio.": "audio_projector.",
}
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def read_header(path):
    size = path.stat().st_size
    with path.open("rb") as f:
        raw_size = f.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Truncated safetensors file: {path.name}")
        length = struct.unpack("<Q", raw_size)[0]
        if length > min(100_000_000, size - 8):
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        header = json.loads(f.read(length))
    ranges = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        count = 1
        for dim in info["shape"]:
            if not isinstance(dim, int) or dim < 0:
                raise ValueError(f"Invalid shape for {name}")
            count *= dim
        if info["dtype"] not in DTYPE_BYTES:
            raise ValueError("Use full, unquantized HF safetensors; unsupported dtype: " + info["dtype"])
        if not 0 <= start <= end <= size - 8 - length or end - start != count * DTYPE_BYTES[info["dtype"]]:
            raise ValueError(f"Invalid tensor range: {name}")
        ranges.append((start, end))
    position = 0
    for start, end in sorted(ranges):
        if start != position:
            raise ValueError("Overlapping tensor ranges or holes in safetensors")
        position = end
    if position != size - 8 - length:
        raise ValueError("Unexpected trailing safetensors data")
    return header, length + 8


def native_name(name):
    for source, target in PREFIXES.items():
        if name.startswith(source):
            return target + name[len(source):]
    raise ValueError(f"Unsupported HF tensor name: {name}; use a full Gemma 4 E2B checkpoint, not an adapter or GGUF.")


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    if source == output.parent or source in output.parents:
        raise ValueError("Choose an output outside the source model directory")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    if config.get("model_type") != "gemma4" or (text.get("hidden_size"), text.get("num_hidden_layers"), text.get("vocab_size")) != (1536, 35, 262144):
        raise ValueError("This converter supports full Gemma 4 E2B checkpoints only")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("Quantized HF checkpoints are not supported; use full BF16/FP16 weights")
    tokenizer = (source / "tokenizer.json").read_bytes()
    if "model" not in json.loads(tokenizer):
        raise ValueError("Invalid tokenizer.json")
    index_path = source / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = ["model.safetensors"]
    tensors, original_names, manifest = {}, set(), []
    for filename in files:
        path = (source / filename).resolve()
        if path.parent != source:
            raise ValueError("Shards must be directly inside the source directory")
        header, offset = read_header(path)
        manifest.append({"file": filename, "bytes": path.stat().st_size})
        for name, info in header.items():
            if name == "__metadata__":
                continue
            target = native_name(name)
            if target in tensors or (weight_map is not None and weight_map.get(name) != filename):
                raise ValueError(f"Duplicate or incorrectly indexed tensor: {name}")
            original_names.add(name)
            tensors[target] = (info, path, offset)
    if weight_map is not None and original_names != set(weight_map):
        raise ValueError("Shard index and actual tensors disagree")
    required = {"model.embed_tokens.weight", "model.embed_tokens_per_layer.weight",
                "multi_modal_projector.embedding_projection.weight"}
    if not required <= tensors.keys() or not any(k.startswith("vision_model.") for k in tensors):
        raise ValueError("Missing language, PLE, vision, or projector weights")
    layers = {int(k.split(".")[2]) for k in tensors if k.startswith("model.layers.")}
    if layers != set(range(35)):
        raise ValueError("Incomplete 35-layer text model")
    result_header = {"__metadata__": {"format": "pt", "rs_conversion": "hf-gemma4-e2b-to-comfy-v1"}}
    position = 0
    for name, (info, _, _) in tensors.items():
        length = info["data_offsets"][1] - info["data_offsets"][0]
        result_header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [position, position + length]}
        position += length
    result_header["tokenizer_json"] = {"dtype": "U8", "shape": [len(tokenizer)], "data_offsets": [position, position + len(tokenizer)]}
    header_bytes = json.dumps(result_header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".part", dir=output.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out:
            def write(data):
                out.write(data)
                digest.update(data)
            write(struct.pack("<Q", len(header_bytes)))
            write(header_bytes)
            for info, path, offset in tensors.values():
                start, end = info["data_offsets"]
                remaining = end - start
                with path.open("rb") as inp:
                    inp.seek(offset + start)
                    while remaining:
                        block = inp.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise IOError(f"Source changed or truncated while copying: {path.name}")
                        write(block)
                        remaining -= len(block)
            write(tokenizer)
            out.flush()
            os.fsync(out.fileno())
        # Atomic publication without replacing an existing destination, even
        # if another converter created it after the initial existence check.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return {"output": str(output), "bytes": output.stat().st_size,
            "sha256": digest.hexdigest(), "source": str(source), "source_files": manifest,
            "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest(),
            "tensor_count": len(tensors), "precision_changed": False,
            "trained": False, "source_training_status": "not inferred by converter",
            "note": "All source tensor payloads, including vision and audio, preserved byte for byte. This conversion performs no quantization or training; source weights may already be fine-tuned."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Full HF Gemma 4 E2B folder containing tokenizer.json, config.json and safetensors")
    parser.add_argument("--output", type=Path, required=True, help="New safetensors file outside source, normally in ComfyUI/models/text_encoders")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
