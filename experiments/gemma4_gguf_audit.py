"""Inspect Gemma 4 GGUF precision and create a byte-preserving vision-only pair.

Requires numpy and llama.cpp's gguf-py on PYTHONPATH. Model files and reports
belong outside the public repository. This does not change a model's defaults.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PLE = "per_layer_token_embd.weight"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest()


def inventory(path):
    import gguf
    reader = gguf.GGUFReader(path)
    types, prefixes, tensors = Counter(), Counter(), []
    for t in reader.tensors:
        kind = t.tensor_type.name
        types[kind] += t.n_bytes
        prefixes[t.name.split(".")[0]] += t.n_bytes
        tensors.append({"name": t.name, "type": kind, "shape": t.shape.tolist(),
                        "bytes": t.n_bytes, "sha256": tensor_hash(t)})
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
            "sha256": sha256(path), "types_bytes": dict(types),
            "prefix_bytes": dict(prefixes), "tensors": tensors}


def vision_only(source, output):
    """Copy vision/projector tensors exactly; reject unknown tensor families."""
    import gguf
    source, output = Path(source), Path(output)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise ValueError("Output and partial paths must not exist")
    reader = gguf.GGUFReader(source)
    if reader.get_field("general.architecture").contents() != "clip":
        raise ValueError("Expected a clip/mmproj GGUF")
    kind = reader.get_field("clip.vision.projector_type")
    if kind is None or kind.contents() != "gemma4v":
        raise ValueError("Expected the Gemma 4 vision projector")
    kept, removed = [], []
    for t in reader.tensors:
        if t.name.startswith("a."):
            removed.append(t)
        elif t.name.startswith(("v.", "mm.")):
            if t.tensor_type not in (gguf.GGMLQuantizationType.F16,
                                     gguf.GGMLQuantizationType.F32):
                raise ValueError(f"Vision tensor is not floating point: {t.name}")
            kept.append(t)
        else:
            raise ValueError(f"Unrecognized tensor; refusing to discard: {t.name}")
    if not kept or not any(t.name.startswith("mm.") for t in kept):
        raise ValueError("Missing vision or multimodal projection tensors")
    writer = gguf.GGUFWriter(partial, "clip", endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        if field.name.startswith("clip.audio."):
            continue
        value = False if field.name == "clip.has_audio_encoder" else field.contents()
        typ = field.types[0]
        sub_type = field.types[-1] if typ == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, typ, sub_type=sub_type)
    for t in kept:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in kept:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    check = gguf.GGUFReader(partial)
    expected = {t.name: (t.tensor_type.name, t.shape.tolist(), tensor_hash(t)) for t in kept}
    actual = {t.name: (t.tensor_type.name, t.shape.tolist(), tensor_hash(t)) for t in check.tensors}
    if actual != expected:
        raise ValueError("Vision tensor preservation check failed")
    del check
    partial.rename(output)
    return {"source": str(source.resolve()), "output": str(output.resolve()),
            "source_sha256": sha256(source), "output_sha256": sha256(output),
            "source_bytes": source.stat().st_size, "output_bytes": output.stat().st_size,
            "kept_tensors": len(kept), "removed_audio_tensors": len(removed),
            "removed_audio_bytes": sum(t.n_bytes for t in removed),
            "all_kept_tensor_payloads_identical": True}


def compare_ple(baseline, candidate):
    left, right = inventory(baseline), inventory(candidate)
    a, b = ({t["name"]: t for t in report["tensors"]} for report in (left, right))
    if a.keys() != b.keys() or PLE not in a:
        raise ValueError("Tensor inventory mismatch or missing PLE")
    changed = [name for name in a if a[name] != b[name]]
    if changed != [PLE] or b[PLE]["type"] != "Q4_K" or a[PLE]["shape"] != b[PLE]["shape"]:
        raise ValueError(f"Expected only PLE to change to Q4_K: {changed}")
    return {"baseline": left, "candidate": right, "changed_tensors": changed,
            "all_non_ple_tensors_identical": True,
            "file_bytes_saved": left["bytes"] - right["bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("inspect")
    scan.add_argument("source", type=Path)
    strip = sub.add_parser("vision-only")
    strip.add_argument("source", type=Path)
    strip.add_argument("output", type=Path)
    diff = sub.add_parser("compare-ple")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("candidate", type=Path)
    for p in (scan, strip, diff):
        p.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.resolve() in {
        p.resolve() for p in (getattr(args, "source", None), getattr(args, "output", None),
                             getattr(args, "baseline", None), getattr(args, "candidate", None)) if p
    }:
        raise ValueError("Report must be a new file, separate from model files")
    if args.command == "inspect":
        report = inventory(args.source)
    elif args.command == "vision-only":
        report = vision_only(args.source, args.output)
    else:
        report = compare_ple(args.baseline, args.candidate)
    args.report.write_text(json.dumps(report, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
