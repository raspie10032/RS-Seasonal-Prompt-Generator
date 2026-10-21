"""Extract native Gemma 4 E2B vision tensors as FP16, without GPU or training.

Input: a ComfyUI-format Gemma safetensors file. Output: vision_model.* only.
No tokenizer, language/audio/projector weights, image data or local paths are
embedded. Requires PyTorch and safetensors (already provided by ComfyUI).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_shapes():
    """The fixed 16-layer E2B tower; fail on other Gemma sizes/incomplete files."""
    shapes = {"patch_embedder.input_proj.weight": (768, 768),
              "patch_embedder.position_embedding_table": (2, 10240, 768)}
    for i in range(16):
        prefix = f"encoder.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            shapes[prefix + name + ".weight"] = (768,)
        for name in ("q_norm", "k_norm"):
            shapes[prefix + "self_attn." + name + ".weight"] = (64,)
        projections = {"self_attn." + name + "_proj": (768, 768) for name in ("q", "k", "v", "o")}
        projections.update({"mlp.gate_proj": (3072, 768), "mlp.up_proj": (3072, 768),
                            "mlp.down_proj": (768, 3072)})
        for name, shape in projections.items():
            shapes[prefix + name + ".linear.weight"] = shape
            for bound in ("input_min", "input_max", "output_min", "output_max"):
                shapes[prefix + name + "." + bound] = ()
    return {"vision_model." + key: shape for key, shape in shapes.items()}


def prepare(source, output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace {output}")
    def signature():
        st = source.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    before = signature()
    schema, tensors = expected_shapes(), {}
    with safe_open(str(source), framework="pt", device="cpu") as stream:
        keys = {k for k in stream.keys() if k.startswith("vision_model.")}
        if keys != set(schema):
            raise ValueError("Expected complete native Gemma 4 E2B vision_model.* tensors; not an adapter, HF or GGUF file")
        for key in sorted(keys):
            tensor = stream.get_tensor(key)
            if tuple(tensor.shape) != schema[key] or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"Unexpected vision tensor shape/dtype: {key}")
            tensor = tensor.to(torch.float16).contiguous()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite FP16 vision tensor: {key}")
            tensors[key] = tensor
    if signature() != before:
        raise ValueError("Source changed during extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".part", dir=output.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata={"format": "pt", "rs_conversion": "native-gemma4-e2b-vision-fp16-v1"})
        with open(temporary, "rb+") as stream:
            os.fsync(stream.fileno())
        if signature() != before:
            raise ValueError("Source changed during extraction")
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return {"file": output.name, "bytes": output.stat().st_size, "sha256": sha256(output),
            "tensor_count": len(tensors), "weights_bytes": sum(t.numel() * 2 for t in tensors.values()),
            "dtype": "float16", "trained": False, "gpu_used": False,
            "contents": "vision_model.* only; no language/audio/projector/tokenizer/personal data"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
