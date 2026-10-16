"""Gemma 4 E2B GGUF -> ComfyUI native CLIP, with packed text weights."""
from pathlib import Path
from types import SimpleNamespace


def tensor_name(name, mapping):
    if name.startswith("vision_model.encoder."):
        name = name.replace("vision_model.encoder.", "vision_model.model.", 1)
    elif name.startswith("vision_model."):
        name = name.replace("vision_model.", "model.vision_tower.", 1)
    elif name.startswith("multi_modal_projector."):
        name = name.replace("multi_modal_projector.", "model.embed_vision.", 1)
    if name.endswith(("layer_scalar", "position_embedding_table", "std_bias", "std_scale")):
        name += ".weight"
    name = name.replace(".linear.weight", ".weight")
    return mapping.get_name(name, try_suffixes=(".weight", ".bias", ".input_max", ".input_min", ".output_max", ".output_min"))


def load_clip(text_path, tokenizer_path, vision_path=None):
    import numpy as np
    import torch
    import gguf
    import comfy.sd
    import comfy.ops
    from comfy.text_encoders import gemma4
    try:
        from .tipo_gguf_ops import GGUFOps, PackedWeight, GGUFModelPatcher
    except ImportError:
        from tipo_gguf_ops import GGUFOps, PackedWeight, GGUFModelPatcher

    class SchemaOps(comfy.ops.manual_cast):
        # Recent ComfyUI lazily constructs Linear weights as None under aimdo,
        # even on meta. Use PyTorch's meta-only shapes for the complete schema.
        Linear = torch.nn.Linear

    class GemmaQ4(gemma4.Gemma4Base):
        tokenizer = gemma4.Gemma4_E2B.tokenizer

        def __init__(self, config_dict, dtype, device, operations):
            super().__init__()
            config = gemma4.Gemma4_E2B_Config(**config_dict)
            self.num_layers, self.dtype = config.num_hidden_layers, dtype
            self.model = gemma4.Gemma4Transformer(config, device=device, dtype=dtype, ops=operations)
            if vision_path:
                vision_ops = SchemaOps if device == "meta" else comfy.ops.manual_cast
                self.multi_modal_projector = gemma4.Gemma4MultiModalProjector(config, dtype=torch.float16, device=device, ops=vision_ops)
                self.vision_model = gemma4.Gemma4VisionEncoder(config.vision_config, dtype=torch.float16, device=device, ops=vision_ops)

        def logits(self, x):
            logits = self.model.embed_tokens.project(x[:, -1:])
            cap = self.model.config.final_logit_softcapping
            if cap:
                logits = cap * torch.tanh(logits / cap)
            if self.model.config.suppress_tokens:
                logits.index_fill_(-1, self.model._suppress_tokens, torch.finfo(logits.dtype).min)
            return logits

    # Get the authoritative ComfyUI schema without allocating model weights.
    schema = GemmaQ4({}, torch.float16, "meta", SchemaOps).state_dict()
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA4, 35)
    vision_mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, 16)
    readers = [gguf.GGUFReader(str(text_path))]
    field = readers[0].get_field("general.architecture")
    if field is None or field.contents() != "gemma4":
        raise ValueError("Expected a Gemma 4 E2B GGUF")
    if vision_path:
        readers.append(gguf.GGUFReader(str(vision_path)))
    inventories = [{t.name: t for t in r.tensors} for r in readers]
    state = {}
    missing = []
    for key, expected in schema.items():
        is_text = key.startswith("model.")
        name = tensor_name(key, mapping if is_text else vision_mapping)
        tensor = inventories[0 if is_text else 1].get(name)
        if tensor is None:
            missing.append((key, name))
            continue
        shape = tuple(int(d) for d in tensor.shape[::-1])
        data = torch.from_numpy(tensor.data.copy())
        if key == "vision_model.patch_embedder.input_proj.weight":
            data = data.permute(0, 2, 3, 1).reshape(expected.shape).contiguous()
            shape = tuple(expected.shape)
        if shape != tuple(expected.shape):
            # GGUF scalar tensors are represented as a one-element vector.
            if expected.numel() == 1 and np.prod(shape) == 1:
                shape = tuple(expected.shape)
            else:
                raise ValueError(f"GGUF shape mismatch: {name} {shape} != {tuple(expected.shape)}")
        if tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            state[key] = data.reshape(shape)
        elif is_text and expected.ndim == 2:
            state[key] = PackedWeight(data, tensor.tensor_type, shape)
        else:
            raise ValueError(f"Unsupported GGUF quantization for {name}: {tensor.tensor_type}")
    if missing:
        raise ValueError(f"Incomplete Gemma 4 E2B GGUF mapping: {missing[:12]}")
    target = SimpleNamespace(params={}, clip=gemma4.gemma4_te(model_class=GemmaQ4), tokenizer=GemmaQ4.tokenizer)
    clip = comfy.sd.CLIP(target, tokenizer_data={"tokenizer_json": Path(tokenizer_path).read_bytes()},
        model_options={"custom_operations": GGUFOps, "dtype": torch.float16,
                       "initial_device": torch.device("cpu")}, disable_dynamic=True)
    absent, extra = clip.load_sd(state)
    if absent or extra:
        raise ValueError(f"GGUF load incomplete: missing={absent}, extra={extra}")
    clip.patcher.__class__ = GGUFModelPatcher
    return clip
