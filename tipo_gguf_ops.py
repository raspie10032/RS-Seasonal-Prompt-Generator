"""Packed GGUF weights with ComfyUI patching and device-local PyTorch operations.

Tensor metadata and delayed LoRA handling adapted from City96/ComfyUI-GGUF
(Apache-2.0); see vendor/gguf_torch/NOTICE.md. No llama.cpp runtime.
"""
import torch
import gguf
import comfy.ops
import comfy.utils
import comfy.lora
import comfy.model_patcher
import comfy.model_management

try:
    from .vendor.gguf_torch.dequant import dequantize, dequantize_functions
except ImportError:
    from vendor.gguf_torch.dequant import dequantize, dequantize_functions


class PackedWeight(torch.Tensor):
    @staticmethod
    def __new__(cls, data, qtype, logical_shape):
        return torch.Tensor._make_subclass(cls, data, False)

    def __init__(self, data, qtype, logical_shape):
        self.qtype = qtype
        self.logical_shape = torch.Size(logical_shape)
        self.patches = []

    @property
    def shape(self):
        return getattr(self, "logical_shape", super().shape)

    def to(self, *args, **kwargs):
        # Packed bytes must never be converted to the model's floating dtype.
        device, _, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        data = self.as_subclass(torch.Tensor).to(device=device, non_blocking=non_blocking,
                                               copy=kwargs.get("copy", False))
        result = PackedWeight(data, self.qtype, self.shape)
        result.patches = self.patches.copy()
        return result

    def detach(self):
        return self

    def clone(self, *args, **kwargs):
        result = PackedWeight(self.as_subclass(torch.Tensor).clone(*args, **kwargs), self.qtype, self.shape)
        result.patches = self.patches.copy()
        return result


def unpack(weight, device, dtype, rows=None):
    if not isinstance(weight, PackedWeight):
        if rows is not None:
            return weight[rows.to(weight.device)].to(device=device, dtype=dtype)
        return weight.to(device=device, dtype=dtype)
    if weight.qtype not in dequantize_functions:
        raise ValueError(f"No PyTorch GGUF decoder for {weight.qtype}; CPU fallback is disabled")
    data = weight.as_subclass(torch.Tensor)
    shape = weight.shape
    if rows is not None:
        # PLE has billions of values. Gather packed rows BEFORE dequantization.
        block, size = gguf.GGML_QUANT_SIZES[weight.qtype]
        if shape[-1] % block:
            raise ValueError("GGUF embedding rows are not block aligned")
        data = data.reshape(shape[0], shape[1] // block * size)
        # An offloaded PLE/embedding must not transfer its entire packed table
        # for a handful of tokens. Select on the storage device first.
        data = data[rows.to(data.device)].to(device)
        shape = (*rows.shape, shape[1])
    else:
        data = data.to(device)
    result = dequantize(data, weight.qtype, shape, dtype=torch.float32).to(dtype)
    if weight.patches:
        if rows is not None:
            raise ValueError("Embedding LoRA is not supported by this GGUF path")
        for patches, key in weight.patches:
            result = comfy.lora.calculate_weight(patches, result, key, intermediate_dtype=dtype)
    return result


class PackedModule:
    comfy_cast_weights = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ("weight", "bias"):
            key = prefix + name
            if key in state_dict:
                value = state_dict[key]
                self._parameters[name] = torch.nn.Parameter(value, requires_grad=False)
            elif name == "weight" or self.bias is not None:
                missing_keys.append(key)

    def _weight(self, x):
        w = unpack(self.weight, x.device, x.dtype)
        for function in self.weight_function:
            w = function(w)
        return w


class GGUFOps(comfy.ops.manual_cast):
    class Linear(PackedModule, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = in_features, out_features
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.weight_function, self.bias_function = [], []

        def forward(self, x):
            return torch.nn.functional.linear(x, self._weight(x),
                self.bias.to(x) if self.bias is not None else None)

    class Embedding(PackedModule, comfy.ops.manual_cast.Embedding):
        def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, **kwargs):
            torch.nn.Module.__init__(self)
            self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
            self.padding_idx, self.max_norm, self.norm_type = None, None, 2.0
            self.scale_grad_by_freq, self.sparse = False, False
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.weight_function, self.bias_function = [], []

        def forward(self, input, out_dtype=None):
            return unpack(self.weight, input.device, out_dtype or torch.float32, rows=input)

        def project(self, x, chunk=8192):
            # Tied output projection: bounded temporary allocation, never a
            # complete 262144 x 1536 float weight matrix.
            output = []
            for start in range(0, self.num_embeddings, chunk):
                rows = torch.arange(start, min(start + chunk, self.num_embeddings), device=x.device)
                w = unpack(self.weight, x.device, x.dtype, rows=rows)
                output.append(torch.nn.functional.linear(x, w))
            return torch.cat(output, dim=-1)


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False,
                               return_weight=False, force_cast=False):
        weight = comfy.utils.get_attr(self.model, key)
        if not isinstance(weight, PackedWeight):
            return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)
        result = weight.to(device=device_to)
        result.patches = [(self.patches[key], key)] if key in self.patches else []
        if return_weight:
            return result
        comfy.utils.set_attr_param(self.model, key, result)
        return result

    def load(self, *args, force_patch_weights=False, **kwargs):
        return super().load(*args, force_patch_weights=True, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if isinstance(p, PackedWeight):
                    p.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def clone(self, *args, **kwargs):
        result = super().clone(*args, **kwargs)
        result.__class__ = GGUFModelPatcher
        return result
