"""Small device preflight using ComfyUI's existing PyTorch, no CUDA imports.

Passing this probe is necessary, not sufficient, for running a full model.
No fallback/retry or global device settings are changed here.
"""
from functools import lru_cache


def require_accelerator(use_cpu=False):
    import comfy.model_management as mm
    if type(use_cpu) is not bool:
        raise ValueError("use_cpu must be boolean")
    if use_cpu:
        import torch
        device = torch.device("cpu")
    else:
        device = mm.get_torch_device()
    if device.type == "meta" or (device.type == "cpu" and not use_cpu):
        raise RuntimeError("VRS needs a ComfyUI PyTorch accelerator; CPU fallback is disabled. "
                           "Enable the VRS use_cpu option to consent to CPU inference, or configure a GPU in ComfyUI.")
    mm.throw_exception_if_processing_interrupted()
    _probe(str(device))
    return device


@lru_cache(maxsize=8)
def _probe(device):
    import torch
    import torch.nn.functional as F
    try:
        with torch.inference_mode():
            x = torch.ones((1, 1, 2, 16), device=device, dtype=torch.float32)
            w = torch.eye(16, device=device, dtype=torch.float16)
            y = F.linear(x, w.float())
            y = F.layer_norm(y, (16,))
            y = F.scaled_dot_product_attention(y, y, y)
            if not torch.isfinite(y).all().item():
                raise RuntimeError("Nonfinite basic PyTorch output")
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        raise RuntimeError(f"ComfyUI PyTorch device {device} cannot run the VRS preflight: {exc}. "
                           "No CPU retry or package installation was attempted.") from exc
