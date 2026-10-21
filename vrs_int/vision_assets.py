"""VRS model selection. No model imports, downloads or private data at import."""
import json
import logging
from pathlib import Path
import struct

AUTO_VISION = "(auto: Gemma FP16 vision)"
VISION_HELP = (
    "Auto: download standalone FP16 vision (~319 MiB), or reuse a verified local TIPO mmproj if gguf is installed. "
    "No language model, tokenizer, llama.cpp or GGUF parser needed for standalone vision. "
    "Requires ComfyUI Gemma 4 vision and supported PyTorch operations; CPU inference needs explicit use_cpu consent. "
    "NVIDIA/CPU tested; AMD/Intel/Apple full-model execution unverified. "
    "첫 실행 시 비전 파일만 자동 준비. 개인 이미지·메타데이터·경험은 로컬 처리. "
    "Downloads are model-only GET requests. RS_MODELS_OFFLINE=1 disables downloads. "
    "Changing encoder weights/runtime may require a separate memory rebuild."
)

# Public vision weights only. No personal experience, inputs or account token.
VISION = {
    "repo": "raspie/vrs-int-gemma4-vision-fp16", "revision": "3fa835feca0093e89f44d5215c7c7633276ce0c7",
    "files": {"vrs-gemma4-e2b-vision-fp16.safetensors":
              (334811920, "fdc1b6fedb100e682eebe5ec54193929877a518f5db5aa9456f3eba637306216")},
}


def has_native_vision(path):
    """Read only the safetensors header; exclude unrelated encoders/adapters."""
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                return False
            length = struct.unpack("<Q", raw)[0]
            if not 0 < length <= min(16 * 1024 * 1024, Path(path).stat().st_size - 8):
                return False
            header = json.loads(stream.read(length))
        return isinstance(header, dict) and any(k.startswith("vision_model.") for k in header)
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def vision_choices():
    import folder_paths
    choices = [AUTO_VISION]
    for name in folder_paths.get_filename_list("text_encoders"):
        if name.lower().endswith(".safetensors"):
            path = folder_paths.get_full_path("text_encoders", name)
            if path and has_native_vision(path):
                choices.append(name)
    return choices


def local_vision_path(name):
    """Validate a manual selection without starting GPU work or networking."""
    import folder_paths
    path = folder_paths.get_full_path_or_raise("text_encoders", name)
    if not has_native_vision(path):
        raise ValueError("VRS needs native Gemma 4 E2B vision_model.* safetensors. "
                         "This file is an unrelated model or legacy HF extraction. "
                         "Select '(auto: Gemma FP16 vision)' to prepare the compatible vision automatically.")
    return path


def resolve_vision_path(name, interrupt=None, *, use_cpu=False):
    if name != AUTO_VISION:
        return local_vision_path(name)
    # Check runtime and GPU BEFORE a large download. Do not install packages or
    # change ComfyUI's torch in the running server.
    import comfy.model_management as mm
    try:
        from comfy.text_encoders import gemma4
    except ImportError as exc:
        raise RuntimeError("Gemma 4 vision is unavailable. Update ComfyUI and install its dependencies using ComfyUI's Python, then restart.") from exc
    from .torch_device import require_accelerator
    require_accelerator(use_cpu=use_cpu)
    if not hasattr(gemma4, "Gemma4VisionEncoder"):
        raise RuntimeError("Update ComfyUI for Gemma 4 vision support, then restart")
    import folder_paths
    import comfy.utils
    try:
        from ..tipo_native_lora import ensure_files
    except ImportError:
        from tipo_native_lora import ensure_files
    progress = comfy.utils.ProgressBar(1)
    def update(current, total):
        mm.throw_exception_if_processing_interrupted()
        if interrupt:
            interrupt()
        progress.update_absolute(current, total)
    directory = Path(folder_paths.models_dir) / "text_encoders" / "rs_vrs_vision"
    target = directory / next(iter(VISION["files"]))
    if not target.exists() and not target.is_symlink():
        cached = cached_tipo_vision(folder_paths.models_dir, update)
        if cached:
            return cached
    directory = ensure_files(VISION, directory, update)
    return directory / next(iter(VISION["files"]))


def cached_tipo_vision(models_dir, progress):
    """Avoid a new download when a usable pinned mmproj is already on disk.

    GGUF is optional here: absence/corruption never blocks standalone vision.
    A corrupt file is preserved. An explicitly installed standalone file is
    always verified first instead of being silently replaced by this fallback.
    """
    try:
        from ..tipo_gguf import VISION as TIPO_VISION
        from ..tipo_native_lora import _check_file
    except ImportError:
        from tipo_gguf import VISION as TIPO_VISION
        from tipo_native_lora import _check_file
    name, expected = next(iter(TIPO_VISION["files"].items()))
    path = Path(models_dir) / "text_encoders" / "rs_gemma_q4" / name
    if not path.exists() or path.is_symlink():
        return None
    try:
        import gguf  # noqa: F401
    except ImportError:
        return None
    progress(0, expected[0])
    try:
        _check_file(path, expected)
    except (OSError, ValueError):
        logging.warning("Cached TIPO mmproj failed verification; preserving it and preparing standalone VRS vision")
        return None
    progress(expected[0], expected[0])
    return path


def gguf_vision_state(path, schema):
    """Map just the FP16 vision tower; ignore audio and language projection."""
    import gguf
    import torch
    try:
        from ..tipo_gguf_runtime import tensor_name
    except ImportError:
        from tipo_gguf_runtime import tensor_name
    reader = gguf.GGUFReader(str(path))
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, 16)
    inventory = {t.name: t for t in reader.tensors}
    state = {}
    for key, expected in schema.items():
        name = tensor_name("vision_model." + key, mapping)
        tensor = inventory.get(name)
        if tensor is None:
            raise ValueError(f"Incomplete Gemma 4 E2B FP16 vision: missing {key} ({name})")
        if tensor.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            raise ValueError(f"VRS requires unquantized FP16/F32 vision: {name}")
        data = torch.from_numpy(tensor.data.copy())
        shape = tuple(int(d) for d in tensor.shape[::-1])
        if key == "patch_embedder.input_proj.weight":
            if len(shape) != 4 or shape != (768, 3, 16, 16):
                raise ValueError(f"Unexpected Gemma 4 E2B patch embedding: {shape}")
            data = data.permute(0, 2, 3, 1).reshape(expected.shape).contiguous()
            shape = tuple(data.shape)
        if shape != tuple(expected.shape):
            if expected.numel() != 1 or data.numel() != 1:
                raise ValueError(f"VRS vision shape mismatch: {key}: {shape} != {tuple(expected.shape)}")
        state[key] = data.reshape(expected.shape).to(torch.float16).contiguous()
    return state
