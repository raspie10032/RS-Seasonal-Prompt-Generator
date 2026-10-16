"""Explicit first-use download and selection, without importing GPU packages."""
import logging
from pathlib import Path
from functools import lru_cache
import threading

try:
    from .tipo_native_lora import ensure_files
except ImportError:
    from tipo_native_lora import ensure_files

SOURCES = ("connected_clip", "heretic_ara_q4")
TEXT = {"repo": "mradermacher/gemma-4-E2B-it-heretic-ara-GGUF",
        "revision": "c02d14dccce74f69566f8b815e648cc9eba243f7", "files": {
            "gemma-4-E2B-it-heretic-ara.Q4_K_M.gguf": (3427874048, "6a7b235e0a3dabd7eb410f1842bad8bb0664e0cff835aa098dee817737bd904a")}}
TOKENIZER = {"repo": "p-e-w/gemma-4-E2B-it-heretic-ara",
        "revision": "c9a1d4c031981f14d86eeb0c7d87de7fafd34513", "files": {
            "tokenizer.json": (32169626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f")}}
VISION = {"repo": "unsloth/gemma-4-E2B-it-GGUF",
        "revision": "0314792d7f1f7e229411f620751375812bb9faf2", "files": {
            "mmproj-F16.gguf": (985654080, "140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa")}}
WARNING = (
    "heretic_ara_q4: first execution downloads a 3.19 GiB text GGUF + 30.7 MiB tokenizer; "
    "IMAGE adds a 0.92 GiB FP16 mmproj. ComfyUI PyTorch execution, no llama.cpp. "
    "첫 실행 시 베이스 모델 다운로드. LoRA는 별도. 고사양 GPU 권장; 결과 품질은 보장하지 않습니다. "
    "Validated on 16GB NVIDIA GPUs; download size is not VRAM usage. connected_clip keeps your existing Load CLIP workflow."
)
_LOCK = threading.Lock()


def _signature(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected a model file: {path}")
    return str(path), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=1)
def _load(text, tokenizer, vision):
    try:
        from .tipo_gguf_runtime import load_clip
    except ImportError:
        from tipo_gguf_runtime import load_clip
    return load_clip(text[0], tokenizer[0], vision[0] if vision else None)


def resolve_clip(clip, source="connected_clip", image=None, gguf_path="", mmproj_path="", tokenizer_path=""):
    if source not in SOURCES:
        raise ValueError("Unknown Gemma model_source")
    if source == "connected_clip":
        if clip is None:
            raise ValueError("Connect a Gemma 4 Load CLIP or select model_source=heretic_ara_q4")
        return clip
    if clip is not None:
        raise ValueError("Disconnect CLIP for heretic_ara_q4, or choose connected_clip")
    try:
        import gguf  # noqa: F401 -- fail before large downloads
    except ImportError as exc:
        raise RuntimeError("Q4 requires Python 3.10+ and requirements.txt installed using ComfyUI's Python (gguf>=0.19.0,<0.20). No llama.cpp needed.") from exc
    import folder_paths
    import comfy.utils
    root = Path(folder_paths.models_dir) / "text_encoders" / "rs_gemma_q4"
    with _LOCK:
        logging.warning("Gemma Q4: %s", WARNING)
        def asset(local, entry):
            if local.strip():
                return _signature(local)
            progress = comfy.utils.ProgressBar(1)
            def update(current, total):
                from comfy.model_management import throw_exception_if_processing_interrupted
                throw_exception_if_processing_interrupted()
                progress.update_absolute(current, total)
            directory = ensure_files(entry, root, update)
            return _signature(directory / next(iter(entry["files"])))
        text = asset(gguf_path, TEXT)
        tokenizer = asset(tokenizer_path, TOKENIZER)
        vision = asset(mmproj_path, VISION) if image is not None else None
        return _load(text, tokenizer, vision)
