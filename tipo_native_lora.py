"""Built-in TIPO PEFT adapters, applied through ComfyUI's CLIP patcher.

No import-time I/O; only explicitly selected published files are fetched.
"""
import hashlib
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import urllib.request


BASE_MODEL = "p-e-w/gemma-4-E2B-it-heretic-ara"
BASE_REVISION = "c9a1d4c031981f14d86eeb0c7d87de7fafd34513"
CHOICES = ("none", "pilot-001", "pilot-002", "image-30k")
REQUIREMENTS = (
    "Requires a base: built-in Q4 GGUF (~3.19 GiB + tokenizer; image adds ~0.92 GiB), "
    "or native Gemma 4 E2B (~9.57 GiB) connected via Load CLIP. LoRA alone cannot run. "
    "베이스 모델 필요: 내부 Q4 자동 다운로드 또는 네이티브 CLIP 연결. "
    "Select none for the base; each adapter downloads ~45.3 MiB on first use. "
    "image-30k improves image-tag agreement; use tags_only scope for dual output. "
    "Tested on 16GB NVIDIA GPUs. GPU compute/RAM demand is significant; results can omit names or alter appearance. "
    "고사양 GPU 권장. 취향과 다른 결과, 이름 누락·외형 변경 가능. "
    "Experimental tag-only adapters; not the default for prose/VRS. "
    "Use p-e-w/gemma-4-E2B-it-heretic-ara; do not apply on an already merged pilot model."
)
# Public release identifiers and SHA-256 hashes, never personal dataset paths.
CATALOG = {
    "image-30k": {
        "repo": "raspie/gemma4-tipo-lora-image-30k", "revision": "5e919e81801a19066d6df8114d10730b1a9d1e5b",
        "files": {
            "adapter_config.json": (11125, "d3db6392a71e36e80486cc4eabe0bde581396399bf07b12f1a1c12157fababaa"),
            "adapter_model.safetensors": (47487272, "9cfae7497b9031923b238e7e94afd1a654cc9e86d732d39007fbf6fc093bf4dc"),
        },
    },
    "pilot-001": {
        "repo": "raspie/gemma4-tipo-lora-pilot-001", "revision": "432e2f074a200cf20137475e13020efa23b74aec",
        "files": {
            "adapter_config.json": (11125, "d3db6392a71e36e80486cc4eabe0bde581396399bf07b12f1a1c12157fababaa"),
            "adapter_model.safetensors": (47487272, "507ce9a126ef87506b7eb04858a711dec71b0c99908ff6dc28d8bf7721bb16ba"),
        },
    },
    "pilot-002": {
        "repo": "raspie/gemma4-tipo-lora-pilot-002", "revision": "c31d210508c0dfb9f3e08981baa73f2e06eafd0a",
        "files": {
            "adapter_config.json": (11125, "d3db6392a71e36e80486cc4eabe0bde581396399bf07b12f1a1c12157fababaa"),
            "adapter_model.safetensors": (47487272, "930d622ca59e5b9025fe8ce101bfb93de4ae8dff6ea0b3ef80abbbdecba90db7"),
        },
    },
}
_DOWNLOAD_LOCK = threading.Lock()
_TARGET = re.compile(r"model\.language_model\.layers\.(?:[0-9]|[12][0-9]|3[0-4])\.(?:self_attn\.(?:q|v|o)_proj|mlp\.(?:gate|up|down)_proj)$")


def _check_file(path, expected):
    size, digest = expected
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"TIPO model/adapter file has wrong size/type: {path}. Move it aside and retry, or install the published file offline.")
    signature = _file_stat(path)
    # If the filesystem cannot provide a change timestamp, do a full check
    # instead of trusting size/mtime alone (both can survive an in-place edit).
    verify = _check_digest if signature[-1] is not None else _check_digest.__wrapped__
    verify(str(path.resolve()), digest, signature)


def _windows_change_time(path):
    """Use Windows ChangeTime; Python st_ctime is creation time there.

    https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_basic_info
    Return None for unsupported filesystems/APIs so verification cannot be cached.
    """
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in
                    ("CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")]
        _fields_.append(("FileAttributes", wintypes.DWORD))

    try:
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
        query.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        query.restype = wintypes.BOOL
        info = FileBasicInfo()
        with Path(path).open("rb") as stream:
            handle = msvcrt.get_osfhandle(stream.fileno())
            if query(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
                return info.ChangeTime or None
    except (AttributeError, OSError):
        pass
    return None


def _file_stat(path):
    stat = Path(path).stat()
    changed = _windows_change_time(path) if os.name == "nt" else stat.st_ctime_ns
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, changed


@lru_cache(maxsize=32)
def _check_digest(filename, digest, signature):
    # Multi-GiB bases should be hashed once per unchanged local file, not once
    # per generation. Change time/inode invalidate edits even if mtime is restored.
    path = Path(filename)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise ValueError(f"TIPO model/adapter checksum mismatch: {path}. Existing files are not overwritten; move it aside and retry.")
    if _file_stat(path) != signature:
        raise ValueError(f"TIPO model/adapter changed during verification: {path}")


def ensure_adapter(choice, root, progress=None):
    """Download two pinned files atomically, or validate an offline install."""
    if choice not in CATALOG:
        raise ValueError(f"Unknown TIPO LoRA: {choice}")
    entry = CATALOG[choice]
    directory = Path(root) / choice
    return ensure_files(entry, directory, progress)


def ensure_files(entry, directory, progress=None):
    """Download a pinned model asset bundle; preserve existing invalid files."""
    directory = Path(directory)
    with _DOWNLOAD_LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        missing = []
        # Validate all existing assets before fetching anything. A broken local
        # bundle must not trigger a multi-GiB download before reporting it.
        for name, expected in entry["files"].items():
            if Path(name).name != name or name in (".", "..") or "\\" in name:
                raise ValueError("Model asset names must be plain filenames")
            destination = directory / name
            if destination.exists() or destination.is_symlink():
                _check_file(destination, expected)
            else:
                missing.append(name)
        if missing:
            if any(os.environ.get(key, "").lower() in ("1", "true", "yes", "on")
                   for key in ("RS_MODELS_OFFLINE", "HF_HUB_OFFLINE")):
                raise RuntimeError("Offline mode: missing model files: " + ", ".join(missing)
                                   + f". Install the pinned files in {directory} or disable RS_MODELS_OFFLINE/HF_HUB_OFFLINE.")
            required = sum(entry["files"][name][0] for name in missing)
            if shutil.disk_usage(directory).free < required + 64 * 1024 * 1024:
                raise OSError(f"Not enough disk space for model download ({required} bytes plus 64 MiB reserve): {directory}")
        for name, expected in entry["files"].items():
            destination = directory / name
            if destination.exists() or destination.is_symlink():
                _check_file(destination, expected)
                continue
            url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{name}"
            if progress:
                progress(0, expected[0])  # allow cancellation before network I/O
            logging.info("Preparing model asset %s (%s bytes)", name, expected[0])
            fd, tmp = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=directory)
            temp = Path(tmp)
            try:
                with os.fdopen(fd, "wb") as output:
                    request = urllib.request.Request(url, headers={"User-Agent": "RS-Gemma-TIPO-Native"})
                    with urllib.request.urlopen(request, timeout=60) as source:
                        total = 0
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > expected[0]:
                                raise ValueError("TIPO download exceeds published size")
                            output.write(chunk)
                            if progress:
                                progress(total, expected[0])
                _check_file(temp, expected)
                # Publish without overwriting files another process installed.
                try:
                    os.link(temp, destination)
                except FileExistsError:
                    _check_file(destination, expected)
            finally:
                temp.unlink(missing_ok=True)
    return directory


def adapter_mapping(config, tensors, model_state):
    """Validate the whole PEFT inventory before any model mutation."""
    if (config.get("peft_type") != "LORA" or config.get("bias") != "none"
            or config.get("base_model_name_or_path") != BASE_MODEL
            or config.get("revision") != BASE_REVISION
            or config.get("r") != 8 or config.get("lora_alpha") != 16):
        raise ValueError("Unsupported TIPO adapter config/base")
    if any(config.get(k) for k in ("use_dora", "use_rslora", "fan_in_fan_out", "lora_bias",
                                  "rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters")):
        raise ValueError("Unsupported TIPO adapter variant")
    targets = config.get("target_modules", [])
    if (not isinstance(targets, list) or len(targets) != 190 or len(set(targets)) != 190
            or any(not isinstance(t, str) or not _TARGET.fullmatch(t) for t in targets)):
        raise ValueError("Expected 190 language-only TIPO targets")
    import torch
    mapping = {}
    expected_keys = set()
    for target in targets:
        prefix = "base_model.model." + target
        akey, bkey = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
        expected_keys.update((akey, bkey))
        key = "gemma4.transformer.model." + target.removeprefix("model.language_model.") + ".weight"
        if akey not in tensors or bkey not in tensors or key not in model_state:
            raise ValueError(f"Missing TIPO adapter/base tensor for {target}")
        a, b, w = tensors[akey], tensors[bkey], model_state[key]
        if (a.ndim != 2 or b.ndim != 2 or w.ndim != 2 or a.shape[0] != 8
                or b.shape[1] != 8 or (b.shape[0], a.shape[1]) != tuple(w.shape)
                or not a.is_floating_point() or not b.is_floating_point()
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ValueError(f"TIPO tensor shape/rank/value mismatch: {target}")
        mapping[prefix] = key
    if set(tensors) != expected_keys:
        raise ValueError("Unexpected tensors in TIPO adapter; vision/PLE adapters are not supported")
    return mapping


def apply_tipo_lora(clip, choice="none"):
    if choice == "none":
        return clip
    if choice not in CATALOG:
        raise ValueError(f"Unknown TIPO LoRA: {choice}")
    patcher = getattr(clip, "patcher", None)
    if patcher is None or not callable(getattr(clip, "clone", None)):
        raise ValueError("TIPO LoRA requires a native Gemma 4 E2B CLIP. " + REQUIREMENTS)
    state = patcher.model_state_dict()
    embed = state.get("gemma4.transformer.model.embed_tokens.weight")
    if embed is None or tuple(embed.shape) != (262144, 1536):
        raise ValueError("TIPO LoRA requires the native Gemma 4 E2B base. " + REQUIREMENTS)
    import folder_paths
    import comfy.lora
    import comfy.utils
    import torch
    logging.warning("TIPO %s: %s", choice, REQUIREMENTS)
    progress = comfy.utils.ProgressBar(1)
    def update(current, total):
        from comfy.model_management import throw_exception_if_processing_interrupted
        throw_exception_if_processing_interrupted()
        progress.update_absolute(current, total)
    directory = ensure_adapter(choice, Path(folder_paths.models_dir) / "loras" / "rs_gemma_tipo",
                               progress=update)
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = comfy.utils.load_torch_file(str(directory / "adapter_model.safetensors"), safe_load=True)
    mapping = adapter_mapping(config, tensors, state)
    if set(mapping.values()) & set(getattr(patcher, "patches", {})):
        raise ValueError("TIPO language weights already have patches. Use the unpatched base to avoid applying LoRA twice.")
    for source in mapping:
        tensors[source + ".alpha"] = torch.tensor(config["lora_alpha"], dtype=torch.float32)
    patches = comfy.lora.load_lora(tensors, mapping)
    if set(patches) != set(mapping.values()):
        raise RuntimeError("ComfyUI did not recognize all 190 TIPO adapter targets. Update ComfyUI.")
    result = clip.clone()
    applied = result.patcher.add_patches(patches, strength_patch=1.0)
    if set(applied) != set(patches):
        raise RuntimeError("ComfyUI rejected TIPO adapter targets")
    return result
