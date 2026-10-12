"""Image -> TIPO Danbooru tags via Gemma-4 vision (llama.cpp mtmd).

Uses a prebuilt llama.cpp `llama-mtmd-cli` binary driving the
trained gemma-4-E2B vision GGUF + its co-trained mmproj. We shell out to
the binary (llama-cpp-python's high-level API doesn't
drive gemma-4 vision), then run the SAME post-processing as the text
TIPO node (kgen sort / dedup / drop-redundant / count-conflict).

GPU auto-detection is enabled with gpu_layers=-1 (0 keeps CPU): the CUDA prebuilt
(Windows/NVIDIA, plus the matching cudart runtime) or the Vulkan prebuilt
(Linux/NVIDIA) is fetched and all layers are offloaded; macOS uses the
Metal-enabled build; otherwise the CPU build is used. A GPU run that
fails transparently falls back to CPU. Set RS_TIPO_FORCE_CPU=1 to force
CPU, or LLAMA_MTMD_CLI to a binary you built yourself.

Everything auto-resolves on first use and every failure falls back to an
empty string so the node never hard-fails.
"""

import os
import re
import sys
import shutil
import zipfile
import tarfile
import platform
import tempfile
import subprocess
import urllib.request
import threading

try:
    from . import tipo_engine
except ImportError:
    import tipo_engine

# Pinned prebuilt llama.cpp release that supports gemma-4 mtmd (verified).
_LLAMA_BUILD = "b9209"
_LLAMA_REL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
              f"{_LLAMA_BUILD}/")
_CLI_NAMES = ("llama-mtmd-cli.exe", "llama-mtmd-cli")
_binary_lock = threading.Lock()

# Trained vision pair (model + mmproj co-trained together). The base unsloth
# mmproj loads but drops the trained image grounding, so the default is the
# trained pair from the same repo as the text model.
_HF_REPO = "raspie/gemma4-tipo-ko-gguf"
_HF_BASE = f"https://huggingface.co/{_HF_REPO}/resolve/main/"
_VISION_MODEL_FILE = "Gemma-tipo-vision-v1-E2B-heretic-ara-Q4_K_M.gguf"
_VISION_MMPROJ_FILE = "Gemma-tipo-vision-v1-E2B-heretic-ara.mmproj-f16.gguf"


def _bin_dir():
    d = os.path.join(os.path.dirname(tipo_engine._gguf_dir()), "tipo_bin")
    os.makedirs(d, exist_ok=True)
    return d


def _nvidia_max_cuda():
    """(major, minor) max CUDA the installed NVIDIA driver supports, or None."""
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=15)
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


# cached backend: ("cuda","13.1")/("cuda","12.4")/("vulkan",None)/
#                 ("metal",None)/None ; "?" = not yet probed
_GPU_BACKEND = "?"


def _gpu_backend():
    """Best GPU llama.cpp prebuilt for this machine, or None for CPU.

    Windows+NVIDIA -> CUDA (13.1 if supported, else 12.4 if supported);
    Linux+NVIDIA   -> Vulkan (no CUDA prebuilt is published for Linux);
    macOS          -> Metal (the standard macOS build is Metal-enabled).
    RS_TIPO_FORCE_CPU=1 or any detection failure -> None (CPU)."""
    global _GPU_BACKEND
    if _force_cpu():
        return None
    if _GPU_BACKEND == "?":
        _GPU_BACKEND = _detect_gpu_backend()
    return _GPU_BACKEND


def _force_cpu():
    return os.environ.get("RS_TIPO_FORCE_CPU", "").lower() in ("1", "true", "yes", "on")


def _vulkan_available():
    """True if the Vulkan loader is installed (a Vulkan-capable GPU is then
    very likely). Gates the Vulkan prebuilt for non-NVIDIA GPUs so CPU-only
    machines don't fetch it; an actual Vulkan failure still falls back to CPU."""
    import ctypes
    if sys.platform.startswith("win"):
        names = ["vulkan-1.dll"]
    elif sys.platform.startswith("linux"):
        names = ["libvulkan.so.1", "libvulkan.so"]
    else:
        return False
    for n in names:
        try:
            ctypes.CDLL(n)
            return True
        except OSError:
            continue
    return False


def _detect_gpu_backend():
    if _force_cpu():
        return None
    if sys.platform == "darwin":
        return ("metal", None)
    mach = platform.machine().lower()
    if sys.platform.startswith("linux") and mach in ("arm64", "aarch64"):
        return ("vulkan", None) if _vulkan_available() else None
    if mach not in ("x86_64", "amd64", "x64"):
        return None
    cuda = _nvidia_max_cuda()
    if sys.platform.startswith("win"):
        if cuda is not None:                 # NVIDIA -> CUDA (fastest path)
            if cuda >= (13, 1):
                return ("cuda", "13.1")
            if cuda >= (12, 4):
                return ("cuda", "12.4")
        if _vulkan_available():              # AMD / Intel / non-NVIDIA GPU
            return ("vulkan", None)
        return None
    if sys.platform.startswith("linux"):
        # no CUDA prebuilt is published for Linux; Vulkan covers NVIDIA too
        if cuda is not None or _vulkan_available():
            return ("vulkan", None)
        return None
    return None


def _backend_tag(backend):
    if not backend:
        return "cpu"
    return f"{backend[0]}-{backend[1]}" if backend[1] else backend[0]


def _llama_asset(backend):
    """(asset filename, archive kind) of the prebuilt for OS/arch/backend."""
    mach = platform.machine().lower()
    if mach in ("x86_64", "amd64", "x64"):
        arch = "x64"
    elif mach in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        raise RuntimeError(
            f"no llama.cpp prebuilt for CPU arch {mach!r}; set "
            "LLAMA_MTMD_CLI to a llama-mtmd-cli built for your machine")
    b = _LLAMA_BUILD
    kind = backend[0] if backend else None
    if sys.platform.startswith("win"):
        if kind == "cuda":
            return f"llama-{b}-bin-win-cuda-{backend[1]}-{arch}.zip", "zip"
        if kind == "vulkan":
            return f"llama-{b}-bin-win-vulkan-{arch}.zip", "zip"
        return f"llama-{b}-bin-win-cpu-{arch}.zip", "zip"
    if sys.platform.startswith("linux"):
        if kind == "vulkan":
            return f"llama-{b}-bin-ubuntu-vulkan-{arch}.tar.gz", "tar"
        return f"llama-{b}-bin-ubuntu-{arch}.tar.gz", "tar"
    if sys.platform == "darwin":
        return f"llama-{b}-bin-macos-{arch}.tar.gz", "tar"
    raise RuntimeError(
        f"no llama.cpp prebuilt for OS {sys.platform!r}; set "
        "LLAMA_MTMD_CLI to a llama-mtmd-cli built for your machine")


def _find_cli(bdir):
    for root, _, files in os.walk(bdir):
        for f in files:
            if f.lower() in _CLI_NAMES:
                return os.path.join(root, f)
    return None


def _download_extract(url, dest_dir, kind):
    os.makedirs(dest_dir, exist_ok=True)
    apath = os.path.join(dest_dir, os.path.basename(url))
    tipo_engine._stream_download(url, apath)
    try:
        if kind == "zip":
            with zipfile.ZipFile(apath) as z:
                for name in z.namelist():
                    _archive_path(dest_dir, name)
                z.extractall(dest_dir)
        else:
            with tarfile.open(apath, "r:gz") as t:
                for member in t.getmembers():
                    _archive_path(dest_dir, member.name)
                    if member.issym() or member.islnk():
                        parent = os.path.dirname(member.name) if member.issym() else ""
                        _archive_path(dest_dir, os.path.join(parent, member.linkname))
                    elif not (member.isfile() or member.isdir()):
                        raise ValueError("unsupported archive entry")
                if hasattr(tarfile, "data_filter"):
                    t.extractall(dest_dir, filter="data")
                else:  # Python 3.9/3.10 without the extraction-filter backport
                    t.extractall(dest_dir)
    finally:
        if os.path.exists(apath):
            os.remove(apath)


def _archive_path(root, name):
    name = name.replace("\\", "/")
    target = os.path.realpath(os.path.join(root, name))
    if (name.startswith("/") or re.match(r"^[A-Za-z]:", name)
            or os.path.commonpath([os.path.realpath(root), target]) != os.path.realpath(root)):
        raise ValueError(f"archive entry outside destination: {name!r}")
    return target


def _resolve_mtmd_cli(backend):
    """Path to llama-mtmd-cli for `backend`; env override else download the
    prebuilt once into a backend-specific subdir (no build). CUDA on Windows
    also fetches the matching cudart runtime DLLs beside the binary."""
    env = os.environ.get("LLAMA_MTMD_CLI")
    if env:
        override = os.path.abspath(os.path.expanduser(env))
        if not os.path.isfile(override):
            raise FileNotFoundError(f"LLAMA_MTMD_CLI not found: {override}")
        return override
    asset, kind = _llama_asset(backend)
    # Asset name includes release, OS, architecture and backend. A directory
    # is published only after BOTH the executable and CUDA runtime arrive.
    bdir = os.path.join(_bin_dir(), asset.replace(".tar.gz", "").replace(".zip", ""))
    with _binary_lock:
        found = _find_cli(bdir)
        if found:
            return found
        print(f"[TIPO-V] downloading llama.cpp {_LLAMA_BUILD} ({asset}; first use)")
        stage = tempfile.mkdtemp(prefix=".install-", dir=_bin_dir())
        try:
            _download_extract(_LLAMA_REL + asset, stage, kind)
            found = _find_cli(stage)
            if not found:
                raise RuntimeError("llama-mtmd-cli not found after extract")
            if backend and backend[0] == "cuda" and sys.platform.startswith("win"):
                cudart = f"cudart-llama-bin-win-cuda-{backend[1]}-x64.zip"
                _download_extract(_LLAMA_REL + cudart, os.path.dirname(found), "zip")
            if not sys.platform.startswith("win"):
                os.chmod(found, 0o755)
            relative = os.path.relpath(found, stage)
            os.replace(stage, bdir)
            return os.path.join(bdir, relative)
        finally:
            if os.path.exists(stage):
                shutil.rmtree(stage)


def _resolve_vision_model(gguf_path):
    """Local vision GGUF: given path, else auto-download the trained vision
    model (co-trained with the mmproj below)."""
    if gguf_path and gguf_path.strip():
        return os.path.abspath(os.path.expanduser(gguf_path.strip()))
    target = os.path.join(tipo_engine._gguf_dir(), _VISION_MODEL_FILE)
    if not os.path.isfile(target) or os.path.getsize(target) == 0:
        print("[TIPO-V] downloading trained vision model (first use)")
        tipo_engine._stream_download(_HF_BASE + _VISION_MODEL_FILE, target)
    return target


def _resolve_mmproj(path):
    """Local mmproj: given path, else auto-download the trained mmproj paired
    with the vision model (the base mmproj loses trained image grounding)."""
    if path and path.strip():
        return os.path.abspath(os.path.expanduser(path.strip()))
    target = os.path.join(tipo_engine._gguf_dir(), _VISION_MMPROJ_FILE)
    if not os.path.isfile(target) or os.path.getsize(target) == 0:
        print("[TIPO-V] downloading trained vision mmproj (first use)")
        tipo_engine._stream_download(_HF_BASE + _VISION_MMPROJ_FILE, target)
    return target


def _save_image(image):
    """ComfyUI IMAGE tensor [B,H,W,C] 0..1 -> temp PNG path."""
    from PIL import Image
    import numpy as np
    arr = image
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError("Gemma TIPO accepts one image; select one batch item first")
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4) or not arr.size:
        raise ValueError("expected a nonempty HWC image with 1, 3 or 4 channels")
    if not np.isfinite(arr).all():
        raise ValueError("image contains non-finite pixels")
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8")
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(arr).convert("RGB").save(p)
    except BaseException:
        os.remove(p)
        raise
    return p


# Shared cleanup keeps text and vision output semantics identical.
_strip_reasoning = tipo_engine._strip_reasoning
_extract_tags = tipo_engine._extract_tags


def _run_cli(backend, model, mmproj, png, instr, gpu_layers, temperature,
             seed, timeout):
    """Resolve+run mtmd-cli for one backend. Returns stdout on success
    (rc==0 and non-empty), else '' (caller may fall back to CPU)."""
    cli = _resolve_mtmd_cli(backend)
    # GPU backends offload all layers unless the user set an explicit count.
    ngl = (999 if int(gpu_layers) < 0 else int(gpu_layers)) if backend else 0
    # Force the built-in plain `gemma` chat template instead of the model's
    # embedded jinja: the embedded template enables a <|channel>thought
    # reasoning block that eats the token budget so the tag list is never
    # emitted. The plain template has no thinking channel.
    cmd = [cli, "-m", model, "--mmproj", mmproj, "--image", png,
           "--chat-template", "gemma", "--no-warmup",
           "-ngl", str(ngl), "-t", "2", "-tb", "2",
           "-n", "1024", "--temp", str(float(temperature)),
           "--seed", str(int(seed) % (2 ** 31)), "-p", instr]
    if backend is None:
        cmd.extend(["--device", "none", "--no-mmproj-offload"])
    env = os.environ.copy()
    if not sys.platform.startswith("win"):
        # prebuilt cli links sibling .so/.dylib; ensure they resolve
        key = ("DYLD_LIBRARY_PATH" if sys.platform == "darwin"
               else "LD_LIBRARY_PATH")
        env[key] = os.path.dirname(cli) + os.pathsep + env.get(key, "")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="ignore",
                          timeout=timeout, env=env, stdin=subprocess.DEVNULL)
    if proc.returncode == 0 and (proc.stdout or "").strip():
        return proc.stdout
    print(f"[TIPO-V] {_backend_tag(backend)} run failed "
          f"(rc={proc.returncode}): {(proc.stderr or '')[-1500:]}")
    return ""


def image_to_tags(image, gguf_path, mmproj_path, tag_length, ban_tags,
                  temperature, seed, sort_preset="danbooru", gpu_layers=0,
                  timeout=600):
    """Run gemma-4 vision -> cleaned, sorted TIPO tag string. Returns ''
    on any failure (caller treats as passthrough/empty)."""
    png = None
    try:
        # Reject unsupported batches before downloading runtime/model files.
        png = _save_image(image)
        model = _resolve_vision_model(gguf_path)
        if not model or not os.path.isfile(model):
            print(f"[TIPO-V] model not found ({model!r})")
            return ""
        mmproj = _resolve_mmproj(mmproj_path)
        if not os.path.isfile(mmproj):
            print(f"[TIPO-V] mmproj not found ({mmproj!r})")
            return ""

        target = {"very_short": 20, "short": 35,
                  "long": 60, "very_long": 90}.get(tag_length, 60)
        instr = ("Output ONLY a comma-separated list of English Danbooru "
                 "tags for this image. No thinking, no explanation, no "
                 "sentences. Capture the image and plausibly expand it "
                 f"(TIPO style). About {target} tags."
                 + tipo_engine._avoid_hint(ban_tags))

        # Try GPU first (if available), then fall back to CPU on failure.
        gpu = _gpu_backend() if int(gpu_layers) != 0 else None
        raw = ""
        for backend in ([gpu, None] if gpu is not None else [None]):
            try:
                raw = _run_cli(backend, model, mmproj, png, instr,
                               gpu_layers, temperature, seed, timeout)
            except Exception as e:
                print(f"[TIPO-V] {_backend_tag(backend)} unavailable "
                      f"({type(e).__name__}: {e})")
                raw = ""
            if raw:
                break

        tags = _extract_tags(raw)
        if not tags:
            print("[TIPO-V] no tags parsed from model output")
            return ""

        tags = tipo_engine._postprocess(tags, ban_tags, sort_preset)
        return tags
    except Exception as e:
        print(f"[TIPO-V] failed ({type(e).__name__}: {e}); empty output")
        return ""
    finally:
        if png and os.path.exists(png):
            try:
                os.remove(png)
            except OSError:
                pass
