"""Image -> TIPO Danbooru tags via Gemma-4 vision (llama.cpp mtmd).

Verified path: a prebuilt llama.cpp `llama-mtmd-cli` binary driving the
trained gemma-4-E2B vision GGUF + its co-trained mmproj. We shell out to
the binary (the proven path; llama-cpp-python's high-level API doesn't
drive gemma-4 vision), then run the SAME post-processing as the text
TIPO node (kgen sort / dedup / drop-redundant / count-conflict).

GPU is used automatically when available: on first use the CUDA prebuilt
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

try:
    from . import tipo_engine
except ImportError:
    import tipo_engine

# Pinned prebuilt llama.cpp release that supports gemma-4 mtmd (verified).
_LLAMA_BUILD = "b9209"
_LLAMA_REL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
              f"{_LLAMA_BUILD}/")
_CLI_NAMES = ("llama-mtmd-cli.exe", "llama-mtmd-cli")

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

    Windows+NVIDIA -> CUDA (13.1 if the driver is CUDA>=13, else 12.4);
    Linux+NVIDIA   -> Vulkan (no CUDA prebuilt is published for Linux);
    macOS          -> Metal (the standard macOS build is Metal-enabled).
    RS_TIPO_FORCE_CPU=1 or any detection failure -> None (CPU)."""
    global _GPU_BACKEND
    if _GPU_BACKEND == "?":
        _GPU_BACKEND = _detect_gpu_backend()
    return _GPU_BACKEND


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
    if os.environ.get("RS_TIPO_FORCE_CPU"):
        return None
    if sys.platform == "darwin":
        return ("metal", None)
    if platform.machine().lower() not in ("x86_64", "amd64", "x64"):
        return None  # GPU prebuilts are x64-only
    cuda = _nvidia_max_cuda()
    if sys.platform.startswith("win"):
        if cuda is not None:                 # NVIDIA -> CUDA (fastest path)
            if cuda >= (13, 0):
                return ("cuda", "13.1")
            if cuda >= (12, 0):
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
    req = urllib.request.Request(url, headers={"User-Agent": "rs-tipo-vision"})
    with urllib.request.urlopen(req, timeout=300) as r, open(apath, "wb") as f:
        shutil.copyfileobj(r, f)
    if kind == "zip":
        with zipfile.ZipFile(apath) as z:
            z.extractall(dest_dir)
    else:
        with tarfile.open(apath, "r:gz") as t:
            t.extractall(dest_dir)
    os.remove(apath)


def _resolve_mtmd_cli(backend):
    """Path to llama-mtmd-cli for `backend`; env override else download the
    prebuilt once into a backend-specific subdir (no build). CUDA on Windows
    also fetches the matching cudart runtime DLLs beside the binary."""
    env = os.environ.get("LLAMA_MTMD_CLI")
    if env and os.path.exists(env):
        return env
    bdir = os.path.join(_bin_dir(), _backend_tag(backend))
    found = _find_cli(bdir)
    if found:
        return found
    asset, kind = _llama_asset(backend)
    print(f"[TIPO-V] downloading llama.cpp {_LLAMA_BUILD} "
          f"({_backend_tag(backend)}: {asset}; first use)")
    _download_extract(_LLAMA_REL + asset, bdir, kind)
    found = _find_cli(bdir)
    if not found:
        raise RuntimeError("llama-mtmd-cli not found after extract")
    if backend and backend[0] == "cuda" and sys.platform.startswith("win"):
        cudart = f"cudart-llama-bin-win-cuda-{backend[1]}-x64.zip"
        print(f"[TIPO-V] downloading CUDA runtime ({cudart})")
        _download_extract(_LLAMA_REL + cudart, os.path.dirname(found), "zip")
    if not sys.platform.startswith("win"):
        try:
            os.chmod(found, 0o755)
        except OSError:
            pass
    return found


def _resolve_vision_model(gguf_path):
    """Local vision GGUF: given path, else auto-download the trained vision
    model (co-trained with the mmproj below)."""
    if gguf_path and gguf_path.strip():
        return gguf_path
    target = os.path.join(tipo_engine._gguf_dir(), _VISION_MODEL_FILE)
    if not os.path.exists(target):
        print("[TIPO-V] downloading trained vision model (first use)")
        tipo_engine._stream_download(_HF_BASE + _VISION_MODEL_FILE, target)
    return target


def _resolve_mmproj(path):
    """Local mmproj: given path, else auto-download the trained mmproj paired
    with the vision model (the base mmproj loses trained image grounding)."""
    if path and path.strip():
        return path
    target = os.path.join(tipo_engine._gguf_dir(), _VISION_MMPROJ_FILE)
    if not os.path.exists(target):
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
        arr = arr[0]
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8")
    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.fromarray(arr).save(p)
    return p


_PROSE = re.compile(r"^\s*(\d+[.)]|[-*#>]|here'?s|the |this |i'?ll|analyze|"
                    r"step|let'?s|okay|sure|\*\*)", re.I)
# gemma-4 reasoning/role channel artifact words to drop if they leak through
_ARTIFACT = {"thought", "thinking", "final", "analysis", "channel",
             "model", "assistant", "user", "system", "turn", "answer",
             "response", "output", "tags"}


def _extract_tags(raw):
    """Pull tag-like comma tokens out of mtmd output (drops gemma-4
    <|channel>thought markers and prose lines; salvages tag fragments)."""
    txt = tipo_engine._clean_special_tokens(raw or "")
    parts = []
    for line in txt.splitlines():
        s = line.strip().strip("`").strip()
        if not s or _PROSE.match(s):
            continue
        # keep lines that look like a tag list (commas, or few short words)
        if "," in s or (len(s.split()) <= 6 and not s.endswith((".", ":"))):
            parts.append(s.rstrip(":."))
    flat = ", ".join(parts) if parts else txt
    out = []
    for t in flat.split(","):
        t = t.strip().strip(".").strip()
        # tag-ish: short, no sentence-like content, not a channel artifact
        if (t and len(t.split()) <= 5 and not re.search(r"[.!?]", t)
                and t.lower() not in _ARTIFACT):
            out.append(t)
    return ", ".join(out)


def _run_cli(backend, model, mmproj, png, instr, gpu_layers, temperature,
             seed, timeout):
    """Resolve+run mtmd-cli for one backend. Returns stdout on success
    (rc==0 and non-empty), else '' (caller may fall back to CPU)."""
    cli = _resolve_mtmd_cli(backend)
    # GPU backends offload all layers unless the user set an explicit count.
    ngl = 999 if (backend is not None and int(gpu_layers) <= 0) else int(gpu_layers)
    # Force the built-in plain `gemma` chat template instead of the model's
    # embedded jinja: the embedded template enables a <|channel>thought
    # reasoning block that eats the token budget so the tag list is never
    # emitted. The plain template has no thinking channel.
    cmd = [cli, "-m", model, "--mmproj", mmproj, "--image", png,
           "--chat-template", "gemma", "--no-warmup",
           "-ngl", str(ngl), "-n", "768", "--temp", str(float(temperature)),
           "--seed", str(int(seed) % (2 ** 31)), "-p", instr]
    env = os.environ.copy()
    if not sys.platform.startswith("win"):
        # prebuilt cli links sibling .so/.dylib; ensure they resolve
        key = ("DYLD_LIBRARY_PATH" if sys.platform == "darwin"
               else "LD_LIBRARY_PATH")
        env[key] = os.path.dirname(cli) + os.pathsep + env.get(key, "")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="ignore",
                          timeout=timeout, env=env)
    if proc.returncode == 0 and (proc.stdout or "").strip():
        return proc.stdout
    print(f"[TIPO-V] {_backend_tag(backend)} run failed "
          f"(rc={proc.returncode})")
    return ""


def image_to_tags(image, gguf_path, mmproj_path, tag_length, ban_tags,
                  temperature, seed, sort_preset="danbooru", gpu_layers=0,
                  timeout=600):
    """Run gemma-4 vision -> cleaned, sorted TIPO tag string. Returns ''
    on any failure (caller treats as passthrough/empty)."""
    png = None
    try:
        model = _resolve_vision_model(gguf_path)
        if not model or not os.path.exists(model):
            print(f"[TIPO-V] model not found ({model!r})")
            return ""
        mmproj = _resolve_mmproj(mmproj_path)
        png = _save_image(image)

        target = {"very_short": 20, "short": 35,
                  "long": 60, "very_long": 90}.get(tag_length, 60)
        instr = ("Output ONLY a comma-separated list of English Danbooru "
                 "tags for this image. No thinking, no explanation, no "
                 "sentences. Capture the image and plausibly expand it "
                 f"(TIPO style). About {target} tags.")

        # Try GPU first (if available), then fall back to CPU on failure.
        gpu = _gpu_backend()
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

        # same post-processing as the text TIPO node
        try:
            tags = tipo_engine._postprocess_with_kgen(
                tags, ban_tags, sort_preset)
            tags = tipo_engine._resolve_count_conflicts(tags, "")
        except Exception as e:
            if not tipo_engine._have("kgen.formatter"):
                try:
                    tipo_engine._pip_install(["tipo-kgen"])
                    tags = tipo_engine._postprocess_with_kgen(
                        tags, ban_tags, sort_preset)
                    tags = tipo_engine._resolve_count_conflicts(tags, "")
                except Exception as e2:
                    print(f"[TIPO-V] post-process skipped ({e2})")
            else:
                print(f"[TIPO-V] post-process skipped ({e})")
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
