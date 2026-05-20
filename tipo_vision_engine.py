"""Image -> TIPO Danbooru tags via Gemma-4 vision (llama.cpp mtmd).

Verified path: a prebuilt llama.cpp `llama-mtmd-cli` binary + a paired
text LM + mmproj GGUF. We shell out to the binary (the proven path;
llama-cpp-python's high-level API doesn't drive gemma-4 vision) on CPU
by default, then run the SAME post-processing as the text TIPO node
(kgen sort / dedup / drop-redundant / count-conflict).

**Default pair (v2.8.0+): the trained Gemma-tipo-vision-v1.** Both files
live in the same HF repo as the text TIPO model and auto-download into
ComfyUI/models/gguf on first vision use:

  Gemma-tipo-vision-v1-E2B-heretic-ara-Q4_K_M.gguf     (text LM)
  Gemma-tipo-vision-v1-E2B-heretic-ara.mmproj-f16.gguf (TRAINED mmproj)

The trained mmproj carries the fine-tuned multi-modal projector, so
pairing it with the vision-v1 text GGUF gives genuine image grounding
(verified character ID + image-specific attributes). Earlier versions
(<= 2.7.3) paired the *text* `gemma4-tipo-ko` model with the *base*
unsloth mmproj — that was TIPO-style hallucination, not real image
understanding.

Backward compat: if the user explicitly selects a non-vision model in
the node dropdown (e.g. the legacy `gemma4-tipo-ko-v2-Q4_K_M.gguf`),
the auto-mmproj falls back to the base unsloth mmproj (the old path).
Model name lookup is by substring ("vision" in basename) so user can
also drop their own custom vision GGUF in and have it paired
automatically.

Everything auto-resolves on first use and every failure falls back to an
empty string so the node never hard-fails. Heavy/native bits are the
llama.cpp binary (downloaded prebuilt, no build) + mmproj (downloaded).
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

# Default vision pair (v2.8.0+) — the trained Gemma-tipo-vision-v1 model.
# Both files live in the same HF repo as the text TIPO model.
VISION_HF_REPO = "raspie/gemma4-tipo-ko-gguf"
VISION_GGUF_FILE = "Gemma-tipo-vision-v1-E2B-heretic-ara-Q4_K_M.gguf"
VISION_MMPROJ_FILE = "Gemma-tipo-vision-v1-E2B-heretic-ara.mmproj-f16.gguf"

# Legacy fallback (still used for backward compat if the user explicitly
# picks the old text TIPO model in the dropdown — see _resolve_mmproj).
_BASE_MMPROJ_URL = ("https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/"
                    "resolve/main/mmproj-F16.gguf")
_BASE_MMPROJ_NAME = "gemma-4-E2B-it.mmproj-F16.gguf"


def _bin_dir():
    d = os.path.join(os.path.dirname(tipo_engine._gguf_dir()), "tipo_bin")
    os.makedirs(d, exist_ok=True)
    return d


def _llama_asset():
    """(asset filename, archive kind) of the CPU prebuilt for this OS/arch."""
    mach = platform.machine().lower()
    if mach in ("x86_64", "amd64", "x64"):
        arch = "x64"
    elif mach in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        raise RuntimeError(
            f"no llama.cpp prebuilt for CPU arch {mach!r}; set "
            "LLAMA_MTMD_CLI to a llama-mtmd-cli built for your machine")
    if sys.platform.startswith("win"):
        return f"llama-{_LLAMA_BUILD}-bin-win-cpu-{arch}.zip", "zip"
    if sys.platform.startswith("linux"):
        return f"llama-{_LLAMA_BUILD}-bin-ubuntu-{arch}.tar.gz", "tar"
    if sys.platform == "darwin":
        return f"llama-{_LLAMA_BUILD}-bin-macos-{arch}.tar.gz", "tar"
    raise RuntimeError(
        f"no llama.cpp prebuilt for OS {sys.platform!r}; set "
        "LLAMA_MTMD_CLI to a llama-mtmd-cli built for your machine")


def _find_cli(bdir):
    for root, _, files in os.walk(bdir):
        for f in files:
            if f.lower() in _CLI_NAMES:
                return os.path.join(root, f)
    return None


def _resolve_mtmd_cli():
    """Path to llama-mtmd-cli; env override, else download the prebuilt
    CPU build for this OS/arch once (Windows/Linux/macOS, no build)."""
    env = os.environ.get("LLAMA_MTMD_CLI")
    if env and os.path.exists(env):
        return env
    bdir = _bin_dir()
    found = _find_cli(bdir)
    if found:
        return found
    asset, kind = _llama_asset()
    print(f"[TIPO-V] downloading llama.cpp {_LLAMA_BUILD} ({asset}; first use)")
    apath = os.path.join(bdir, asset)
    req = urllib.request.Request(_LLAMA_REL + asset,
                                 headers={"User-Agent": "rs-tipo-vision"})
    with urllib.request.urlopen(req, timeout=300) as r, \
            open(apath, "wb") as f:
        shutil.copyfileobj(r, f)
    if kind == "zip":
        with zipfile.ZipFile(apath) as z:
            z.extractall(bdir)
    else:
        with tarfile.open(apath, "r:gz") as t:
            t.extractall(bdir)
    os.remove(apath)
    found = _find_cli(bdir)
    if not found:
        raise RuntimeError("llama-mtmd-cli not found after extract")
    if not sys.platform.startswith("win"):
        try:
            os.chmod(found, 0o755)
        except OSError:
            pass
    return found


def _resolve_vision_model(gguf_path):
    """Default to the trained vision LM (Gemma-tipo-vision-v1), NOT the
    text TIPO Korean model. Honors an explicit user pick (gguf_path)."""
    if gguf_path and gguf_path.strip():
        return gguf_path
    target_dir = tipo_engine._gguf_dir()
    target = os.path.join(target_dir, VISION_GGUF_FILE)
    if os.path.exists(target):
        return target
    url = f"https://huggingface.co/{VISION_HF_REPO}/resolve/main/{VISION_GGUF_FILE}"
    print(f"[TIPO-V] downloading trained vision LM "
          f"{VISION_HF_REPO}/{VISION_GGUF_FILE} -> {target_dir} (first use)")
    tipo_engine._stream_download(url, target)
    return target


def _resolve_mmproj(path, model_basename=""):
    """Pair mmproj with the LM. Explicit `path` wins. Otherwise:
      - vision-trained LM (basename contains 'vision') -> trained mmproj
        from VISION_HF_REPO. This is the v2.8.0+ default.
      - any other model (legacy text TIPO etc.) -> base unsloth mmproj
        (the old <=2.7.3 behaviour, kept for backward compat).
    """
    if path and path.strip():
        return path
    target_dir = tipo_engine._gguf_dir()
    is_vision_lm = "vision" in (model_basename or "").lower()
    if is_vision_lm:
        target = os.path.join(target_dir, VISION_MMPROJ_FILE)
        if os.path.exists(target):
            return target
        url = (f"https://huggingface.co/{VISION_HF_REPO}/resolve/main/"
               f"{VISION_MMPROJ_FILE}")
        print(f"[TIPO-V] downloading TRAINED mmproj "
              f"{VISION_HF_REPO}/{VISION_MMPROJ_FILE} (first use)")
        tipo_engine._stream_download(url, target)
        return target
    # legacy path: pair non-vision LMs with the base unsloth mmproj
    target = os.path.join(target_dir, _BASE_MMPROJ_NAME)
    if os.path.exists(target):
        return target
    print("[TIPO-V] downloading base gemma-4-E2B mmproj (legacy pair, "
          "first use)")
    tipo_engine._stream_download(_BASE_MMPROJ_URL, target)
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


def image_to_tags(image, gguf_path, mmproj_path, tag_length, ban_tags,
                  temperature, seed, sort_preset="danbooru", gpu_layers=0,
                  timeout=600):
    """Run gemma-4 vision -> cleaned, sorted TIPO tag string. Returns ''
    on any failure (caller treats as passthrough/empty)."""
    png = None
    try:
        cli = _resolve_mtmd_cli()
        # Vision mode defaults to the trained vision-v1 LM (not the
        # text TIPO Korean model), then pairs the right mmproj based
        # on the chosen LM (trained mmproj for vision LM, base mmproj
        # for legacy text LMs picked manually).
        model = _resolve_vision_model(gguf_path)
        if not model or not os.path.exists(model):
            print(f"[TIPO-V] model not found ({model!r})")
            return ""
        mmproj = _resolve_mmproj(mmproj_path, os.path.basename(model))
        png = _save_image(image)

        target = {"very_short": 20, "short": 35,
                  "long": 60, "very_long": 90}.get(tag_length, 60)
        instr = ("Output ONLY a comma-separated list of English Danbooru "
                 "tags for this image. No thinking, no explanation, no "
                 "sentences. Capture the image and plausibly expand it "
                 f"(TIPO style). About {target} tags.")
        # Force the built-in plain `gemma` chat template instead of the
        # model's embedded jinja: the embedded template enables a
        # <|channel>thought reasoning block that eats the whole token
        # budget so the final tag list is never emitted (caption returns
        # empty). The plain template has no thinking channel -> the model
        # outputs the comma-separated tags directly.
        cmd = [cli, "-m", model, "--mmproj", mmproj, "--image", png,
               "--chat-template", "gemma", "--no-warmup",
               "-ngl", str(int(gpu_layers)),
               "-n", "768", "--temp", str(float(temperature)),
               "--seed", str(int(seed) % (2 ** 31)), "-p", instr]
        env = os.environ.copy()
        if not sys.platform.startswith("win"):
            # prebuilt cli links sibling .so/.dylib; ensure they resolve
            key = ("DYLD_LIBRARY_PATH" if sys.platform == "darwin"
                   else "LD_LIBRARY_PATH")
            cdir = os.path.dirname(cli)
            env[key] = cdir + os.pathsep + env.get(key, "")
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="ignore",
                               timeout=timeout, env=env)
        raw = proc.stdout or ""
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
