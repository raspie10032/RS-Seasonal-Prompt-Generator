"""Image -> TIPO Danbooru tags via Gemma-4 vision (llama.cpp mtmd).

Verified path: a prebuilt llama.cpp `llama-mtmd-cli` binary + the base
`unsloth/gemma-4-E2B-it` `mmproj-F16.gguf` paired with any gemma-4-E2B
text/merged GGUF (the TIPO model). We shell out to the binary (the
proven path; llama-cpp-python's high-level API doesn't drive gemma-4
vision) on CPU by default, then run the SAME post-processing as the text
TIPO node (kgen sort / dedup / drop-redundant / count-conflict).

Everything auto-resolves on first use and every failure falls back to an
empty string so the node never hard-fails. Heavy/native bits are the
llama.cpp binary (downloaded prebuilt, no build) + mmproj (downloaded).
"""

import os
import re
import sys
import zipfile
import tempfile
import subprocess
import urllib.request

try:
    from . import tipo_engine
except ImportError:
    import tipo_engine

# Pinned prebuilt llama.cpp release that supports gemma-4 mtmd (verified).
_LLAMA_BUILD = "b9209"
_LLAMA_ZIP = {
    "win": f"llama-{_LLAMA_BUILD}-bin-win-cpu-x64.zip",
}
_LLAMA_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
              f"{_LLAMA_BUILD}/" + _LLAMA_ZIP["win"])
_MMPROJ_URL = ("https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/"
               "resolve/main/mmproj-F16.gguf")
_MMPROJ_NAME = "gemma-4-E2B-it.mmproj-F16.gguf"


def _bin_dir():
    d = os.path.join(os.path.dirname(tipo_engine._gguf_dir()), "tipo_bin")
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_mtmd_cli():
    """Path to llama-mtmd-cli; env override, else download prebuilt once."""
    env = os.environ.get("LLAMA_MTMD_CLI")
    if env and os.path.exists(env):
        return env
    if not sys.platform.startswith("win"):
        raise RuntimeError(
            "auto-download of llama-mtmd-cli is Windows-only; set "
            "LLAMA_MTMD_CLI to a built llama-mtmd-cli for your OS")
    bdir = _bin_dir()
    for root, _, files in os.walk(bdir):
        for f in files:
            if f.lower() == "llama-mtmd-cli.exe":
                return os.path.join(root, f)
    print(f"[TIPO-V] downloading llama.cpp {_LLAMA_BUILD} (first use)")
    zpath = os.path.join(bdir, "llamacpp.zip")
    req = urllib.request.Request(_LLAMA_URL,
                                 headers={"User-Agent": "rs-tipo-vision"})
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(zpath, "wb") as f:
        f.write(r.read())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(bdir)
    os.remove(zpath)
    for root, _, files in os.walk(bdir):
        for f in files:
            if f.lower() == "llama-mtmd-cli.exe":
                return os.path.join(root, f)
    raise RuntimeError("llama-mtmd-cli.exe not found after extract")


def _resolve_mmproj(path):
    if path and path.strip():
        return path
    target = os.path.join(tipo_engine._gguf_dir(), _MMPROJ_NAME)
    if os.path.exists(target):
        return target
    print("[TIPO-V] downloading gemma-4-E2B mmproj (first use)")
    tipo_engine._stream_download(_MMPROJ_URL, target)
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
        model = tipo_engine._resolve_model_path(gguf_path)
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
        cmd = [cli, "-m", model, "--mmproj", mmproj, "--image", png,
               "--jinja", "--no-warmup", "-ngl", str(int(gpu_layers)),
               "-n", "768", "--temp", str(float(temperature)),
               "--seed", str(int(seed) % (2 ** 31)), "-p", instr]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="ignore",
                               timeout=timeout)
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
