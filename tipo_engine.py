"""In-process TIPO prompt expansion.

Inference: a local GGUF model via llama-cpp-python (ported from
NAI-FaceDetailer's tipo.py chat-completion approach), so a custom
Gemma-4-E2B GGUF can be used as-is.

Post-processing: kgen.formatter (pure-Python, lightweight) — only the
light formatter is imported, never kgen.executor / kgen.models.

Optional packages must be installed explicitly in ComfyUI's Python.
Inference never installs or upgrades packages. All heavy imports are lazy and every failure path falls
back to the original prompt so the node never hard-fails.
"""

import os
import re
import sys
import time
import tempfile
import threading
import importlib
import importlib.metadata
import urllib.request
from contextlib import contextmanager

# Dependencies are checked on use, never installed into a running ComfyUI.
# llama-cpp-python below this version cannot load the gemma4 architecture.
_MIN_LLAMA = (0, 3, 23)
_download_lock = threading.Lock()
_inference_lock = threading.RLock()
_inference_state = threading.local()
_LLAMA_IMPORT_ERROR = ""


def _have(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def _ver_tuple(s):
    nums = re.findall(r"\d+", s or "")
    return tuple(int(x) for x in nums[:3]) if nums else None


def _imported_llama_version():
    global _LLAMA_IMPORT_ERROR
    try:
        import llama_cpp
        _LLAMA_IMPORT_ERROR = ""
        return _ver_tuple(getattr(llama_cpp, "__version__", ""))
    except Exception as e:
        _LLAMA_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        return None


def _installed_llama_version():
    """Read package metadata even in uv-managed environments without pip."""
    try:
        return _ver_tuple(importlib.metadata.version("llama-cpp-python"))
    except importlib.metadata.PackageNotFoundError:
        return None


def _pip_install(args):
    # Keep the old internal entry point fail-closed for integrations importing it.
    raise RuntimeError("Runtime package installation is disabled. Use Gemma TIPO Native "
                       "to avoid llama.cpp wheels, or install optional packages explicitly "
                       "using ComfyUI's Python and restart.")


def _ensure_formatter():
    """Vision only needs the formatter, never the Python inference backend."""
    return _have("kgen.formatter")


def _restart_needed():
    """True if a gemma4-capable llama-cpp-python is on disk but an older one
    is already imported in this process (C-extension can't be hot-reloaded)."""
    ins = _installed_llama_version()
    iv = _imported_llama_version()
    return bool(ins and ins >= _MIN_LLAMA and (iv is None or iv < _MIN_LLAMA)
                and "llama_cpp" in sys.modules)


def _ensure_deps():
    """Check the existing optional backend without changing installed packages."""
    iv = _imported_llama_version()
    if iv is not None and iv >= _MIN_LLAMA:
        return True
    if _restart_needed():
        print("[TIPO] RESTART ComfyUI to use the installed llama-cpp-python version.")
    detail = _LLAMA_IMPORT_ERROR or f"loaded version {iv}; need >= {_MIN_LLAMA}"
    print(f"[TIPO] Optional llama-cpp-python unavailable: {detail}. "
          "No packages were installed or upgraded. Use Gemma TIPO Native / "
          "Gemma Prompt Expander Native for ComfyUI PyTorch without llama.cpp "
          "or install a compatible optional backend using ComfyUI's Python and restart.")
    return False

# Default model auto-downloaded from HuggingFace when no local path is given.
HF_REPO = "raspie/gemma4-tipo-ko-gguf"
HF_FILE = "gemma4-tipo-ko-v2-Q4_K_M.gguf"


def _gguf_dir():
    """Resolve <ComfyUI>/models/gguf so the model lives inside ComfyUI.

    Uses ComfyUI's folder_paths when running inside ComfyUI; otherwise walks
    up from this file to a `custom_nodes` parent; final fallback is a
    repo-local models/gguf (keeps standalone use self-contained)."""
    try:
        import folder_paths

        base = os.path.join(folder_paths.models_dir, "gguf")
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        base = None
        p = here
        while True:
            parent = os.path.dirname(p)
            if parent == p:
                break
            if os.path.basename(p) == "custom_nodes":
                base = os.path.join(os.path.dirname(p), "models", "gguf")
                break
            p = parent
        if base is None:
            base = os.path.join(here, "models", "gguf")
    os.makedirs(base, exist_ok=True)
    return base


def _resolve_model_path(gguf_path):
    """Return a local GGUF path: use the given path, else auto-download the
    default model into <ComfyUI>/models/gguf (reused if already present)."""
    if gguf_path and gguf_path.strip():
        return os.path.abspath(os.path.expanduser(gguf_path.strip()))

    target_dir = _gguf_dir()
    target = os.path.join(target_dir, HF_FILE)
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return target

    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}"
    print(f"[TIPO] downloading default model {HF_REPO}/{HF_FILE} -> {target_dir} (first use)")
    _stream_download(url, target)
    return target


def _progress_bar(total):
    """ComfyUI node progress bar if available, else None (stdlib-only)."""
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(total)
    except Exception:
        return None


def _stream_download(url, dest):
    """Download `url` to `dest` showing a ComfyUI node progress bar (and
    console %). Writes to a .part file and atomically renames on success."""
    # Recheck under the lock: two nodes may resolve the same missing model.
    with _download_lock:
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            return
        _download_file(url, dest)


def _download_file(url, dest):
    """Publish only a complete, nonempty download; clean up even on cancel."""
    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, part = tempfile.mkstemp(prefix=os.path.basename(dest) + ".",
                                suffix=".part", dir=os.path.dirname(dest))
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "rs-seasonal-tipo"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            pbar = _progress_bar(total) if total else None
            done = 0
            last = 0.0
            with open(part, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if pbar is not None:
                        # ComfyUI's progress hook also raises on cancellation.
                        # Propagate it so the finally block removes the partial
                        # file instead of completing a cancelled large download.
                        pbar.update_absolute(done, total)
                    now = time.time()
                    if total and now - last > 2:
                        last = now
                        print(f"[TIPO] downloading model: "
                              f"{done * 100 // total}% "
                              f"({done >> 20}/{total >> 20} MB)")
            if not done or (total and done != total):
                raise IOError(f"incomplete download: received {done} of {total} bytes")
        os.replace(part, dest)
        print("[TIPO] download complete")
    finally:
        if os.path.exists(part):
            os.remove(part)


SYSTEM_TIPO = (
    "You are a Danbooru tag expert. "
    "Given partial image tags, output a clean, concise, and high-impact Danbooru tag set. "
    "Focus on essential descriptive tags that define the scene, character, and atmosphere. "
    "Avoid redundant or meta tags. "
    "Output ONLY comma-separated tags in order of importance. No explanation."
)

# tag_length -> approximate target tag count hint for the system/user message
_TARGET_TAGS = {
    "very_short": 15,
    "short": 25,
    "long": 45,
    "very_long": 70,
}

# Prompt-sorting presets: category order applied by kgen.formatter.apply_format.
# Category names must match kgen exactly (note plural: characters/copyrights).
# Rating category tag-list override (applied to kgen at runtime, not patched
# into site-packages). A tag is classified as "rating" iff it is in this set.
RATING_TAGS = [
    "safe", "nsfw",
    "general", "rating:general",
    "sensitive", "rating:sensitive",
    "questionable", "rating:questionable",
    "explicit", "rating:explicit",
]

SORT_PRESETS = {
    "danbooru": "<|artist|>, <|rating|>, <|special|>, <|characters|>, "
                "<|copyrights|>, <|general|>, <|quality|>, <|meta|>",
    "quality_first": "<|quality|>, <|special|>, <|characters|>, <|copyrights|>, "
                     "<|artist|>, <|general|>, <|meta|>, <|rating|>",
    "artist_first": "<|artist|>, <|characters|>, <|copyrights|>, <|special|>, "
                    "<|general|>, <|quality|>, <|meta|>, <|rating|>",
    "general_only": "<|special|>, <|characters|>, <|general|>",
    "simple": "<|special|>, <|rating|>, <|general|>",
}
DEFAULT_SORT = "danbooru"
SORT_CHOICES = ["danbooru", "quality_first", "artist_first",
                "general_only", "simple"]

_model_cache = {}
_cache_lock = threading.Lock()


@contextmanager
def tipo_inference():
    """Own cached model acquisition through the last inference using its handle.

    A cached handle is borrowed only until this scope exits. It must not escape
    the scope. Nested scopes may reuse the same settings, but cannot replace a
    model still borrowed by the outer scope. Private cache=False handles keep
    their existing caller-owned lifetime.
    """
    with _inference_lock:
        depth = getattr(_inference_state, "depth", 0)
        if depth == 0:
            _inference_state.key = None
        _inference_state.depth = depth + 1
        try:
            yield
        finally:
            _inference_state.depth = depth
            if depth == 0:
                _inference_state.key = None


def _clean_special_tokens(raw):
    # TIPO category tokens (<|general|> etc.) -> comma so adjacent tags don't
    # glue together. Match named model tokens, not tag punctuation: ':<' must
    # never consume the remainder of the answer, including supplied names.
    raw = re.sub(r"<\|[^|>]*\|>", ", ", raw)
    raw = re.sub(r"<(?:/?[A-Za-z][A-Za-z0-9_|]*|\|[^|>]*\|?)(?:>|$)", "", raw)
    return raw


def _is_korean(text):
    return bool(re.search(r"[가-힣]", text))


def _looks_like_nl(text):
    """Heuristic: Korean text, or a comma-segment that reads like a phrase."""
    if _is_korean(text):
        return True
    return any(len(seg.split()) > 4 for seg in text.split(","))


def _dedup(tags):
    seen, out = set(), []
    for t in tags.split(","):
        t = t.strip()
        key = _tag_key(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return ", ".join(out)


def _tag_key(tag):
    """Compare spelling/weight variants without losing counts or identities."""
    tag = tag.strip()
    weighted = re.fullmatch(r"\((.*):[+-]?(?:\d+(?:\.\d*)?|\.\d+)\)", tag)
    if weighted:
        tag = weighted.group(1)
    return re.sub(r"\s+", " ", tag.replace("_", " ")).strip().casefold()


def _filter_banned(tags, ban_tags):
    banned = {_tag_key(b) for b in (ban_tags or "").split(",") if b.strip()}
    return ", ".join(t.strip() for t in tags.split(",")
                     if t.strip() and _tag_key(t) not in banned)


def _strip_reasoning(raw):
    """Discard explicit reasoning blocks, including unfinished ones."""
    raw = raw or ""
    raw = re.sub(r"<think>.*?(?:</think>|$)", "", raw, flags=re.S | re.I)
    markers = list(re.finditer(r"<\|channel>thought|<\|channel\|>analysis", raw))
    if markers:
        tail = raw[markers[-1].end():]
        end = re.search(r"<channel\|>|(?<!<)channel\|>|<\|channel\|>final", tail)
        if end is None:
            return ""
        raw = tail[end.end():]
    else:
        # Some CLI builds print only the channel-close marker.
        raw = re.split(r"<channel\|>|(?<!<)channel\|>", raw)[-1]
    return raw


def _extract_tags(raw):
    """Read plain, category-delimited or grouped tag lists from either model."""
    txt = _clean_special_tokens(_strip_reasoning(raw))
    out = []
    for line in txt.splitlines():
        line = line.strip().strip("`").strip()
        if not line or re.match(r"^(?:here(?:'s| are)|the |this |i |i'll|let's|okay|sure)\b",
                                line, re.I):
            continue
        line = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", line)
        for tag in line.split(","):
            tag = re.sub(r"^[a-z][a-z_ ]*:\s+", "", tag.strip(), flags=re.I)
            tag = tag.strip().strip(".").strip()
            # Internal dots occur in names and weighted tags (d.va, :1.2).
            if (tag and len(tag.split()) <= 5 and not re.search(r"[!?]|\.\s", tag)
                    and tag.casefold() not in {"thought", "thinking", "final", "analysis",
                                              "model", "assistant", "user", "system",
                                              "answer", "output", "tags"}):
                out.append(tag)
    return ", ".join(out)


def _drop_redundant(csv):
    """Drop a general tag when a more specific tag with the same head noun
    is present, to save tokens: e.g. with "white skirt" present, "skirt" is
    redundant and removed. Uses a suffix rule (the longer tag must end with
    " " + the shorter tag) so distinct tags like "dress" vs "dress shirt"
    are NOT collapsed."""
    tags = [t.strip() for t in csv.split(",") if t.strip()]
    low = [_tag_key(t) for t in tags]
    drop = set()
    for i, s in enumerate(low):
        for j, l in enumerate(low):
            if i == j:
                continue
            if l.endswith(" " + s):  # l = "<modifier> <s>"
                drop.add(i)
                break
    return ", ".join(t for k, t in enumerate(tags) if k not in drop)


_COUNT_RE = {
    "girls": re.compile(r"^(?:\d+\+?\s?girls?|multiple girls)$"),
    "boys": re.compile(r"^(?:\d+\+?\s?boys?|multiple boys)$"),
    "others": re.compile(r"^(?:\d+\+?\s?others?|multiple others)$"),
}


def _count_group(tag):
    t = _tag_key(tag)
    for g, rx in _COUNT_RE.items():
        if rx.match(t):
            return g
    return None


def _resolve_count_conflicts(csv, input_prompt):
    """Each person-count group (girls / boys / others) must have at most one
    tag: e.g. with `1girl` present, `4girls` / `multiple girls` are dropped.
    Groups are independent (`1girl` + `1boy` is fine). The kept tag is the
    one present in the user's input (first by input order), else the first
    in the output."""
    input_low = [_tag_key(t) for t in input_prompt.split(",")]
    tags = [t.strip() for t in csv.split(",") if t.strip()]

    by_group = {}
    for t in tags:
        g = _count_group(t)
        if g:
            by_group.setdefault(g, []).append(t)

    keep = {}
    for g, members in by_group.items():
        if len(members) <= 1:
            continue
        chosen = None
        for it in input_low:
            for m in members:
                if _tag_key(m) == it:
                    chosen = m
                    break
            if chosen:
                break
        keep[g] = chosen or members[0]

    out = []
    for t in tags:
        g = _count_group(t)
        if g in keep and t != keep[g]:
            continue
        out.append(t)
    return ", ".join(out)


def _is_model_file(name):
    return name.lower().endswith(".gguf") and "mmproj" not in name.lower()


def require_gpu_offload():
    """Reject a CPU-only build instead of silently ignoring requested offload."""
    from llama_cpp import llama_supports_gpu_offload
    if not llama_supports_gpu_offload():
        raise RuntimeError("GPU offload requested, but llama-cpp-python has no usable GPU backend. "
                           "Install a compatible GPU build and runtime. CPU fallback is disabled.")


def load_tipo(model_path, gpu_layers=0, n_ctx=2048, n_threads=2, cache=True):
    """Borrow one cached model inside tipo_inference(), or own a private model.

    Internal cache=True callers must hold tipo_inference() from this call until
    generation finishes. A settings change closes the previous cached model
    before loading its replacement. Bare cached loads are rejected because an
    escaped raw handle cannot safely participate in eviction. cache=False
    requires no scope and gives the caller ownership of close().
    """
    if cache and not getattr(_inference_state, "depth", 0):
        raise RuntimeError("Cached load_tipo() requires a tipo_inference() scope "
                           "covering loading and generation; use cache=False "
                           "for a caller-owned model and close it explicitly.")
    if not model_path:
        return None

    if os.path.isdir(model_path):
        gguf_files = sorted(f for f in os.listdir(model_path)
                            if _is_model_file(f) and os.path.isfile(os.path.join(model_path, f)))
        if not gguf_files:
            return None
        selected = next((f for f in gguf_files if "q4" in f.lower()), gguf_files[0])
        model_path = os.path.join(model_path, selected)

    if not os.path.isfile(model_path):
        return None

    model_path = os.path.realpath(model_path)
    threads = max(1, int(n_threads))
    key = (model_path, int(gpu_layers), int(n_ctx), threads)
    with _cache_lock:
        if cache and _inference_state.key not in (None, key):
            raise RuntimeError("Cannot change cached TIPO settings while a model "
                               "is borrowed in the current tipo_inference() scope.")
        if cache and key in _model_cache:
            _inference_state.key = key
            return _model_cache[key]
        from llama_cpp import Llama

        if cache:
            # All cached users hold the same inference scope, so no earlier
            # borrower can still be generating. Remove before close(): a
            # failed close must never leave a possibly invalid cached object.
            while _model_cache:
                _, obsolete = _model_cache.popitem()
                obsolete.close()
        llm = Llama(model_path=model_path, n_gpu_layers=int(gpu_layers), n_ctx=int(n_ctx),
                    n_threads=threads, n_threads_batch=threads,
                    verbose=os.environ.get("RS_TIPO_VERBOSE", "") == "1")
        if cache:
            _model_cache[key] = llm
            _inference_state.key = key
        return llm


def _postprocess_with_kgen(tags_csv, ban_tags, sort_preset, *, drop_redundant=True):
    """Categorize + reorder tags using kgen's lightweight formatter."""
    import kgen.formatter as kf
    from kgen.formatter import seperate_tags, apply_format

    # Runtime override of the rating tag-list (no site-packages patching).
    kf.tag_lists["rating"] = set(RATING_TAGS)

    all_tags = [t.strip() for t in tags_csv.split(",") if t.strip()]
    tag_map = seperate_tags(all_tags)

    format_template = SORT_PRESETS.get(sort_preset, SORT_PRESETS[DEFAULT_SORT])
    formatted = apply_format(tag_map, format_template)

    formatted = _clean_special_tokens(formatted)
    formatted = _dedup(formatted)

    formatted = _filter_banned(formatted, ban_tags)

    if drop_redundant:
        formatted = _drop_redundant(formatted)
    return formatted.strip().strip(",").strip()


def _postprocess(tags, ban_tags, sort_preset, input_prompt="", *, drop_redundant=True):
    # Filtering and conflict handling still work when optional kgen is absent.
    tags = _filter_banned(_dedup(_clean_special_tokens(tags)), ban_tags)
    if sort_preset is not None:
        try:
            if not _ensure_formatter():
                raise ImportError("tipo-kgen is unavailable")
            tags = _postprocess_with_kgen(tags, ban_tags, sort_preset, drop_redundant=drop_redundant)
        except Exception as e:
            print(f"[TIPO] category sorting unavailable ({type(e).__name__}: {e}); "
                  "using basic tag cleanup")
    tags = _filter_banned(_dedup(tags), ban_tags)
    tags = _resolve_count_conflicts(tags, input_prompt)
    return _drop_redundant(tags) if drop_redundant else tags


def _avoid_hint(ban_tags):
    """Soft prompt hint so the model leaves banned tags out from the start
    (better backfill / tag count). Post-processing still removes them as the
    hard guarantee — small models don't reliably honor negative instructions."""
    bans = [b.strip() for b in (ban_tags or "").split(",") if b.strip()]
    if not bans:
        return ""
    return " Do not include these tags: " + ", ".join(bans) + "."


def expand_prompt(prompt, gguf_path, tag_length, ban_tags, temperature, seed,
                  sort_preset=DEFAULT_SORT, gpu_layers=0):
    """Expand a comma-separated tag prompt with a local GGUF model.

    On any failure (missing llama-cpp-python, model not found, inference or
    post-process error) the original prompt is returned unchanged.
    """
    if not prompt or not prompt.strip():
        return prompt

    try:
        if not _ensure_deps():
            print("[TIPO] dependencies unavailable; using original prompt")
            return prompt
        layers = 0 if os.environ.get("RS_TIPO_FORCE_CPU", "").lower() in ("1", "true", "yes", "on") else int(gpu_layers)
        if layers != 0:
            require_gpu_offload()
        resolved = _resolve_model_path(gguf_path)

        target = _TARGET_TAGS.get(tag_length, 45)
        # This is a TIPO tag model: the "Partial tags:" framing yields far
        # richer/on-topic output than a "Description:" framing, even for
        # Korean natural-language input. is_nl is used ONLY to decide whether
        # to keep the raw input in the result (below), not the framing.
        is_nl = _looks_like_nl(prompt)
        user_msg = (
            f"Partial tags: {prompt.strip()}\n"
            f"Output around {target} comma-separated English Danbooru tags."
            + _avoid_hint(ban_tags)
        )

        # Borrow under the same ownership used to evict/replace cached models.
        with tipo_inference():
            llm = load_tipo(resolved, layers)
            if llm is None:
                print(f"[TIPO] model not loaded (path: {resolved!r}); using original prompt")
                return prompt
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_TIPO},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=512,
                temperature=float(temperature),
                top_p=0.9,
                repeat_penalty=1.2,
                seed=int(seed) % (2 ** 31),
            )
        raw = resp["choices"][0]["message"]["content"]

        tags = _extract_tags(raw)
        if not tags:
            return prompt

        # For tag input keep the original tags up front; for natural-language
        # input drop the raw text so it doesn't pollute the tag output.
        combined = _dedup(tags if is_nl else prompt.strip() + ", " + tags)
        return _postprocess(combined, ban_tags, sort_preset, prompt)

    except Exception as e:
        print(f"[TIPO] expansion skipped ({type(e).__name__}: {e}); using original prompt")
        return prompt
