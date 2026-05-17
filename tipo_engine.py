"""In-process TIPO prompt expansion.

Inference: a local GGUF model via llama-cpp-python (ported from
NAI-FaceDetailer's tipo.py chat-completion approach), so a custom
Gemma-4-E2B GGUF can be used as-is.

Post-processing: kgen.formatter (pure-Python, lightweight) — only the
light formatter is imported, never kgen.executor / kgen.models.

Dependencies (llama-cpp-python, tipo-kgen) are auto-installed on first
TIPO use only. All heavy imports are lazy and every failure path falls
back to the original prompt so the node never hard-fails.
"""

import os
import re
import sys
import time
import random
import threading
import importlib
import subprocess
import urllib.request

# Dependencies are installed on first actual TIPO use only (not at node import),
# so the base node stays zero-dependency and ComfyUI startup is never blocked.
# llama-cpp-python below this version cannot load the gemma4 architecture.
_MIN_LLAMA = (0, 3, 23)
_DEPS_ATTEMPTED = False
_deps_lock = threading.Lock()


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
    try:
        import llama_cpp
        return _ver_tuple(getattr(llama_cpp, "__version__", ""))
    except Exception:
        return None


def _installed_llama_version():
    """Version on disk (via pip), independent of what's imported in-process."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", "llama-cpp-python"],
            text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.lower().startswith("version:"):
                return _ver_tuple(line.split(":", 1)[1])
    except Exception:
        pass
    return None


def _pip_install(args):
    print(f"[TIPO] auto-installing: {' '.join(args)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def _restart_needed():
    """True if a gemma4-capable llama-cpp-python is on disk but an older one
    is already imported in this process (C-extension can't be hot-reloaded)."""
    ins = _installed_llama_version()
    iv = _imported_llama_version()
    return bool(ins and ins >= _MIN_LLAMA and (iv is None or iv < _MIN_LLAMA)
                and "llama_cpp" in sys.modules)


def _ensure_deps():
    """On first TIPO use: install tipo-kgen and a gemma4-capable
    llama-cpp-python (>= %s). Existing too-old llama-cpp-python is upgraded.
    """ % ".".join(map(str, _MIN_LLAMA))
    global _DEPS_ATTEMPTED

    def _ok():
        iv = _imported_llama_version()
        return _have("kgen.formatter") and iv is not None and iv >= _MIN_LLAMA

    if _ok():
        return True
    with _deps_lock:
        if _ok():
            return True
        if _DEPS_ATTEMPTED:
            if _restart_needed():
                print("[TIPO] llama-cpp-python was upgraded; RESTART ComfyUI "
                      "to load the gemma4-capable build, then run again.")
            return False
        _DEPS_ATTEMPTED = True
        try:
            if not _have("kgen.formatter"):
                _pip_install(["tipo-kgen"])
            ins = _installed_llama_version()
            if ins is None or ins < _MIN_LLAMA:
                # prebuilt CPU wheel index -> no native build required
                _pip_install([
                    "llama-cpp-python>=%s" % ".".join(map(str, _MIN_LLAMA)),
                    "--upgrade",
                    "--extra-index-url",
                    "https://abetlen.github.io/llama-cpp-python/whl/cpu",
                    "--prefer-binary",
                ])
        except Exception as e:
            print(f"[TIPO] auto-install failed ({type(e).__name__}: {e})")
        importlib.invalidate_caches()
        if _restart_needed():
            print("[TIPO] llama-cpp-python upgraded to a gemma4-capable "
                  "version; RESTART ComfyUI, then run again.")
            return False
        return _ok()

# Default model auto-downloaded from HuggingFace when no local path is given.
HF_REPO = "raspie/gemma4-tipo-ko-gguf"
HF_FILE = "gemma4-tipo-ko-Q4_K_M.gguf"


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
        return gguf_path

    target_dir = _gguf_dir()
    target = os.path.join(target_dir, HF_FILE)
    if os.path.exists(target):
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
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "rs-seasonal-tipo"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        pbar = _progress_bar(total) if total else None
        done = 0
        last = 0.0
        try:
            with open(part, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if pbar is not None:
                        try:
                            pbar.update_absolute(done, total)
                        except Exception:
                            pbar = None
                    now = time.time()
                    if total and now - last > 2:
                        last = now
                        print(f"[TIPO] downloading model: "
                              f"{done * 100 // total}% "
                              f"({done >> 20}/{total >> 20} MB)")
        except BaseException:
            if os.path.exists(part):
                os.remove(part)
            raise
    os.replace(part, dest)
    print("[TIPO] model download complete")


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
}
DEFAULT_SORT = "danbooru"

_model_cache = {}
_cache_lock = threading.Lock()


def _clean_special_tokens(raw):
    # TIPO category tokens (<|general|> etc.) -> comma so adjacent tags don't
    # glue together; any other angle-bracket token -> dropped.
    raw = re.sub(r"<\|[^|>]*\|>", ", ", raw)
    raw = re.sub(r"<[^>]*>?", "", raw)  # also strips unterminated "<..." tails
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
        key = re.sub(r"[():\d.]", "", t).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return ", ".join(out)


def _drop_redundant(csv):
    """Drop a general tag when a more specific tag with the same head noun
    is present, to save tokens: e.g. with "white skirt" present, "skirt" is
    redundant and removed. Uses a suffix rule (the longer tag must end with
    " " + the shorter tag) so distinct tags like "dress" vs "dress shirt"
    are NOT collapsed."""
    tags = [t.strip() for t in csv.split(",") if t.strip()]
    low = [t.lower() for t in tags]
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
    t = tag.strip().lower()
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
    input_low = [t.strip().lower() for t in input_prompt.split(",")]
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
                if m.lower() == it:
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


def load_tipo(model_path):
    """Load and cache a GGUF model (CPU, n_gpu_layers=0 for max compatibility)."""
    if not model_path:
        return None

    if os.path.isdir(model_path):
        gguf_files = [f for f in os.listdir(model_path) if f.endswith(".gguf")]
        if not gguf_files:
            return None
        selected = next((f for f in gguf_files if "q4" in f.lower()), gguf_files[0])
        model_path = os.path.join(model_path, selected)

    if not os.path.exists(model_path):
        return None

    with _cache_lock:
        if model_path in _model_cache:
            return _model_cache[model_path]
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=2048, verbose=False)
        _model_cache[model_path] = llm
        return llm


def _postprocess_with_kgen(tags_csv, ban_tags, sort_preset):
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

    banned = {b.strip().lower() for b in ban_tags.split(",") if b.strip()}
    if banned:
        formatted = ", ".join(
            t.strip()
            for t in formatted.split(",")
            if t.strip() and t.strip().lower() not in banned
        )

    formatted = _drop_redundant(formatted)
    return formatted.strip().strip(",").strip()


def expand_prompt(prompt, gguf_path, tag_length, ban_tags, temperature, seed,
                  sort_preset=DEFAULT_SORT):
    """Expand a comma-separated tag prompt with a local GGUF model.

    On any failure (missing llama-cpp-python, model not found, inference or
    post-process error) the original prompt is returned unchanged.
    """
    if not prompt or not prompt.strip():
        return prompt

    if not _ensure_deps():
        print("[TIPO] dependencies unavailable; using original prompt")
        return prompt

    try:
        resolved = _resolve_model_path(gguf_path)
        llm = load_tipo(resolved)
        if llm is None:
            print(f"[TIPO] model not loaded (path: {resolved!r}); using original prompt")
            return prompt

        target = _TARGET_TAGS.get(tag_length, 45)
        # This is a TIPO tag model: the "Partial tags:" framing yields far
        # richer/on-topic output than a "Description:" framing, even for
        # Korean natural-language input. is_nl is used ONLY to decide whether
        # to keep the raw input in the result (below), not the framing.
        is_nl = _looks_like_nl(prompt)
        user_msg = (
            f"Partial tags: {prompt.strip()}\n"
            f"Output around {target} comma-separated English Danbooru tags."
        )

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

        tags_raw = _clean_special_tokens(raw).strip()
        parsed = [
            line.strip()
            for line in tags_raw.splitlines()
            if "," in line or len(line.split()) > 2
        ]
        tags = ", ".join(parsed) if parsed else tags_raw

        # For tag input keep the original tags up front; for natural-language
        # input drop the raw text so it doesn't pollute the tag output.
        combined = _dedup(tags if is_nl else prompt.strip() + ", " + tags)
        expanded = _postprocess_with_kgen(combined, ban_tags, sort_preset)
        expanded = _resolve_count_conflicts(expanded, prompt)
        return expanded if expanded else prompt

    except Exception as e:
        print(f"[TIPO] expansion skipped ({type(e).__name__}: {e}); using original prompt")
        return prompt
