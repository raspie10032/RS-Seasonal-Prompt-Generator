"""In-process TIPO prompt expansion.

Inference: a local GGUF model via llama-cpp-python (ported from
NAI-FaceDetailer's tipo.py chat-completion approach), so a custom
Gemma-4-E2B GGUF can be used as-is.

Post-processing: kgen.formatter (pure-Python, lightweight) — only the
light formatter is imported, never kgen.executor / kgen.models.

All heavy imports are lazy and every failure path falls back to the
original prompt so the node never hard-fails.
"""

import os
import re
import random
import threading

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
    return re.sub(r"<[^|][^>]*>", "", raw)


def _dedup(tags):
    seen, out = set(), []
    for t in tags.split(","):
        t = t.strip()
        key = re.sub(r"[():\d.]", "", t).strip().lower()
        if key and key not in seen:
            seen.add(key)
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

    return formatted.strip().strip(",").strip()


def expand_prompt(prompt, gguf_path, tag_length, ban_tags, temperature, seed,
                  sort_preset=DEFAULT_SORT):
    """Expand a comma-separated tag prompt with a local GGUF model.

    On any failure (missing llama-cpp-python, model not found, inference or
    post-process error) the original prompt is returned unchanged.
    """
    if not prompt or not prompt.strip():
        return prompt

    try:
        llm = load_tipo(gguf_path)
        if llm is None:
            print(f"[TIPO] model not loaded (path: {gguf_path!r}); using original prompt")
            return prompt

        target = _TARGET_TAGS.get(tag_length, 45)
        user_msg = (
            f"Partial tags: {prompt.strip()}\n"
            f"Output around {target} comma-separated Danbooru tags."
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

        # Keep the original prompt's tags up front, then post-process via kgen.
        combined = _dedup(prompt.strip() + ", " + tags)
        expanded = _postprocess_with_kgen(combined, ban_tags, sort_preset)
        return expanded if expanded else prompt

    except Exception as e:
        print(f"[TIPO] expansion skipped ({type(e).__name__}: {e}); using original prompt")
        return prompt
