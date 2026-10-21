"""TIPO-style expansion through ComfyUI's own Gemma 4 PyTorch runtime.

The base CLIP comes from ComfyUI's Load CLIP. Optional built-in adapters download
on first use; ComfyUI owns model patching and GPU allocation.
"""
import math
import re
import sys

try:
    from .tipo_status import node_status, report_status
except ImportError:
    from tipo_status import node_status, report_status

try:
    from .tipo_prompt import build_user_prompt
    from .tipo_native_lora import CHOICES, REQUIREMENTS, apply_tipo_lora
    from .tipo_gguf import SOURCES, WARNING, resolve_clip
except ImportError:
    from tipo_prompt import build_user_prompt
    from tipo_native_lora import CHOICES, REQUIREMENTS, apply_tipo_lora
    from tipo_gguf import SOURCES, WARNING, resolve_clip


def _engine():
    try:
        from . import tipo_engine
    except ImportError:
        import tipo_engine
    return tipo_engine


def build_prompt(prompt, tag_length, ban_tags, has_image):
    return build_user_prompt(prompt, tag_length, ban_tags, has_image)


def clean_tags(raw, ban_tags, prompt):
    te = _engine()
    text = te._strip_reasoning(raw).strip()
    text = re.sub(r"^```[^\n]*\n|\n?```\s*$", "", text)
    text = re.sub(r"^(?:tags|prompt)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s+)", "", text, flags=re.M)
    text = re.sub(r"\s*\n\s*", ", ", text)
    # None skips optional kgen category sorting; no runtime package installs.
    # A suffix is not proof of redundancy: a longer generated title can
    # contain an independently valid, user-supplied copyright or artist tag.
    return te._postprocess(text, ban_tags, None, input_prompt=prompt, drop_redundant=False)


def validate_input(prompt, tag_length, image):
    """Shared input checks; do not load a model for an empty request."""
    if tag_length not in _engine()._TARGET_TAGS:
        raise ValueError("Unknown tag_length")
    if image is not None:
        shape = getattr(image, "shape", ())
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3 or min(shape[1:3]) < 1:
            raise ValueError("Gemma native nodes accept one RGB IMAGE (batch=1). Split image batches before this node.")
    return bool(prompt.strip()) or image is not None


def validate_repetition_penalty(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 1.0 <= value <= 2.0):
        raise ValueError("repetition_penalty must be a finite number from 1.0 to 2.0")


def generate_text(clip, text, image, max_tokens, temperature, seed, repetition_penalty=1.05):
    """Use ComfyUI's allocator/template and propagate errors without retry."""
    validate_repetition_penalty(repetition_penalty)
    if not all(callable(getattr(clip, method, None)) for method in ("tokenize", "generate", "decode")):
        raise RuntimeError("ComfyUI is missing native text generation. Update ComfyUI and load a Gemma 4 E2B safetensors with Load CLIP.")
    tokens = clip.tokenize(
        text, image=image, min_length=1, thinking=False, skip_template=False,
    )
    if not isinstance(tokens, dict) or set(tokens) != {"gemma4"}:
        raise ValueError("Expected a Gemma 4 CLIP input. Use a native Gemma 4 E2B Load CLIP or the built-in heretic_ara_q4 source; ordinary SD/SDXL CLIP is not supported.")
    generated = clip.generate(
        tokens, do_sample=temperature > 0, max_length=max_tokens,
        temperature=max(temperature, 0.01), top_k=64, top_p=0.95,
        min_p=0.05, repetition_penalty=repetition_penalty, seed=seed,
    )
    # ComfyUI normally clears captured graphs after a node. This node can
    # generate twice with different prompt/KV lengths: do the same cleanup at
    # the generation boundary so the second call cannot replay the first KV
    # cache. Older ComfyUI releases without this module need no such cleanup.
    prefetch = sys.modules.get("comfy.model_prefetch")
    cleanup = getattr(prefetch, "cleanup_prefetch_queues", None)
    if callable(cleanup):
        cleanup()
    raw = clip.decode(generated)
    if not raw.strip():
        raise RuntimeError("Gemma returned an empty response. Check the model/tokenizer pairing and generation settings.")
    if not _engine()._strip_reasoning(raw).strip():
        raise RuntimeError("Gemma returned only reasoning. Increase max_tokens or check the model/template pairing.")
    return raw


class GemmaTipoNative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {"unique_id": "UNIQUE_ID"},
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "tag_length": (["very_short", "short", "long", "very_long"], {"default": "long"}),
                "max_tokens": ("INT", {"default": 384, "min": 16, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05}),
                "ban_tags": ("STRING", {"default": "", "multiline": True,
                                          "tooltip": "Comma-separated English tags. Removed from tag output after generation; raw output is unfiltered."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {"clip": ("CLIP", {"tooltip": "For connected_clip: native Gemma 4 E2B Load CLIP."}),
                "image": ("IMAGE",), "tipo_lora": (list(CHOICES), {
                "default": "none", "tooltip": REQUIREMENTS}),
                "model_source": (list(SOURCES), {"default": "connected_clip", "tooltip": WARNING}),
                "gguf_path": ("STRING", {"default": "", "tooltip": "Optional local Gemma 4 E2B GGUF; blank downloads pinned Heretic ARA Q4."}),
                "mmproj_path": ("STRING", {"default": "", "tooltip": "Optional matching FP16 vision GGUF; only used with IMAGE."}),
                "tokenizer_path": ("STRING", {"default": "", "tooltip": "Optional matching tokenizer.json; blank downloads the pinned original."}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Tag sampling repetition penalty. Current ComfyUI ignores it at temperature=0; use a positive temperature. 1.0 disables it; larger values can also omit useful tags or names. Existing workflows retain 1.05."})},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "raw_output")
    FUNCTION = "run"
    CATEGORY = "prompt"
    DESCRIPTION = (
        "Gemma 4 tag expansion with ComfyUI's native PyTorch/model management. "
        "Choose built-in Q4 or connect a Gemma 4 Load CLIP; optionally connect one image. "
        "Built-in Q4 downloads its base on first use; no llama.cpp. " + REQUIREMENTS
    )

    @node_status
    def run(self, clip=None, prompt="", tag_length="long", max_tokens=384, temperature=0.5,
            ban_tags="", seed=0, image=None, tipo_lora="none", model_source="connected_clip",
            gguf_path="", mmproj_path="", tokenizer_path="", repetition_penalty=1.05, unique_id=None):
        validate_repetition_penalty(repetition_penalty)
        if not validate_input(prompt, tag_length, image):
            return ("", "")
        report_status(unique_id, "preparing-models")
        clip = resolve_clip(clip, model_source, image, gguf_path, mmproj_path, tokenizer_path)
        clip = apply_tipo_lora(clip, tipo_lora)
        report_status(unique_id, "generating-tags")
        raw = generate_text(clip, build_prompt(prompt, tag_length, ban_tags, image is not None),
                            image, max_tokens, temperature, seed, repetition_penalty=repetition_penalty)
        tags = clean_tags(raw, ban_tags, prompt)
        return (tags, raw)
