"""TIPO-style expansion through ComfyUI's own Gemma 4 PyTorch runtime.

The base CLIP comes from ComfyUI's Load CLIP. Optional built-in adapters download
on first use; ComfyUI owns model patching and GPU allocation.
"""
import re
import sys

try:
    from .tipo_prompt import build_user_prompt
    from .tipo_native_lora import CHOICES, REQUIREMENTS, apply_tipo_lora
except ImportError:
    from tipo_prompt import build_user_prompt
    from tipo_native_lora import CHOICES, REQUIREMENTS, apply_tipo_lora


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
    # None skips optional kgen category sorting and its auto-install.
    return te._postprocess(text, ban_tags, None, input_prompt=prompt)


def validate_input(prompt, tag_length, image):
    """Shared input checks; do not load a model for an empty request."""
    if tag_length not in _engine()._TARGET_TAGS:
        raise ValueError("Unknown tag_length")
    if image is not None:
        shape = getattr(image, "shape", ())
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3 or min(shape[1:3]) < 1:
            raise ValueError("Gemma native nodes accept one RGB IMAGE (batch=1). Split image batches before this node.")
    return bool(prompt.strip()) or image is not None


def generate_text(clip, text, image, max_tokens, temperature, seed):
    """Use ComfyUI's allocator/template and propagate errors without retry."""
    if not all(callable(getattr(clip, method, None)) for method in ("tokenize", "generate", "decode")):
        raise RuntimeError("ComfyUI is missing native text generation. Update ComfyUI and load a Gemma 4 E2B safetensors with Load CLIP.")
    tokens = clip.tokenize(
        text, image=image, min_length=1, thinking=False, skip_template=False,
    )
    if not isinstance(tokens, dict) or set(tokens) != {"gemma4"}:
        raise ValueError("Expected a Gemma 4 CLIP input. Use Load CLIP with a ComfyUI-format Gemma 4 E2B model; ordinary SD/SDXL CLIP and GGUF are not supported.")
    generated = clip.generate(
        tokens, do_sample=temperature > 0, max_length=max_tokens,
        temperature=max(temperature, 0.01), top_k=64, top_p=0.95,
        min_p=0.05, repetition_penalty=1.05, seed=seed,
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
            "required": {
                "clip": ("CLIP", {"tooltip": "Load CLIP: ComfyUI-format Gemma 4 E2B safetensors; type stable_diffusion, device default."}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "tag_length": (["very_short", "short", "long", "very_long"], {"default": "long"}),
                "max_tokens": ("INT", {"default": 384, "min": 16, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05}),
                "ban_tags": ("STRING", {"default": "", "multiline": True,
                                          "tooltip": "Comma-separated English tags. Removed from tag output after generation; raw output is unfiltered."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {"image": ("IMAGE",), "tipo_lora": (list(CHOICES), {
                "default": "none", "tooltip": REQUIREMENTS})},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "raw_output")
    FUNCTION = "run"
    CATEGORY = "prompt"
    DESCRIPTION = (
        "Gemma 4 tag expansion with ComfyUI's native PyTorch/model management. "
        "Connect a Gemma 4 Load CLIP; optionally connect one image. "
        "No llama.cpp, mmproj or extra pip installation. " + REQUIREMENTS
    )

    def run(self, clip, prompt, tag_length, max_tokens, temperature, ban_tags, seed, image=None, tipo_lora="none"):
        if not validate_input(prompt, tag_length, image):
            return ("", "")
        clip = apply_tipo_lora(clip, tipo_lora)
        raw = generate_text(clip, build_prompt(prompt, tag_length, ban_tags, image is not None),
                            image, max_tokens, temperature, seed)
        tags = clean_tags(raw, ban_tags, prompt)
        return (tags, raw)
