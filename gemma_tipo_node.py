import os

AUTO_MODEL = "(auto: download default)"


def _engines():
    try:
        from . import tipo_engine, tipo_vision_engine
    except ImportError:
        import tipo_engine
        import tipo_vision_engine
    return tipo_engine, tipo_vision_engine


def _has_image(image):
    if image is None:
        return False
    try:
        # ComfyUI IMAGE is a tensor/ndarray; empty/None -> text mode
        return getattr(image, "numel", lambda: len(image))() > 0
    except Exception:
        return image is not None


class GemmaTipoNode:
    """Gemma-4 TIPO: text/tags OR image -> rich category-sorted Danbooru
    tag set (TIPO-style: capture + plausible expansion).

    One unified node:
      - no image connected  -> text mode: `prompt` (Korean/English NL or
        tags) is expanded in-process via llama-cpp-python.
      - image connected      -> vision mode: the image is captioned via
        gemma-4 vision (llama.cpp mtmd) + the trained vision mmproj, on
        GPU when available (CUDA/Vulkan/Metal auto, CPU fallback).

    Model is picked from ComfyUI/models/gguf (or auto-download default).
    Same kgen post-processing either way. Any failure falls back safely
    (text -> original prompt; vision -> empty string).
    """

    @classmethod
    def INPUT_TYPES(cls):
        te, _ = _engines()
        try:
            files = sorted(f for f in os.listdir(te._gguf_dir())
                           if f.lower().endswith(".gguf"))
        except Exception:
            files = []
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Korean/English description or tags (ignored if an image is connected)",
                }),
                "model": ([AUTO_MODEL] + files, {"default": AUTO_MODEL}),
                "tag_length": (["very_short", "short", "long", "very_long"],
                               {"default": "long"}),
                "sort": (te.SORT_CHOICES, {"default": te.DEFAULT_SORT}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.1,
                                          "max": 1.5, "step": 0.05}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mmproj_path": ("STRING", {
                    "default": "",
                    "placeholder": "vision only; empty = auto-download trained vision mmproj"}),
                "gpu_layers": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "prompt"

    def run(self, prompt, model, tag_length, sort, temperature, ban_tags,
            seed, image=None, mmproj_path="", gpu_layers=0):
        te, ve = _engines()
        gguf = "" if model == AUTO_MODEL else os.path.join(
            te._gguf_dir(), model)

        if _has_image(image):  # vision mode
            return (ve.image_to_tags(image, gguf, mmproj_path, tag_length,
                                     ban_tags, temperature, seed, sort,
                                     gpu_layers),)
        # text mode
        if not prompt or not prompt.strip():
            return (prompt,)
        return (te.expand_prompt(prompt, gguf, tag_length, ban_tags,
                                 temperature, seed, sort),)
