import os
import random

AUTO_MODEL = "(auto: download default)"


def _tipo_engine():
    try:
        from . import tipo_engine
    except ImportError:
        import tipo_engine
    return tipo_engine


class GemmaTipoNode:
    """Convert a Korean/English natural-language (or tag) prompt into a
    Danbooru-style tag prompt using a local GGUF model (TIPO).

    The model is chosen from a dropdown of *.gguf files found in
    ComfyUI/models/gguf. The first entry auto-downloads the default model
    there on first use. Dependencies auto-install on first run; any failure
    falls back to the input text unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        gdir = _tipo_engine()._gguf_dir()
        try:
            files = sorted(f for f in os.listdir(gdir)
                           if f.lower().endswith(".gguf"))
        except Exception:
            files = []
        model_choices = [AUTO_MODEL] + files
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Describe the outfit/scene in Korean or English, or pass tags",
                }),
                "model": (model_choices, {"default": AUTO_MODEL}),
                "tag_length": (["very_short", "short", "long", "very_long"],
                               {"default": "long"}),
                "sort": (["danbooru", "quality_first", "artist_first",
                          "general_only"], {"default": "danbooru"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.1,
                                          "max": 1.5, "step": 0.05}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": random.randint(0, 2 ** 31 - 1)}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "expand"
    CATEGORY = "prompt"

    def expand(self, prompt, model, tag_length, sort, temperature, ban_tags,
               seed):
        if not prompt or not prompt.strip():
            return (prompt,)
        engine = _tipo_engine()
        if model == AUTO_MODEL:
            gguf_path = ""  # empty -> resolve/download default into models/gguf
        else:
            gguf_path = os.path.join(engine._gguf_dir(), model)
        return (engine.expand_prompt(prompt, gguf_path, tag_length, ban_tags,
                                     temperature, seed, sort),)
