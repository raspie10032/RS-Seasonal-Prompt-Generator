import random


class GemmaTipoNode:
    """Convert a Korean/English natural-language (or tag) prompt into a
    Danbooru-style tag prompt using a local GGUF model (TIPO).

    Standalone node: feed it any text (typed, or wired from another node
    such as the Seasonal Fashion Prompt Generator) and it returns expanded,
    category-sorted tags. Dependencies auto-install on first use; any
    failure falls back to the input text unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Describe the outfit/scene in Korean or English, or pass tags",
                }),
                "tipo_gguf_path": ("STRING", {
                    "default": "",
                    "placeholder": "leave empty to auto-download from HF, or a local .gguf file/folder path",
                }),
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

    def expand(self, prompt, tipo_gguf_path, tag_length, sort, temperature,
               ban_tags, seed):
        if not prompt or not prompt.strip():
            return (prompt,)
        try:
            from . import tipo_engine
        except ImportError:
            import tipo_engine
        return (tipo_engine.expand_prompt(prompt, tipo_gguf_path, tag_length,
                                          ban_tags, temperature, seed, sort),)
