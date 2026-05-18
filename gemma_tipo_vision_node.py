import os

AUTO_MODEL = "(auto: download default)"


def _engines():
    try:
        from . import tipo_engine, tipo_vision_engine
    except ImportError:
        import tipo_engine
        import tipo_vision_engine
    return tipo_engine, tipo_vision_engine


class GemmaTipoVisionNode:
    """Image -> TIPO Danbooru tags via Gemma-4 vision (llama.cpp mtmd).

    Feed a ComfyUI IMAGE; the node runs a gemma-4-E2B text/merged GGUF
    (the TIPO model, chosen from ComfyUI/models/gguf) together with the
    base gemma-4-E2B mmproj (auto-downloaded) to caption the image into a
    rich, category-sorted Danbooru tag set (TIPO-style: capture + plausibly
    expand, not exact labeling). First use auto-downloads the prebuilt
    llama.cpp mtmd binary + mmproj. Any failure -> empty string.

    CPU by default (gpu_layers=0) so it won't fight other GPU work.
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
                "image": ("IMAGE",),
                "model": ([AUTO_MODEL] + files, {"default": AUTO_MODEL}),
                "tag_length": (["very_short", "short", "long", "very_long"],
                               {"default": "long"}),
                "sort": (["danbooru", "quality_first", "artist_first",
                          "general_only"], {"default": "danbooru"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1,
                                          "max": 1.5, "step": 0.05}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mmproj_path": ("STRING", {
                    "default": "",
                    "placeholder": "leave empty to auto-download base gemma-4-E2B mmproj"}),
                "gpu_layers": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "caption"
    CATEGORY = "prompt"

    def caption(self, image, model, tag_length, sort, temperature, ban_tags,
                seed, mmproj_path="", gpu_layers=0):
        te, ve = _engines()
        gguf = "" if model == AUTO_MODEL else os.path.join(
            te._gguf_dir(), model)
        return (ve.image_to_tags(image, gguf, mmproj_path, tag_length,
                                 ban_tags, temperature, seed, sort,
                                 gpu_layers),)
