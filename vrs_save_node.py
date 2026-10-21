"""Save runtime prompts explicitly, with ComfyUI's normal local SaveImage."""


class VRSSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "positive": ("STRING", {"forceInput": True, "tooltip":
                "Connect the exact final string used for image generation (after TIPO/wildcards/concatenation)."}),
            "filename_prefix": ("STRING", {"default": "ComfyUI"}),
        }, "optional": {
            "negative": ("STRING", {"default": "", "multiline": True}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save"
    CATEGORY = "prompt/VRS"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Save images locally with the actual positive/negative strings for folder memory. "
        "Connect the same final text as the generator, including TIPO/wildcard results. "
        "Uses ComfyUI SaveImage; --disable-metadata is respected. "
        "Prompts remain embedded in saved images: remove metadata before sharing if private.")

    def save(self, images, positive, filename_prefix="ComfyUI", negative="", prompt=None, extra_pnginfo=None):
        if not isinstance(positive, str) or not positive.strip() or not isinstance(negative, str):
            raise ValueError("Connect a nonempty final positive prompt and a text negative prompt")
        from nodes import SaveImage
        # Per-save copy: multiple outputs in one graph cannot overwrite each
        # other's prompts or leak runtime text into a different save branch.
        info = dict(extra_pnginfo or {})
        info["rs_vrs_prompt"] = {"version": 1, "positive": positive, "negative": negative}
        saved = SaveImage().save_images(images, filename_prefix, prompt, info)
        return dict(saved, result=(images,))
