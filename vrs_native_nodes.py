"""Native ComfyUI VRS: vision loader, personal builder, memory loader, recall."""
import json
from pathlib import Path

try:
    from .vrs_int.native import NativeVision, build_native, pixel_digest
    from .vrs_int.recommend import Recommender
    from .vrs_int.generation_state import build_state
    from .vrs_int.store import canonical, digest
    from .vrs_int.vision_assets import vision_choices, resolve_vision_path, VISION_HELP
except ImportError:
    from vrs_int.native import NativeVision, build_native, pixel_digest
    from vrs_int.recommend import Recommender
    from vrs_int.generation_state import build_state
    from vrs_int.store import canonical, digest
    from vrs_int.vision_assets import vision_choices, resolve_vision_path, VISION_HELP


class VRSNativeVisionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (vision_choices(), {"tooltip": VISION_HELP})},
                "optional": {"use_cpu": ("BOOLEAN", {"default": False,
                    "tooltip": "Explicit consent to CPU inference: slower and may affect other applications. Creates a separate encoder contract from GPU memory. Does not change global PyTorch thread settings."})}}

    RETURN_TYPES = ("VRS_ENCODER",)
    FUNCTION = "load"
    CATEGORY = "prompt/VRS"
    DESCRIPTION = VISION_HELP

    def load(self, model_name, use_cpu=False):
        path = resolve_vision_path(model_name, use_cpu=use_cpu)
        return (NativeVision(path, use_cpu=use_cpu),)


class VRSNativeMemoryLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"store_path": ("STRING", {"default": ""}),
                             "snapshot": ("STRING", {"default": "", "tooltip": "Exact 64-character snapshot hash; CURRENT is never followed implicitly."})}}

    RETURN_TYPES = ("VRS_MEMORY", "STRING")
    RETURN_NAMES = ("memory", "report")
    FUNCTION = "load"
    CATEGORY = "prompt/VRS"

    def load(self, store_path, snapshot):
        memory = Recommender(Path(store_path).expanduser(), snapshot.strip())
        return (memory, json.dumps(memory.manifest, ensure_ascii=False))


class VRSNativeBuild:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"encoder": ("VRS_ENCODER",),
            "manifest_path": ("STRING", {"default": ""}), "images_root": ("STRING", {"default": ""}),
            "store_path": ("STRING", {"default": ""}), "base_snapshot": ("STRING", {"default": ""}),
            "label": ("STRING", {"default": "native-personal"})}}

    RETURN_TYPES = ("VRS_MEMORY", "STRING", "STRING")
    RETURN_NAMES = ("memory", "snapshot", "report")
    FUNCTION = "build"
    CATEGORY = "prompt/VRS"
    OUTPUT_NODE = True
    DESCRIPTION = "Explicitly build/update immutable personal experience from local image/tag JSONL. Use a separate build workflow; retrieval uses Memory Loader."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Explicit builder runs once per submitted build workflow, never a
        # hidden write by the read-only recommendation/generation workflow.
        return float("nan")

    def build(self, encoder, manifest_path, images_root, store_path, base_snapshot="", label="native-personal"):
        if not all(v.strip() for v in (manifest_path, images_root, store_path)):
            raise ValueError("Manifest, image root and output store are required")
        report = build_native(encoder, Path(manifest_path).expanduser(), Path(images_root).expanduser(),
                              Path(store_path).expanduser(), base=base_snapshot.strip() or None, label=label)
        memory = Recommender(Path(store_path).expanduser(), report["snapshot"])
        value = json.dumps(report, ensure_ascii=False)
        return {"ui": {"text": [value]}, "result": (memory, report["snapshot"], value)}


class VRSNativeRecall:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"encoder": ("VRS_ENCODER",), "memory": ("VRS_MEMORY",),
            "image": ("IMAGE",), "prompt": ("STRING", {"multiline": True, "default": ""}),
            "keep_tags": ("STRING", {"default": "", "tooltip": "Explicit names/elements to preserve, comma-separated."}),
            "target_tags": ("STRING", {"default": ""}), "exclude_tags": ("STRING", {"default": ""}),
            "ratings": ("STRING", {"default": "g,s,q,e,u", "tooltip": "Allowed local-source ratings; u is unrated generated artwork."}),
            "neighbors": ("INT", {"default": 32, "min": 1, "max": 256}),
            "limit": ("INT", {"default": 64, "min": 1, "max": 100})}}

    # Append only: saved workflows retain their original output slot indices.
    RETURN_TYPES = ("VRS_STATE", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("vrs_state", "candidate_tags", "report", "request", "image")
    FUNCTION = "recall"
    CATEGORY = "prompt/VRS"
    DESCRIPTION = "Read-only image retrieval. Candidate names are not automatic identity claims; select names in keep_tags. Connect vrs_state, request and image outputs to Gemma Prompt Expander Native to avoid duplicate/mismatched inputs."

    def recall(self, encoder, memory, image, prompt, keep_tags="", target_tags="", exclude_tags="",
               ratings="g,s,q,e,u", neighbors=32, limit=64):
        import comfy.model_management as mm
        if memory.manifest["contract"] != encoder.contract(memory.manifest["contract"]["codec"]):
            raise ValueError("Incompatible VRS encoder. Legacy PIL/NF4 memories must be re-encoded into a separate native store")
        if not prompt.strip():
            raise ValueError("Enter the same creative request here and in Gemma Prompt Expander")
        split = lambda text: [s.strip() for s in text.split(",") if s.strip()]
        # Validate controls before any GPU work.
        build_state({"snapshot": memory.snapshot}, prompt, keep=split(keep_tags),
                    target=split(target_tags), exclude=split(exclude_tags))
        ratings = split(ratings)
        if not ratings or any(r not in ("g", "s", "q", "e", "u") for r in ratings):
            raise ValueError("ratings must list g,s,q,e,u or a nonempty subset")
        sha = pixel_digest(image)
        vector = encoder.encode_tensor(image)
        result = memory.recommend(vector, None, neighbors=neighbors, limit=limit, ratings=ratings,
            device=encoder.load_device, pixel_sha256=sha, interrupt=mm.throw_exception_if_processing_interrupted)
        state = build_state(result, prompt, keep=split(keep_tags), target=split(target_tags), exclude=split(exclude_tags))
        state["native_query"] = {"pixel_sha256": sha,
            "encoder_contract_sha256": digest(canonical(encoder.encoder_contract))}
        report = {"state": state, "recommendation": result,
                  "query_pixel_sha256": sha, "query_file_sha256": None,
                  "self_exclusion": "Decoded RGB pixel hash; excludes native-build copies with identical pixels, not crops/near-duplicates.",
                  "weights_bytes": encoder.weights_bytes}
        return (state, ", ".join(c["tag"] for c in result["candidates"]), json.dumps(report, ensure_ascii=False), prompt, image)
