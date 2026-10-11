"""Corrected Gemma vision-only adapter, independent of the experiment package.

Keeps the tested PIL/NF4/float-pixel/mean-pool recipe. Weights stay external and
are checked against their provenance file. No downloads or CPU fallback.
"""
import json
from pathlib import Path

from .import_snapshot import file_digest


class ImageEncoder:
    def __init__(self, directory):
        import torch
        from transformers import BitsAndBytesConfig, Gemma4VisionModel
        from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        directory = Path(directory)
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        for name, expected in provenance["files"].items():
            if Path(name).name != name or file_digest(directory / name) != expected:
                raise ValueError("Encoder file or digest mismatch")
        self.processor = Gemma4ImageProcessorPil.from_pretrained(directory, local_files_only=True)
        self.model = Gemma4VisionModel.from_pretrained(
            directory, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, llm_int8_skip_modules=[]), device_map={"": "cuda:0"})
        self.model.eval()

    def encode(self, image):
        import torch
        from PIL import Image
        with Image.open(image) as source:
            pixels = source.convert("RGB")
        inputs = self.processor(images=pixels, return_tensors="pt")
        with torch.inference_mode():
            features = self.model(pixel_values=inputs["pixel_values"].to("cuda", torch.bfloat16),
                                  pixel_position_ids=inputs["image_position_ids"].to("cuda")).last_hidden_state
            if not torch.isfinite(features).all():
                raise ValueError("Nonfinite image features")
            vector = features.float().reshape(-1, features.shape[-1]).mean(dim=0)
            norm = torch.linalg.vector_norm(vector)
            if norm <= 0:
                raise ValueError("Zero image vector")
            return vector / norm
