"""Corrected Gemma vision-only adapter, independent of the experiment package.

Keeps the tested PIL/NF4/float-pixel/mean-pool recipe. Weights stay external and
are checked against their provenance file. No downloads or CPU fallback.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
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
        return self.encode_many([image])[0]

    def encode_many(self, images):
        """Pool each image separately, including mixed aspect ratios in a batch."""
        import torch
        from PIL import Image
        if not images:
            raise ValueError("Empty image batch")
        def prepare(image):
            with Image.open(image) as source:
                pixels = source.convert("RGB")
            return self.processor(images=pixels, return_tensors="pt")
        workers = max(1, min(len(images), int(os.environ.get("VRS_PREPROCESS_WORKERS", "1"))))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(prepare, images))
            inputs = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
        else:
            pixels = []
            for image in images:
                with Image.open(image) as source:
                    pixels.append(source.convert("RGB"))
            inputs = self.processor(images=pixels, return_tensors="pt")
        with torch.inference_mode():
            features = self.model(pixel_values=inputs["pixel_values"].to("cuda", torch.bfloat16),
                                  pixel_position_ids=inputs["image_position_ids"].to("cuda")).last_hidden_state
            if not torch.isfinite(features).all():
                raise ValueError("Nonfinite image features")
            # Gemma strips pooled padding and flattens images into one token
            # stream. Different aspect ratios retain different token counts;
            # reshaping into equal pieces either fails or mixes two images.
            positions = inputs["image_position_ids"].clamp(min=0)
            kernel = self.model.config.pooling_kernel_size
            if kernel <= 1:
                raise ValueError("Unsupported Gemma pooling layout")
            max_x = positions[..., 0].max(dim=-1, keepdim=True).values + 1
            cells = positions // kernel
            cell_ids = cells[..., 0] + (max_x // kernel) * cells[..., 1]
            counts = [int(torch.unique(row).numel()) for row in cell_ids]
            if features.ndim != 2 or sum(counts) != features.shape[0]:
                raise ValueError("Gemma image token boundaries differ from processor positions")
            vectors = torch.stack([part.float().mean(dim=0) for part in features.split(counts)])
            norm = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
            if (norm <= 0).any():
                raise ValueError("Zero image vector")
            return vectors / norm
