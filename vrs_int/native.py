"""ComfyUI-managed Gemma vision and immutable personal memory construction.

Imports ComfyUI/PyTorch lazily. No llama.cpp, Transformers, bitsandbytes,
subprocess, downloads, global device/thread changes, or CPU inference fallback.
Native FP16 vectors have a different contract from historical PIL/NF4 vectors.
"""
import hashlib
from pathlib import Path

from .store import Store, canonical, digest
from .tag_policy import POLICY_VERSION


def image_pixels(image):
    import torch
    if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1
            or image.shape[-1] != 3 or min(image.shape[1:3]) < 1):
        raise ValueError("VRS expects exactly one RGB ComfyUI IMAGE (batch 1, H, W, 3)")
    if not torch.isfinite(image).all():
        raise ValueError("Nonfinite IMAGE pixels")
    return (image.detach().clamp(0, 1) * 255).to(device="cpu", dtype=torch.uint8).contiguous()


def pixel_digest(image):
    pixels = image_pixels(image)
    return digest(canonical(list(pixels.shape)) + pixels.numpy().tobytes())


def load_image(path):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        pixels = np.array(ImageOps.exif_transpose(source).convert("RGB"), copy=True)
    return torch.from_numpy(pixels).float().div(255).unsqueeze(0)


class NativeVision:
    """Load only the E2B vision tensors from a native Gemma safetensors file."""
    def __init__(self, path):
        import torch
        import torchvision
        from safetensors import safe_open
        import comfy.model_management as mm
        import comfy.model_patcher as mp
        import comfy.ops
        from comfy.text_encoders import gemma4
        self.load_device = mm.get_torch_device()
        if self.load_device.type != "cuda":
            raise RuntimeError("VRS native currently requires CUDA; CPU fallback is disabled")
        path = Path(path)
        if path.suffix != ".safetensors":
            raise ValueError("Select native unquantized Gemma 4 E2B safetensors, not GGUF/mmproj")
        sd = {}
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith("vision_model."):
                    tensor = stream.get_tensor(key)
                    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                        raise ValueError("VRS vision requires unquantized floating-point tensors")
                    sd[key.removeprefix("vision_model.")] = tensor.to(torch.float16).contiguous()
        if not sd:
            raise ValueError("No native vision_model weights; use the converted Gemma 4 E2B file")
        # Fingerprint the actual FP16 tensors plus runtime implementation. Never
        # infer compatibility from a filename, model family, or vector dimension.
        h = hashlib.sha256()
        for key, tensor in sorted(sd.items()):
            h.update(canonical([key, list(tensor.shape), str(tensor.dtype)]))
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        h.update(Path(gemma4.__file__).read_bytes())
        h.update(canonical([str(torch.__version__), str(torchvision.__version__)]))
        self.encoder_contract = {
            "provenance_sha256": h.hexdigest(), "preprocess": "comfy-gemma4-rgb-u8-280-v1",
            "backend": "comfyui-pytorch-fp32-compute", "quantization": "fp16",
            "pooling": "mean-tokens-l2-v1"}
        self.model = gemma4.Gemma4VisionEncoder(
            gemma4.GEMMA4_VISION_CONFIG, dtype=torch.float16,
            device=mm.text_encoder_offload_device(), ops=comfy.ops.manual_cast).eval()
        self.model.load_state_dict(sd, strict=True)
        self.patcher = mp.CoreModelPatcher(self.model, load_device=self.load_device,
                                          offload_device=mm.text_encoder_offload_device())
        self.weights_bytes = sum(t.numel() * t.element_size() for t in sd.values())

    def contract(self, codec="f32le"):
        return {"encoder": dict(self.encoder_contract), "dimension": 768,
                "codec": codec, "tag_policy": POLICY_VERSION}

    def encode_tensor(self, image):
        import torch
        import torchvision.transforms.functional as tvf
        import comfy.model_management as mm
        from comfy.text_encoders.gemma4 import _get_aspect_ratio_preserving_size
        pixels = image_pixels(image).movedim(-1, 1).squeeze(0)
        height, width = pixels.shape[-2:]
        size = _get_aspect_ratio_preserving_size(height, width, 16, 280 * 9, 3)
        if size != (height, width):
            pixels = tvf.resize(pixels, list(size), interpolation=tvf.InterpolationMode.BICUBIC, antialias=True)
        mm.throw_exception_if_processing_interrupted()
        mm.load_model_gpu(self.patcher)
        with torch.inference_mode():
            # Match native Gemma preprocess_embed: FP16 resident weights,
            # float32 pixels/compute through manual_cast, including RMSNorm.
            features = self.model(pixels.unsqueeze(0).to(self.load_device).float().mul(1.0 / 255.0), max_soft_tokens=280)
            vector = features.float().mean(dim=1)[0]
            if not torch.isfinite(vector).all() or vector.norm() <= 0:
                raise ValueError("Invalid native vision features")
            return (vector / vector.norm()).cpu()

    def encode(self, path):
        image = load_image(path)
        self.last_pixel_sha256 = pixel_digest(image)
        return self.encode_tensor(image)


def build_native(encoder, manifest, images, destination, *, base=None, codec="f32le", label="native-personal"):
    from .build import local_entries, upsert_operations
    from .import_snapshot import file_digest
    import comfy.model_management as mm
    contract = encoder.contract(codec)
    store = Store(destination)
    if base and store.manifest(base)["contract"] != contract:
        raise ValueError("Incompatible base. Native FP16 and legacy PIL/NF4 need separate stores/snapshots")
    # Read-only lookup may use any existing store. A new native build must not
    # accidentally publish a replacement branch inside a legacy store.
    if not base and any((store.root / "snapshots").glob("*.json")):
        raise ValueError("Destination already contains snapshots; select base_snapshot or a new directory")
    manifest_sha = file_digest(manifest)

    def entries():
        for entry in local_entries(manifest, images, contract, encoder.encode):
            mm.throw_exception_if_processing_interrupted()
            entry["audit"].append({"native_pixel_sha256": encoder.last_pixel_sha256})
            yield entry
        if file_digest(manifest) != manifest_sha:
            raise ValueError("Input manifest changed during build")

    provenance = {"input_manifest_sha256": manifest_sha, "images_included": False,
                  "collection": "comfyui-user-selected-local-images", "automatic_downloads": False}
    if base:
        snapshot = store.update(base, upsert_operations(store, base, entries()),
                                label, contract, provenance)
    else:
        snapshot = store.build(entries(), contract, label, provenance, 256)
    return store.verify(snapshot)
