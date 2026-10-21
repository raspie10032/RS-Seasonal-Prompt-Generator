"""Explicit optional PixAI fallback; pinned GET-only assets, ComfyUI-owned GPU."""
from pathlib import Path

PIXAI = {
    "repo": "pixai-labs/pixai-tagger-v1.0",
    "revision": "9fe10addf9326e292da8a85a98ea74cd91b41771",
    "files": {
        "config.json": (693918, "f8a19b38661c37dc5fd519f2137be70f6f897972021b4ebc17b7d35a0e27b9dd"),
        "preprocessor_config.json": (600, "20451575e627950993de7b00641d5b263eceb3e7436e4ad518fea2411b60e49d"),
        "tagger_pipeline.py": (43696, "3431279fdfff234f62664999fc1fe6bb20f1e8756f7435d0e165ebd4456f7fcd"),
        "model.safetensors": (1945425796, "f29e475205cbcbc25b52a075840c7809d6215be15f45375113ba09c49bf90292"),
    },
}


def resolve_tagger(interrupt, *, use_cpu=False):
    """No pip, tokens, image requests, or online trust_remote_code resolution."""
    import importlib.util
    import folder_paths
    import comfy.model_management as mm
    import comfy.utils
    from .torch_device import require_accelerator
    require_accelerator(use_cpu=use_cpu)
    missing = [name for name in ("transformers", "timm") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Optional PixAI dependencies missing: " + ", ".join(missing)
                           + ". Install requirements-tagger.txt using ComfyUI's Python, then restart. "
                             "Stop the collection and disable PixAI to use metadata only.")
    try:
        from ..tipo_native_lora import ensure_files, _check_file
    except ImportError:
        from tipo_native_lora import ensure_files, _check_file
    progress = comfy.utils.ProgressBar(1)
    def update(current, total):
        mm.throw_exception_if_processing_interrupted()
        interrupt()
        progress.update_absolute(current, total)
    root = Path(folder_paths.models_dir)
    target = root / "rs_vrs_tagger" / PIXAI["revision"]
    # Existing offline CLI installs use HF snapshot symlinks. Verify their
    # resolved regular files before executing the pinned local custom code.
    cached = root / "rs_wd_tagger" / "models--pixai-labs--pixai-tagger-v1.0" / "snapshots" / PIXAI["revision"]
    if not target.exists() and cached.is_dir():
        for name, expected in PIXAI["files"].items():
            update(0, expected[0])
            _check_file((cached / name).resolve(strict=True), expected)
            update(expected[0], expected[0])
        return cached
    return ensure_files(PIXAI, target, update)


class FolderTagger:
    def __init__(self, directory, *, use_cpu=False):
        import torch
        import comfy.model_management as mm
        import comfy.model_patcher as mp
        from .generated_tagger import LocalPixAI
        from .torch_device import require_accelerator
        device = require_accelerator(use_cpu=use_cpu)
        self.tagger = LocalPixAI(directory, workers=1, device="cpu")
        self.tagger.device = device
        # Transformers exposes a read-only .device property. ComfyUI writes
        # model.device when moving/offloading, so manage a plain module shell.
        class ManagedTagger(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, batch):
                return self.model(batch)
        self.tagger.model = ManagedTagger(self.tagger.model).eval()
        self.patcher = mp.CoreModelPatcher(self.tagger.model, load_device=self.tagger.device,
                                          offload_device=device if use_cpu else mm.text_encoder_offload_device())

    def predict(self, path, sha):
        import comfy.model_management as mm
        from .generated_tagger import prediction_record
        batch = self.tagger.prepare_batch([path])
        mm.throw_exception_if_processing_interrupted()
        mm.load_model_gpu(self.patcher)
        scores = self.tagger.score_prepared(batch)[0]
        return prediction_record(path, sha, self.tagger.labels, scores, self.tagger.provenance)
