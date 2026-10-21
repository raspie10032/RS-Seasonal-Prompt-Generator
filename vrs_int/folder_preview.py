"""Bounded, read-only folder check. No models, catalog writes or networking."""
import importlib.util
import os
from pathlib import Path
import time

from .folders import normalize_roots
from .generated import EXTENSIONS
from .generation_metadata import extract, prompt_tags
from .vision_assets import AUTO_VISION, local_vision_path


def sample_folder(root, *, entry_limit=256, image_limit=8, seconds=2):
    """Inspect a small prefix, never interpret it as a complete corpus count.

    Bound directory entries as well as image opens: a tree with no images must
    not cause an unbounded traversal. The time budget is cooperative; a single
    filesystem operation can take longer on a slow or disconnected mount.
    """
    from PIL import Image
    deadline = time.monotonic() + seconds
    pending, entries, sampled = [Path(root)], 0, 0
    readable, absent, errors = 0, 0, 0
    limited = False
    while pending and not limited:
        directory = pending.pop()
        with os.scandir(directory) as paths:
            for item in paths:
                if entries >= entry_limit or sampled >= image_limit or time.monotonic() >= deadline:
                    limited = True
                    break
                entries += 1
                if item.is_symlink():
                    continue
                if item.is_dir(follow_symlinks=False):
                    pending.append(Path(item.path))
                elif item.is_file(follow_symlinks=False) and Path(item.name).suffix.lower() in EXTENSIONS:
                    sampled += 1
                    try:
                        with Image.open(item.path) as image:
                            metadata = extract(image.info, item.name)
                        prompts = (metadata["positive"] + [c["prompt"] for c in metadata["characters"]]) if metadata else []
                        if prompts and prompt_tags(prompts)[0]:
                            readable += 1
                        else:
                            absent += 1
                    except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError):
                        errors += 1
    return dict(path=str(root), entries_checked=entries, images_checked=sampled,
                readable_metadata=readable, no_metadata=absent, read_errors=errors,
                limited=limited, complete=not limited)


def preview_folders(config):
    """Validate a proposed setup without starting collection or inference."""
    import folder_paths
    roots = config.get("roots", [])
    if not isinstance(roots, list) or any(not isinstance(p, str) for p in roots):
        raise ValueError("roots must be a list of server-side folder paths")
    roots = list(roots)
    if config.get("include_output", False):
        roots.append(str(folder_paths.get_output_directory()))
    roots = normalize_roots(roots)
    if len(roots) > 32:
        raise ValueError("Check at most 32 folders at a time")
    tagger = config.get("tagger", "none")
    if tagger not in ("none", "pixai"):
        raise ValueError("tagger must be none or pixai")
    for name in ("use_cpu", "resume_on_start", "include_output"):
        if name in config and type(config[name]) is not bool:
            raise ValueError(name + " must be boolean")
    batch = config.get("batch_size", 4)
    if type(batch) is not int or not 1 <= batch <= 32:
        raise ValueError("Batch size must be 1..32")
    model = config.get("model_name", AUTO_VISION)
    if not isinstance(model, str) or not model:
        raise ValueError("Select the native Gemma vision model")
    if model != AUTO_VISION:
        local_vision_path(model)
    missing = [name for name in ("transformers", "timm")
               if tagger == "pixai" and importlib.util.find_spec(name) is None]
    deadline = time.monotonic() + 2
    samples = [sample_folder(root, seconds=max(0, deadline - time.monotonic())) for root in roots]
    return dict(roots=roots, samples=samples,
                model_name=model, automatic_vision=model == AUTO_VISION,
                missing_packages=missing, can_start=not missing,
                offline=any(os.environ.get(key, "").lower() in ("1", "true", "yes", "on")
                            for key in ("RS_MODELS_OFFLINE", "HF_HUB_OFFLINE")),
                use_cpu=config.get("use_cpu", False))
