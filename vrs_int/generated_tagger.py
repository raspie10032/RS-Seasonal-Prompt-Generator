"""Local PixAI (default) or WD fallback for unreadable generation prompts.

Keeps predictions distinct from original generation intent. No downloads,
workflow execution, image edits, or verified artist attribution/rating labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import csv
from itertools import islice
import json
import os
from pathlib import Path
import sqlite3
import time

from .generated import bind_config, bounded_score, compact, open_catalog, write_report
from .generation_metadata import VERSION, extract, make_record
from .import_snapshot import file_digest
from .tag_policy import canonical


class LocalWD:
    def __init__(self, directory, workers=4):
        import timm
        import torch
        from safetensors.torch import load_file
        from timm.data import create_transform, resolve_data_config
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        self.provenance = {"family": "wd", "directory_name": directory.name,
                           "files": {name: file_digest(directory / name) for name in
                                     ("config.json", "selected_tags.csv", "model.safetensors")},
                           "precision": "fp32-weights-bf16-autocast", "preprocess": "white-square-bgr-timm-config"}
        self.model = timm.create_model(config["architecture"], pretrained=False,
                                      num_classes=config["num_classes"], **config.get("model_args", {}))
        self.model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
        self.transform = create_transform(**resolve_data_config(config["pretrained_cfg"], model=self.model), is_training=False)
        with (directory / "selected_tags.csv").open(newline="", encoding="utf-8") as stream:
            self.labels = list(csv.DictReader(stream))
        if len(self.labels) != config["num_classes"]:
            raise ValueError("WD vocabulary/model size mismatch")
        self.model.eval().to("cuda")
        self.workers = workers

    def score(self, paths):
        import torch
        from PIL import Image
        def prepare(path):
            with Image.open(path) as source:
                rgba = source.convert("RGBA")
            rgb = Image.new("RGBA", rgba.size, "white")
            rgb.alpha_composite(rgba)
            rgb = rgb.convert("RGB")
            side = max(rgb.size)
            square = Image.new("RGB", (side, side), "white")
            square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
            return self.transform(square)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            batch = torch.stack(list(pool.map(prepare, paths)))[:, [2, 1, 0]].to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            scores = self.model(batch).float().sigmoid()
        if scores.shape != (len(paths), len(self.labels)) or not torch.isfinite(scores).all():
            raise ValueError("Invalid WD scores")
        return scores.cpu().tolist()


class LocalPixAI:
    """Use the installed, pinned PixAI code and its exact image processor."""
    CODE_SHA256 = "3431279fdfff234f62664999fc1fe6bb20f1e8756f7435d0e165ebd4456f7fcd"

    def __init__(self, directory, workers=4, device="cuda"):
        from transformers import AutoImageProcessor, AutoModel
        directory = Path(directory)
        if file_digest(directory / "tagger_pipeline.py") != self.CODE_SHA256:
            raise ValueError("PixAI custom code differs from the reviewed revision")
        config = json.loads((directory / "config.json").read_text())
        self.thresholds = config["category_best_threshold"]
        self.provenance = {"family": "pixai", "model": "pixai-labs/pixai-tagger-v1.0",
                           "revision": directory.name, "files": {name: file_digest(directory / name)
                           for name in ("config.json", "preprocessor_config.json", "tagger_pipeline.py", "model.safetensors")},
                           "precision": "fp32-weights-bf16-autocast-fp32-sigmoid",
                           "preprocess": "official-RescalePadProcessor-1008", "thresholds": self.thresholds,
                           "artist_field_semantics": "style reference candidates, not author identification"}
        self.labels = []
        for kind, count in config["tags_split"]:
            start = len(self.labels)
            self.labels.extend({"name": name, "category": kind} for name in config["tags"][start:start + count])
        if len(self.labels) != config["num_classes"] or len(self.labels) != len(config["tags"]):
            raise ValueError("PixAI category/vocabulary mismatch")
        self.processor = AutoImageProcessor.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True)
        self.device = device
        self.model = AutoModel.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True).eval().to(device)
        self.workers = workers

    def prepare_batch(self, paths, pool=None):
        """CPU-only preprocessing, also usable ahead of GPU inference."""
        import torch
        from PIL import Image
        def prepare(path):
            with Image.open(path) as source:
                # Preserve alpha: the official processor composites it on white.
                pixels = source.copy()
            return self.processor(images=pixels, return_tensors="pt")["pixel_values"]
        if pool is not None:
            return torch.cat(list(pool.map(prepare, paths)))
        with ThreadPoolExecutor(max_workers=self.workers) as temporary:
            return torch.cat(list(temporary.map(prepare, paths)))

    def score_prepared(self, batch):
        """Same PixAI forward pass; returns an array to avoid huge Python lists."""
        import torch
        size = len(batch)
        device = torch.device(self.device)
        self.provenance["precision"] = ("fp32-weights-bf16-autocast-fp32-sigmoid"
                                        if device.type == "cuda" else "fp32-weights-fp32-sigmoid")
        batch = batch.to(device, non_blocking=batch.is_pinned())
        autocast = (torch.autocast(device.type, dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext())
        with torch.inference_mode(), autocast:
            scores = self.model(batch).float().sigmoid()
        if scores.shape != (size, len(self.labels)) or not torch.isfinite(scores).all():
            raise ValueError("Invalid PixAI scores")
        return scores.cpu().numpy()

    def score(self, paths):
        return self.score_prepared(self.prepare_batch(paths)).tolist()


def prediction_record(image, sha, labels, scores, provenance, general=.35, character=.85):
    from PIL import Image
    with Image.open(image) as source:
        metadata = extract(dict(source.info), Path(image).name)
    row = make_record(metadata or {"format": "unavailable", "positive": [], "characters": [],
                                   "negative": [], "warnings": ["no-readable-generation-prompt"]},
                      Path(image).relative_to(Path(image).anchor).as_posix(), sha)
    predicted = []
    thresholds = provenance.get("thresholds", {"general": general, "character": character})
    for label, score in zip(labels, scores):
        kind = {"0": "general", "4": "character"}.get(label["category"], label["category"])
        if kind in thresholds and score >= thresholds[kind]:
            predicted.append({"tag": canonical(label["name"]), "category": kind, "score": float(score)})
    predicted.sort(key=lambda x: (-x["score"], x["tag"]))
    row["tags"] = sorted({x["tag"] for x in predicted if x["category"] != "rating"})
    for kind in ("general", "character", "copyright", "meta"):
        row["tags_" + kind] = sorted({x["tag"] for x in predicted if x["category"] == kind})
    row["tags_artist"] = sorted({x["tag"] for x in predicted if x["category"] == "style"})
    if row["caption"]:
        row["original_generation_caption"] = row["caption"]
    row["caption"] = ", ".join(x["tag"] for x in predicted if x["category"] not in ("rating", "meta"))
    family = provenance.get("family", "wd")
    row["caption_source"] = family + "-visual-tag-prediction"
    row["tagger_prediction"] = {"provenance": provenance, "thresholds": thresholds, "predictions": predicted,
                                 "artist_field_semantics": "style reference; not authorship"}
    return row


def fallback(catalog, output, model_path, batch_size=32, workers=4, limit=None, tagger_factory=LocalWD):
    source = sqlite3.connect(Path(catalog).resolve().as_uri() + "?mode=ro", uri=True)
    if not source.execute("SELECT value FROM config WHERE key='scan_complete'").fetchone():
        raise ValueError("Finish metadata scan first")
    db = open_catalog(output)
    tagger = tagger_factory(model_path, workers=workers)
    family = tagger.provenance.get("family", "wd")
    bind_config(db, "tagger", {"source_catalog": str(Path(catalog).resolve()), "model": tagger.provenance,
                           "parser": VERSION, "general": .35, "character": .85})
    selection = "SELECT path FROM files WHERE status IN ('no-generation-metadata','no-usable-tags') ORDER BY path"
    candidates = source.execute(selection)
    started = time.time()
    counts = {}
    processed = 0
    microbatch = [batch_size]
    last_report = time.monotonic()
    def report(state):
        nonlocal counts
        counts = dict(db.execute("SELECT status,count(*) FROM files GROUP BY status"))
        result = {"state": state, "files": counts, "elapsed_seconds": round(time.time() - started, 2),
                  "unique_records": db.execute("SELECT count(*) FROM records").fetchone()[0],
                  "caption_source": family + "-visual-tag-prediction", "images_modified": False,
                  "model": tagger.provenance, "processed_this_run": processed, "effective_batch_size": microbatch[0]}
        write_report(Path(output).with_suffix(".tagger.json"), result)
        print(json.dumps(result), flush=True)
        return result
    def pending():
        for (path,) in islice(candidates, limit):
            if not db.execute("SELECT 1 FROM files WHERE path=? AND status!='error'", (path,)).fetchone():
                yield path
    try:
        iterator = iter(pending())
        report("tagging")
        while paths := list(islice(iterator, batch_size)):
            before = {p: Path(p).stat() for p in paths}
            try:
                outputs = bounded_score(tagger.score, paths, microbatch)
            except (OSError, ValueError):
                outputs = []
                for path in paths:
                    try:
                        outputs.append(tagger.score([path])[0])
                    except (OSError, ValueError):
                        outputs.append(None)
            for path, scores in zip(paths, outputs):
                sha, status = None, "error"
                try:
                    if scores is not None:
                        sha = file_digest(path)
                        row = prediction_record(path, sha, tagger.labels, scores, tagger.provenance)
                        after = Path(path).stat()
                        if (before[path].st_size, before[path].st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                            raise ValueError("Image changed during tagging")
                        if row["tags"]:
                            inserted = db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?)", (sha, path, compact(row))).rowcount
                            status = "included" if inserted else "duplicate-bytes"
                        else:
                            status = "no-confident-tagger-tags"
                except (OSError, ValueError):
                    status = "error"
                db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?)", (
                    path, before[path].st_size, before[path].st_mtime_ns, status, sha, family + "-fallback"))
            processed += len(paths)
            db.commit()
            if time.monotonic() - last_report >= 10:
                report("tagging")
                last_report = time.monotonic()
        bind_config(db, "scan_complete", True)
        return report("complete")
    finally:
        db.commit(); db.close(); source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("catalog", "output", "model"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--tagger", choices=("pixai", "wd"), default="pixai")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128 or not 1 <= args.workers <= 14 or args.limit is not None and args.limit < 1:
        parser.error("Invalid batch size, workers or limit")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    free, total = torch.cuda.mem_get_info()
    budget = (free - .1 * total) / total
    if budget <= .1:
        raise RuntimeError("Insufficient VRAM headroom")
    torch.cuda.set_per_process_memory_fraction(budget)
    fallback(args.catalog, args.output, args.model, args.batch_size, args.workers, args.limit,
             tagger_factory=LocalPixAI if args.tagger == "pixai" else LocalWD)


if __name__ == "__main__":
    main()
