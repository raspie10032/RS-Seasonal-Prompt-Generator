"""Resumable local metadata import and GPU encoding of personal generations.

Examples: python -m vrs_int.generated scan --help / build --help.
Catalogs, manifests, checkpoints and stores are private artifacts, not code.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import zlib

from .generation_metadata import VERSION, extract, make_record
from .import_snapshot import file_digest

EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}


def compact(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, sort_keys=True).encode(), 6)


def expand(value):
    return json.loads(zlib.decompress(value))


def write_report(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def bounded_score(score, items, maximum, release=None):
    """Remember an OOM-reduced microbatch without dropping or reordering input."""
    result = []
    offset = 0
    while offset < len(items):
        chunk = items[offset:offset + maximum[0]]
        try:
            values = score(chunk)
        except RuntimeError as exc:
            if type(exc).__name__ != "OutOfMemoryError" or len(chunk) <= 1:
                raise
            maximum[0] = max(1, len(chunk) // 2)
        else:
            if len(values) != len(chunk):
                raise ValueError("Model batch count mismatch")
            result.extend(values)
            offset += len(chunk)
            continue
        # Leave the exception block first so its traceback releases activations.
        if release is None:
            import torch
            torch.cuda.empty_cache()
        else:
            release()
        print(json.dumps({"state": "reduced-batch", "batch_size": maximum[0]}), flush=True)
    return result


def open_catalog(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,size INTEGER,mtime INTEGER,status TEXT,sha TEXT,detail TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS records(sha TEXT PRIMARY KEY,path TEXT,record BLOB)")
    return db


def bind_config(db, key, value):
    text = json.dumps(value, sort_keys=True)
    previous = db.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    if previous and previous[0] != text:
        raise ValueError("Checkpoint configuration changed: " + key)
    db.execute("INSERT OR IGNORE INTO config VALUES(?,?)", (key, text))
    db.commit()


_SCAN_HINTS = {}


def _scan_init(hints):
    global _SCAN_HINTS
    _SCAN_HINTS = hints


def _scan_image(path):
    from PIL import Image
    image = Path(path)
    before = image.stat()
    status, sha, detail, payload = "no-generation-metadata", None, None, None
    try:
        with Image.open(image) as im:
            parsed = extract(dict(im.info), image.name)
        if parsed:
            sha = file_digest(image)
            row = make_record(parsed, image.relative_to(image.anchor).as_posix(), sha, _SCAN_HINTS)
            after = image.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source changed during metadata read")
            status = "included" if row["tags"] else "no-usable-tags"
            if row["tags"]:
                payload = compact(row)
                detail = parsed["format"]
    except Exception as exc:
        status, detail = "error", type(exc).__name__
    return str(image), before.st_size, before.st_mtime_ns, status, sha, detail, payload


def scan(roots, catalog, categories=None, progress_every=1000, workers=1):
    roots = sorted({str(Path(p).resolve(strict=True)) for p in roots})
    hints = json.loads(Path(categories).read_text(encoding="utf-8")) if categories else {}
    catalog = Path(catalog)
    db = open_catalog(catalog)
    started = time.time()
    bind_config(db, "scan", {"roots": roots, "parser": VERSION,
                "categories_sha256": file_digest(categories) if categories else None})
    seen = 0

    def report(state):
        result = {"state": state, "roots": roots, "parser": VERSION,
                  "files": dict(db.execute("SELECT status,count(*) FROM files GROUP BY status")),
                  "unique_records": db.execute("SELECT count(*) FROM records").fetchone()[0],
                  "current_pass_seen": seen, "elapsed_seconds": round(time.time() - started, 2),
                  "images_modified": False, "rating_policy": "u: unknown; no content rating inferred"}
        write_report(catalog.with_suffix(".scan.json"), result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    def pending():
        nonlocal seen
        for root in roots:
            for directory, dirs, filenames in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not Path(directory, d).is_symlink())
                for filename in sorted(filenames):
                    image = Path(directory, filename)
                    if image.suffix.lower() not in EXTENSIONS or image.is_symlink():
                        continue
                    seen += 1
                    before = image.stat()
                    old = db.execute("SELECT size,mtime,status FROM files WHERE path=?", (str(image),)).fetchone()
                    if old and old[:2] == (before.st_size, before.st_mtime_ns) and old[2] != "error":
                        continue
                    yield str(image)
    pool = None
    try:
        _scan_init(hints)
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers, initializer=_scan_init, initargs=(hints,))
        # Bound pending work and memory even for very large folders.
        paths = iter(pending())
        completed = 0
        while group := list(islice(paths, max(1, workers) * 64)):
            results = pool.map(_scan_image, group, chunksize=8) if pool else map(_scan_image, group)
            for path, size, mtime, status, sha, detail, payload in results:
                if payload:
                    inserted = db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?)", (sha, path, payload)).rowcount
                    status = "included" if inserted else "duplicate-bytes"
                db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?)", (path, size, mtime, status, sha, detail))
                completed += 1
                if completed % progress_every == 0:
                    db.commit(); report("scanning")
        db.commit()
        # A readable, private JSONL accompanies the compressed working catalog.
        manifest = catalog.with_suffix(".jsonl")
        temporary = manifest.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for (record,) in db.execute("SELECT record FROM records ORDER BY sha"):
                stream.write(json.dumps(expand(record), ensure_ascii=False) + "\n")
        temporary.replace(manifest)
        bind_config(db, "scan_complete", True)
        return report("complete")
    finally:
        if pool:
            pool.shutdown()
        db.close()


def entry_from_record(row, vector):
    from .identity import observation_key
    from .tag_policy import filter_tags, typed_tags
    kept, removed = filter_tags(row)
    typed = typed_tags(row)
    return {"key": observation_key(row), "document": {
        "source": "generated", "id": row["id"], "image": row["image"],
        "image_sha256": row["image_sha256"], "rating": row["rating"], "split": row["split"],
        "source_ids": [row["id"]], "tags": kept},
        "categories": {tag: sorted(k for k, names in typed.items() if tag in names) or ["unknown"] for tag in kept},
        "vector": vector, "audit": [{"source_record": row, "removed_tags": removed}]}


def build(catalog, encoder_path, destination, checkpoint, *, limit=None, delay=0.02, batch_size=1, encoder_factory=None, shard_index=0, shard_count=1):
    from .build import encoder_contract
    from .store import Store, canonical, digest, validate_entry
    from .vectors import encode_vector
    catalog, checkpoint = Path(catalog), Path(checkpoint)
    source = sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)
    if not source.execute("SELECT value FROM config WHERE key='scan_complete'").fetchone():
        source.close()
        raise ValueError("Finish the metadata scan before building")
    contract = encoder_contract(encoder_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(checkpoint)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS encoded(sha TEXT PRIMARY KEY,record_hash TEXT,entry BLOB)")
    db.execute("CREATE TABLE IF NOT EXISTS errors(sha TEXT PRIMARY KEY,detail TEXT)")
    if shard_count not in (1, 2, 4, 8, 16) or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid hash shard")
    where = f" WHERE (instr('0123456789abcdef',substr(sha,1,1))-1) % {shard_count} = {shard_index}"
    bind_config(db, "build", {"catalog": str(catalog.resolve()), "contract": contract,
                               "shard_index": shard_index, "shard_count": shard_count})
    if encoder_factory is None:
        from .encoder import ImageEncoder
        encoder_factory = ImageEncoder
    total = source.execute("SELECT count(*) FROM records" + where).fetchone()[0]
    if limit:
        total = min(limit, total)
    started, processed, encoder = time.time(), 0, None
    microbatch = [batch_size]

    def report(state, **extra):
        result = {"state": state, "target": total,
                  "encoded": db.execute("SELECT count(*) FROM encoded").fetchone()[0],
                  "errors": db.execute("SELECT count(*) FROM errors").fetchone()[0],
                  "processed_this_run": processed, "elapsed_seconds": round(time.time() - started, 2),
                  "encoder_contract": contract, **extra}
        write_report(checkpoint.with_suffix(".build.json"), result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    try:
        report("encoding")
        def pending():
            rows = source.execute("SELECT sha,path,record FROM records" + where + " ORDER BY sha")
            for sha, image, payload in islice(rows, limit):
                row = expand(payload)
                row_hash = digest(canonical(row))
                cached = db.execute("SELECT record_hash FROM encoded WHERE sha=?", (sha,)).fetchone()
                if cached:
                    if cached[0] != row_hash:
                        raise ValueError("Metadata changed; use a fresh checkpoint")
                    continue
                yield sha, image, row, row_hash

        def save(item, vector):
            sha, image, row, row_hash, before = item
            after = Path(image).stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source changed during encoding")
            if hasattr(vector, "detach"):
                vector = vector.detach().cpu().numpy()
            entry = entry_from_record(row, encode_vector(vector, contract["codec"]))
            validate_entry(entry, contract)
            db.execute("INSERT INTO encoded VALUES(?,?,?)", (sha, row_hash, compact(entry)))
            db.execute("DELETE FROM errors WHERE sha=?", (sha,))

        def error(sha, exc):
            db.execute("INSERT OR REPLACE INTO errors VALUES(?,?)", (sha, type(exc).__name__ + ": " + str(exc)[:160]))

        iterator = iter(pending())
        last_report = time.monotonic()
        while group := list(islice(iterator, batch_size)):
            valid = []
            for item in group:
                try:
                    before = Path(item[1]).stat()
                    if file_digest(item[1]) != item[0]:
                        raise ValueError("Source no longer matches scan")
                    valid.append((*item, before))
                except (OSError, ValueError) as exc:
                    error(item[0], exc)
            if valid:
                if encoder is None:
                    encoder = encoder_factory(encoder_path)
                try:
                    vectors = bounded_score(encoder.encode_many, [x[1] for x in valid], microbatch) if batch_size > 1 else [encoder.encode(valid[0][1])]
                except (OSError, ValueError):
                    # Isolate malformed images; never lose the rest of a batch.
                    vectors = []
                    for item in valid:
                        try:
                            vectors.append(encoder.encode(item[1]))
                        except (OSError, ValueError) as exc:
                            error(item[0], exc); vectors.append(None)
                if len(vectors) != len(valid):
                    raise ValueError("Encoder batch count mismatch")
                for item, vector in zip(valid, vectors):
                    if vector is not None:
                        try:
                            save(item, vector)
                        except (OSError, ValueError) as exc:
                            error(item[0], exc)
            processed += len(group)
            db.commit()
            if time.monotonic() - last_report >= 10:
                report("encoding", batch_size=batch_size, effective_batch_size=microbatch[0])
                last_report = time.monotonic()
                disk = shutil.disk_usage(checkpoint.parent)
                if disk.free / disk.total < .20:
                    raise RuntimeError("Keep at least 20% disk free; checkpoints preserved")
            if delay:
                time.sleep(delay)
        db.commit()
        # Decode failures remain explicit in the report and keep their metadata
        # observations, but never acquire fabricated visual vectors.
        def entries():
            for number, (sha, payload) in enumerate(source.execute("SELECT sha,record FROM records" + where + " ORDER BY sha")):
                if limit and number >= limit:
                    break
                encoded = db.execute("SELECT entry FROM encoded WHERE sha=?", (sha,)).fetchone()
                yield expand(encoded[0]) if encoded else entry_from_record(expand(payload), None)
        store = Store(destination)
        snapshot = store.build(entries(), contract, "personal-generated-images", bucket_count=1024, provenance={
            "metadata_parser": VERSION, "catalog_configuration": dict(source.execute("SELECT key,value FROM config")),
            "images_included": False, "captions": "generation intent, not verified visual descriptions",
            "rating_policy": "unknown", "subset_limit": limit,
            "shard_index": shard_index, "shard_count": shard_count})
        verified = store.verify(snapshot)
        return report("complete-with-errors" if db.execute("SELECT count(*) FROM errors").fetchone()[0] else "complete",
                      snapshot=snapshot, verification=verified, store=str(Path(destination).resolve()))
    finally:
        del encoder
        db.commit(); db.close(); source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    scan_parser = subs.add_parser("scan")
    scan_parser.add_argument("--root", type=Path, action="append", required=True)
    scan_parser.add_argument("--catalog", type=Path, required=True)
    scan_parser.add_argument("--categories", type=Path)
    scan_parser.add_argument("--cpu-threads", type=int, default=2)
    build_parser = subs.add_parser("build")
    for field in ("catalog", "encoder", "store", "checkpoint"):
        build_parser.add_argument("--" + field, type=Path, required=True)
    build_parser.add_argument("--gpu", required=True, help="Explicit CUDA GPU UUID")
    build_parser.add_argument("--limit", type=int, help="Optional subset smoke run")
    build_parser.add_argument("--delay", type=float, default=.02, help="Sleep after each batch")
    build_parser.add_argument("--batch-size", type=int, default=1, help="1..256 images; larger batches use more VRAM")
    build_parser.add_argument("--cpu-threads", type=int, default=2)
    build_parser.add_argument("--preprocess-workers", type=int, default=1)
    build_parser.add_argument("--gpu-fraction", type=float, default=.12, help="Max total VRAM fraction; existing allocations are subtracted")
    build_parser.add_argument("--shard-index", type=int, default=0)
    build_parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu threads must be positive")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.cpu_threads)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_threads])
    if hasattr(os, "nice"):
        os.nice(15)
    if args.command == "scan":
        scan(args.root, args.catalog, args.categories, workers=args.cpu_threads)
    else:
        if args.limit is not None and args.limit < 1 or args.delay < 0 or not 1 <= args.batch_size <= 256 or not 0 < args.gpu_fraction <= .9 or not 1 <= args.preprocess_workers <= args.cpu_threads:
            parser.error("Use positive limit, nonnegative delay, batch size 1..256, GPU fraction (0,.9], and workers within CPU budget")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["VRS_PREPROCESS_WORKERS"] = str(args.preprocess_workers)
        import torch
        torch.set_num_threads(min(args.cpu_threads, 4)); torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        free, total = torch.cuda.mem_get_info()
        available = (free - total * (1 - args.gpu_fraction)) / total
        if available <= .02:
            raise RuntimeError("Insufficient free VRAM within requested total budget")
        torch.cuda.set_per_process_memory_fraction(available)
        build(args.catalog, args.encoder, args.store, args.checkpoint, limit=args.limit, delay=args.delay,
              batch_size=args.batch_size, shard_index=args.shard_index, shard_count=args.shard_count)


if __name__ == "__main__":
    main()
