"""Private, rebuildable search data shared across snapshots and processes.

Only derived data lives here. Source hashes are checked on every open. A sealed
SQLite catalogue references an append-only vector file; snapshots select rows,
so adding a record does not copy the old vectors or voting documents. An OS lock
serializes preparation, while leases protect readers from cache eviction.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

import numpy as np

from .generation_metadata import recommendation_view
from .store import canonical, digest, revision, _sync_directory
from .vectors import decode_vector
from .privacy import private_directory

META = np.dtype([("image", "S64"), ("pixel", "S64"), ("rating", "S1"), ("order", "<i8")])
FORMAT = "search-cache/1"


def cache_root():
    override = os.environ.get("RS_VRS_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "rs-vrs-int" / "search"


def rules_digest():
    # Include implementation, not just hand-maintained version constants. A
    # changed parser/validator must not silently reuse yesterday's conversion.
    root = Path(__file__).parent
    return digest(b"".join((root / name).read_bytes() for name in (
        "search_cache.py", "store.py", "vectors.py", "identity.py",
        "generation_metadata.py", "tag_policy.py")))


def file_hash(path, interrupt=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            if interrupt:
                interrupt()
            h.update(block)
    return h.hexdigest()


class FileLock:
    """Crash-released exclusive lock; never infer liveness from a PID file."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("a+b")
        if os.name == "nt" and self.path.stat().st_size == 0:
            self.file.write(b"0")
            self.file.flush()

    def acquire(self, interrupt=None, blocking=True):
        while True:
            if interrupt:
                interrupt()
            try:
                if os.name == "nt":
                    import msvcrt
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (BlockingIOError, PermissionError):
                if not blocking:
                    return False
                time.sleep(.05)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def unused(path):
    for name in path.glob("lease-*"):
        lock = FileLock(name)
        try:
            if not lock.acquire(blocking=False):
                return False
        finally:
            lock.close()
        name.unlink(missing_ok=True)
    return True


def cache_limit():
    value = float(os.environ.get("RS_VRS_CACHE_MAX_GIB", "8"))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("RS_VRS_CACHE_MAX_GIB must be a positive size in GiB")
    return int(value * 1024 ** 3)


def reserve(root, current, extra=0):
    """Called under the preparation lock. Evict only unleased derived data."""
    paths = list(root.glob("*/generation-*"))
    sizes = {p: tree_bytes(p) for p in paths}
    total = sum(sizes.values())
    limit = cache_limit()
    for path in sorted(paths, key=lambda p: p.stat().st_mtime_ns):
        disk = shutil.disk_usage(root)
        if total + extra <= limit and disk.free - extra >= disk.total * .2:
            return
        if path != current and unused(path):
            shutil.rmtree(path)
            total -= sizes[path]
    disk = shutil.disk_usage(root)
    if total + extra > limit:
        raise OSError("VRS search cache exceeds RS_VRS_CACHE_MAX_GIB; close unused memories or increase the limit")
    if disk.free - extra < disk.total * .2:
        raise OSError("VRS search cache would cross the 20% free disk reserve; set RS_VRS_CACHE_DIR")


def seal(path, count, interrupt):
    value = {"format": FORMAT, "count": count,
             "files": {name: file_hash(path / name, interrupt)
                       for name in ("vectors.f32", "votes.sqlite3")}}
    pending = path / "seal.pending"
    with pending.open("wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path / "seal.json")
    _sync_directory(path)


def sealed(path, dimension, interrupt):
    try:
        value = json.loads((path / "seal.json").read_bytes())
        if value["format"] != FORMAT or type(value["count"]) is not int or value["count"] < 0:
            return False
        if (path / "vectors.f32").stat().st_size != value["count"] * dimension * 4:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    # Cancellation is not cache corruption; don't swallow the callback's error.
    for name in ("vectors.f32", "votes.sqlite3"):
        if name not in value.get("files", {}):
            return False
        try:
            expected = value["files"][name]
            actual = file_hash(path / name, interrupt)
        except FileNotFoundError:
            return False
        if actual != expected:
            return False
    return True


class PersistentCache:
    def __init__(self, store, manifest, interrupt=None):
        self.lease = None
        self.path = None
        self.rows = []
        self.stats = dict(converted_entries=0, reused_entries=0, built_objects=0, reused_objects=0)
        root = cache_root() / "persistent-v1"
        private_directory(root)
        namespace = root / digest(canonical({"format": FORMAT, "rules": rules_digest(),
                                            "contract": manifest["contract"]}))
        lock = FileLock(root / "prepare.lock")
        try:
            lock.acquire(interrupt)
            # Check even unchanged sources before trusting a cached descriptor.
            # Hash compressed bytes without unpacking or reinterpreting tags.
            for oid in manifest["objects"].values():
                if interrupt:
                    interrupt()
                if file_hash(store.object_path(oid), interrupt) != oid:
                    raise ValueError("Bucket hash mismatch")
            namespace.mkdir(exist_ok=True, mode=0o700)
            dimension = manifest["contract"]["dimension"]
            for path in sorted(namespace.glob("generation-*"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
                if sealed(path, dimension, interrupt):
                    self.path = path
                    break
            fresh = self.path is None
            reserve(root, self.path)
            if fresh:
                self.path = namespace / ("generation-" + uuid.uuid4().hex)
                self.path.mkdir(mode=0o700)
                (self.path / "vectors.f32").touch()
            self.lease = FileLock(self.path / ("lease-" + uuid.uuid4().hex))
            self.lease.acquire()
            try:
                self._prepare(store, manifest, root, fresh, interrupt)
                os.utime(self.path, None)
            except BaseException:
                self.close()
                if fresh:
                    shutil.rmtree(self.path)
                raise
        finally:
            lock.close()

    def _prepare(self, store, manifest, root, fresh, interrupt):
        with closing(sqlite3.connect(self.path / "votes.sqlite3")) as db:
            db.execute("PRAGMA cache_size=-4096")
            if fresh:
                db.executescript("""
                    CREATE TABLE records (id INTEGER PRIMARY KEY, revision TEXT UNIQUE NOT NULL,
                                          key TEXT NOT NULL, value BLOB NOT NULL, meta BLOB NOT NULL);
                    CREATE TABLE objects (id TEXT PRIMARY KEY, rows BLOB NOT NULL);
                """)
            count = db.execute("SELECT count(*) FROM records").fetchone()[0]
            self.pool_count = count
            size = (self.path / "vectors.f32").stat().st_size
            changed = fresh
            with (self.path / "vectors.f32").open("r+b") as vectors:
                vectors.seek(size)
                db.execute("BEGIN IMMEDIATE")
                try:
                    for bucket in sorted(map(int, manifest["objects"])):
                        if interrupt:
                            interrupt()
                        oid = manifest["objects"][str(bucket)]
                        object_key = f"{manifest['bucket_count']}:{bucket}:{oid}"
                        cached = db.execute("SELECT rows FROM objects WHERE id=?", (object_key,)).fetchone()
                        if cached is not None:
                            self.rows.extend(np.frombuffer(cached[0], dtype="<i8").tolist())
                            self.stats["reused_objects"] += 1
                            continue
                        changed = True
                        rows = []
                        for entry in store.read_bucket(manifest, bucket):
                            if interrupt:
                                interrupt()
                            if entry.get("deleted") or entry["vector"] is None or entry["document"]["split"] != "train":
                                continue
                            rev = revision(entry)
                            cached = db.execute("SELECT id FROM records WHERE revision=?", (rev,)).fetchone()
                            if cached is not None:
                                rows.append(cached[0])
                                self.stats["reused_entries"] += 1
                                continue
                            vector, meta, vote = convert(entry, manifest["contract"]["dimension"])
                            vectors.write(vector)
                            db.execute("INSERT INTO records VALUES (?,?,?,?,?)",
                                       (self.pool_count, rev, entry["key"], vote, meta))
                            rows.append(self.pool_count)
                            self.pool_count += 1
                            self.stats["converted_entries"] += 1
                        db.execute("INSERT INTO objects VALUES (?,?)",
                                   (object_key, np.asarray(rows, dtype="<i8").tobytes()))
                        self.rows.extend(rows)
                        self.stats["built_objects"] += 1
                        # Include uncommitted journal/DB and buffered vector bytes.
                        vectors.flush()
                        reserve(root, self.path)
                    if interrupt:
                        interrupt()
                    if changed:
                        vectors.flush()
                        os.fsync(vectors.fileno())
                        db.commit()
                    else:
                        db.rollback()
                except BaseException:
                    db.rollback()
                    vectors.truncate(size)
                    vectors.flush()
                    os.fsync(vectors.fileno())
                    raise
            if changed:
                # Commit is durable before publication. A crash here leaves an
                # unsealed generation, never an accepted partial index.
                # Publication is a short non-cancellable critical section after
                # commit. Cancellation before commit rolls back appended rows.
                seal(self.path, self.pool_count, None)
            reserve(root, self.path)
            self.stats["cache_bytes"] = tree_bytes(self.path)

    def close(self):
        if self.lease is not None:
            path = self.lease.path
            self.lease.close()
            path.unlink(missing_ok=True)
            self.lease = None

    def __del__(self):
        self.close()


def convert(entry, dimension):
    doc = entry["document"]
    pixel = ""
    for audit in entry.get("audit", []):
        if isinstance(audit, dict) and audit.get("native_pixel_sha256"):
            pixel = audit["native_pixel_sha256"]
    if not isinstance(pixel, str) or len(pixel) != 64 or not pixel.isascii():
        pixel = ""
    meta = np.array([(doc["image_sha256"], pixel, doc["rating"], 0)], dtype=META).tobytes()
    vector = decode_vector(entry["vector"], dimension).astype("<f4", copy=False).tobytes()
    view = recommendation_view(entry)
    vote = {"key": entry["key"], "document": {
        k: view["document"][k] for k in ("id", "image_sha256", "rating", "tags")},
        "categories": view["categories"]}
    return vector, meta, canonical(vote)
