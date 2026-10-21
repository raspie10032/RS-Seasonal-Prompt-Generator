"""Immutable hash-addressed snapshots and explicit, atomic local activation.

A snapshot is a flat map of stable key buckets, not a replay-dependent delta
chain. Packages carry only new compressed buckets. Old snapshots remain readable.
Hashes detect changes; they do NOT authenticate a publisher.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import zlib

from .vectors import decode_vector
from .identity import observation_key
from .privacy import private_directory

FORMAT = "vrs-int/1"
GENERATED_FORMAT = "vrs-int/2"
COMPACT_FORMAT = "vrs-int/3"
SUPPORTED_FORMATS = (FORMAT, GENERATED_FORMAT, COMPACT_FORMAT)
XZ_MAGIC = b"VRSXZ1\x00"
MAX_OBJECT_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def check_id(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError("Expected a SHA-256 content ID")
    return value


def revision(entry):
    return digest(canonical(entry))


def bucket(key, count):
    return int(digest(key.encode("utf-8"))[:16], 16) % count


def _sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_once(path, data):
    """Publish a complete immutable file without overwriting another writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
            _sync_directory(path.parent)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("Immutable file already exists with different content")
    finally:
        os.unlink(name)


def _bounded_read(path, limit=MAX_OBJECT_BYTES):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Object exceeds format size limit; use more buckets")
    return raw


def _inflate(raw):
    if raw.startswith(XZ_MAGIC):
        import lzma
        try:
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)
            value = decoder.decompress(raw[len(XZ_MAGIC):], max_length=MAX_OBJECT_BYTES + 1)
        except lzma.LZMAError as exc:
            raise ValueError("Invalid compressed object") from exc
        if len(value) > MAX_OBJECT_BYTES or not decoder.eof or decoder.unused_data:
            raise ValueError("Invalid or oversized compressed object")
        return value
    decoder = zlib.decompressobj()
    value = decoder.decompress(raw, MAX_OBJECT_BYTES + 1)
    if len(value) > MAX_OBJECT_BYTES or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid or oversized compressed object")
    return value


def pack(entries, compression="zlib"):
    # Bucket-local integer dictionaries prevent a new name renumbering the
    # entire collection. Names with frequency one are never discarded.
    tags = sorted({tag for e in entries if not e.get("deleted") for tag in e["document"]["tags"]})
    lookup = {tag: i for i, tag in enumerate(tags)}
    records = []
    for entry in sorted(entries, key=lambda e: e["key"]):
        row = dict(entry)
        if not entry.get("deleted"):
            row["document"] = dict(entry["document"], tags=[lookup[t] for t in entry["document"]["tags"]])
            row["categories"] = [[lookup[t], c] for t, c in sorted(entry["categories"].items())]
        records.append(row)
    raw = canonical({"tags": tags, "records": records})
    if len(raw) > MAX_OBJECT_BYTES:
        raise ValueError("Bucket too large; rebuild with more buckets")
    if compression == "zlib":
        result = zlib.compress(raw, level=6)
    elif compression == "xz":
        import lzma
        result = XZ_MAGIC + lzma.compress(raw, preset=3)
    else:
        raise ValueError("Unsupported bucket compression")
    if len(result) > MAX_OBJECT_BYTES:
        raise ValueError("Compressed bucket too large; rebuild with more buckets")
    return result


def unpack(raw):
    payload = json.loads(_inflate(raw))
    tags = payload["tags"]
    if not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags):
        raise ValueError("Invalid tag dictionary")
    if tags != sorted(set(tags)):
        raise ValueError("Noncanonical tag dictionary")

    def tag(index):
        if type(index) is not int or not 0 <= index < len(tags):
            raise ValueError("Invalid dictionary index")
        return tags[index]

    records = payload["records"]
    for row in records:
        if not row.get("deleted"):
            row["document"]["tags"] = [tag(i) for i in row["document"]["tags"]]
            pairs = row["categories"]
            row["categories"] = {tag(i): c for i, c in pairs}
            if len(row["categories"]) != len(pairs):
                raise ValueError("Duplicate category mapping")
    return records


def validate_contract(contract):
    if set(contract) != {"encoder", "dimension", "codec", "tag_policy"}:
        raise ValueError("Incomplete compatibility contract")
    encoder = contract["encoder"]
    if set(encoder) != {"provenance_sha256", "preprocess", "backend", "quantization", "pooling"}:
        raise ValueError("Incomplete encoder contract")
    check_id(encoder["provenance_sha256"])
    if not all(isinstance(v, str) and v for v in encoder.values()):
        raise ValueError("Invalid encoder contract")
    if type(contract["dimension"]) is not int or not 1 <= contract["dimension"] <= 8192:
        raise ValueError("Invalid vector dimension")
    if contract["codec"] not in ("f32le", "i8") or not contract["tag_policy"]:
        raise ValueError("Invalid codec or tag policy")


def validate_entry(entry, contract):
    if not re.fullmatch(r"(?:danbooru:[1-9][0-9]*|generated:[0-9a-f]{64})", entry.get("key", "")):
        raise ValueError("Invalid observation key")
    if entry.get("deleted"):
        if set(entry) != {"key", "deleted", "previous_revision", "reason"} or not entry["reason"]:
            raise ValueError("Invalid tombstone")
        check_id(entry["previous_revision"])
        return
    if set(entry) != {"key", "document", "categories", "vector", "audit"}:
        raise ValueError("Invalid observation schema")
    doc = entry["document"]
    if entry["key"] != observation_key(doc):
        raise ValueError("ID or rating mismatch")
    if doc["split"] not in ("train", "val", "test"):
        raise ValueError("Invalid evaluation split")
    check_id(doc["image_sha256"])
    tags = doc["tags"]
    if not tags or any(not isinstance(t, str) or not t for t in tags) or len(set(tags)) != len(tags):
        raise ValueError("Invalid observation tags")
    if set(entry["categories"]) - set(tags):
        raise ValueError("Category without a retained tag")
    for categories in entry["categories"].values():
        if not isinstance(categories, list) or any(c not in ("artist", "character", "copyright", "general", "meta", "unknown") for c in categories):
            raise ValueError("Invalid tag categories")
    if not isinstance(entry["audit"], list):
        raise ValueError("Audit records must be a list")
    if entry["vector"] is not None:
        if entry["vector"]["codec"] != contract["codec"]:
            raise ValueError("Mixed vector representations")
        decode_vector(entry["vector"], contract["dimension"])
    canonical(entry)  # Reject NaN in metadata too.


class Store:
    def __init__(self, root):
        self.root = Path(root)

    def object_path(self, object_id):
        return self.root / "objects" / (check_id(object_id) + ".z")

    def manifest_path(self, snapshot):
        return self.root / "snapshots" / (check_id(snapshot) + ".json")

    def put_object(self, entries, compression="zlib"):
        raw = pack(entries, compression)
        object_id = digest(raw)
        write_once(self.object_path(object_id), raw)
        return object_id

    def manifest(self, snapshot):
        raw = _bounded_read(self.manifest_path(snapshot), MAX_MANIFEST_BYTES)
        if digest(raw) != snapshot:
            raise ValueError("Snapshot hash mismatch")
        manifest = json.loads(raw)
        if manifest["format"] not in SUPPORTED_FORMATS:
            raise ValueError("Unsupported VRS-INT format")
        validate_contract(manifest["contract"])
        count = manifest["bucket_count"]
        if type(count) is not int or not 1 <= count <= 65536:
            raise ValueError("Invalid bucket count")
        for key, value in manifest["objects"].items():
            if not key.isdecimal() or str(int(key)) != key or not 0 <= int(key) < count:
                raise ValueError("Invalid bucket number")
            check_id(value)
        if manifest["parent"] is not None:
            check_id(manifest["parent"])
        return manifest

    def read_bucket(self, manifest, index):
        object_id = manifest["objects"].get(str(index))
        if object_id is None:
            return []
        raw = _bounded_read(self.object_path(object_id))
        if digest(raw) != object_id:
            raise ValueError("Bucket hash mismatch")
        if raw.startswith(XZ_MAGIC) and manifest["format"] != COMPACT_FORMAT:
            raise ValueError("Compact buckets require vrs-int/3")
        entries = unpack(raw)
        keys = set()
        for entry in entries:
            validate_entry(entry, manifest["contract"])
            key = entry["key"]
            if key in keys or bucket(key, manifest["bucket_count"]) != index:
                raise ValueError("Duplicate key or wrong bucket")
            keys.add(key)
        return entries

    def entries(self, snapshot, include_deleted=False):
        manifest = self.manifest(snapshot)
        for index in sorted(map(int, manifest["objects"])):
            for entry in self.read_bucket(manifest, index):
                if include_deleted or not entry.get("deleted"):
                    yield entry

    def get(self, snapshot, key):
        manifest = self.manifest(snapshot)
        return next((e for e in self.read_bucket(manifest, bucket(key, manifest["bucket_count"]))
                     if e["key"] == key), None)

    def verify(self, snapshot):
        manifest = self.manifest(snapshot)
        counts = {"observations": 0, "vectors": 0, "tombstones": 0, "source_records": 0}
        for entry in self.entries(snapshot, include_deleted=True):
            if entry.get("deleted"):
                counts["tombstones"] += 1
            else:
                counts["observations"] += 1
                counts["vectors"] += entry["vector"] is not None
                counts["source_records"] += len(entry["audit"])
        if counts != manifest["counts"]:
            raise ValueError("Snapshot counts mismatch")
        return {"snapshot": snapshot, **counts,
                "snapshot_bytes": self.manifest_path(snapshot).stat().st_size + sum(
                    self.object_path(v).stat().st_size for v in set(manifest["objects"].values()))}

    def _publish(self, contract, count, objects, parent, label, counts, provenance, format=FORMAT):
        manifest = {"format": format, "contract": contract, "bucket_count": count,
                    "objects": objects, "parent": parent, "label": label,
                    "counts": counts, "provenance": provenance}
        raw = canonical(manifest)
        snapshot = digest(raw)
        write_once(self.manifest_path(snapshot), raw)
        return snapshot

    def build(self, entries, contract, label, provenance=None, bucket_count=256):
        validate_contract(contract)
        if not 1 <= bucket_count <= 65536:
            raise ValueError("Invalid bucket count")
        counts = {"observations": 0, "vectors": 0, "tombstones": 0, "source_records": 0}
        input_bytes = 0
        format = FORMAT
        private_directory(self.root)
        # Spool metadata, never original images. Only one bucket is packed in
        # RAM at a time, including at corpus sizes beyond this initial trial.
        with tempfile.TemporaryDirectory(prefix=".build-", dir=self.root) as staging:
            # sqlite3's transaction context does not close the connection.
            # Close on both success and failure before removing the spool on Windows.
            with closing(sqlite3.connect(str(Path(staging) / "spool.sqlite3"))) as db, db:
                db.execute("CREATE TABLE records (key TEXT PRIMARY KEY, bucket INTEGER, value BLOB)")
                for entry in entries:
                    validate_entry(entry, contract)
                    if entry["key"].startswith("generated:"):
                        format = GENERATED_FORMAT
                    if entry.get("deleted"):
                        raise ValueError("Deleted observation in base")
                    try:
                        encoded = canonical(entry)
                        db.execute("INSERT INTO records VALUES (?,?,?)", (
                            entry["key"], bucket(entry["key"], bucket_count), encoded))
                        input_bytes += len(encoded) + 1
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("Duplicate observation in base") from exc
                    counts["observations"] += 1
                    counts["vectors"] += entry["vector"] is not None
                    counts["source_records"] += len(entry["audit"])
                db.execute("CREATE INDEX by_bucket ON records(bucket)")
                db.commit()
                objects = {}
                for index in range(bucket_count):
                    rows = [json.loads(r[0]) for r in db.execute("SELECT value FROM records WHERE bucket=?", (index,))]
                    if rows:
                        objects[str(index)] = self.put_object(rows)
        provenance = dict(provenance or {}, canonical_input_jsonl_bytes=input_bytes)
        return self._publish(contract, bucket_count, objects, None, label, counts, provenance, format)

    def update(self, parent, operations, label, contract=None, provenance=None):
        """Compare revisions before replacing observations, never add extra votes.

        Expected revision None means a previously unseen key. Restoring a
        tombstone requires its own current revision. Any conflict aborts before
        publishing a snapshot. A changed encoder needs a new base.
        """
        old = self.manifest(parent)
        if contract is not None and contract != old["contract"]:
            raise ValueError("Incompatible update; re-encode into a separate base")
        counts = dict(old["counts"])
        format = old["format"]
        objects = dict(old["objects"])
        private_directory(self.root)
        # Spool operations rather than retaining every touched bucket in RAM.
        # A late conflict may leave unused immutable objects, but never publishes
        # a snapshot or changes its parent. Spool files close before removal.
        with tempfile.TemporaryDirectory(prefix=".update-", dir=self.root) as staging:
            with closing(sqlite3.connect(str(Path(staging) / "operations.sqlite3"))) as db, db:
                db.execute("CREATE TABLE operations (key TEXT PRIMARY KEY, bucket INTEGER, value BLOB)")
                for op in operations:
                    key = op["key"]
                    if "expected_revision" not in op:
                        raise ValueError("Duplicate operation or missing expected revision")
                    try:
                        db.execute("INSERT INTO operations VALUES (?,?,?)", (
                            key, bucket(key, old["bucket_count"]), canonical(op)))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("Duplicate operation or missing expected revision") from exc
                db.execute("CREATE INDEX by_bucket ON operations(bucket)")
                db.commit()
                for index, in db.execute("SELECT DISTINCT bucket FROM operations ORDER BY bucket"):
                    rows = {e["key"]: e for e in self.read_bucket(old, index)}
                    for raw, in db.execute("SELECT value FROM operations WHERE bucket=?", (index,)):
                        op = json.loads(raw)
                        key = op["key"]
                        previous = rows.get(key)
                        if op["expected_revision"] != (revision(previous) if previous else None):
                            raise ValueError("Observation revision conflict: " + key)
                        if op["op"] == "upsert":
                            entry = op["entry"]
                            if entry["key"] != key or entry.get("deleted"):
                                raise ValueError("Invalid upsert")
                        elif op["op"] == "delete" and previous and not previous.get("deleted"):
                            entry = {"key": key, "deleted": True, "previous_revision": revision(previous),
                                     "reason": op["reason"]}
                        else:
                            raise ValueError("Invalid operation or deletion target")
                        validate_entry(entry, old["contract"])
                        if key.startswith("generated:") and format != COMPACT_FORMAT:
                            format = GENERATED_FORMAT
                        for value, sign in ((previous, -1), (entry, 1)):
                            if value is None:
                                continue
                            if value.get("deleted"):
                                counts["tombstones"] += sign
                            else:
                                counts["observations"] += sign
                                counts["vectors"] += sign * (value["vector"] is not None)
                                counts["source_records"] += sign * len(value["audit"])
                        rows[key] = entry
                    objects[str(index)] = self.put_object(list(rows.values()),
                        "xz" if format == COMPACT_FORMAT else "zlib")
                    del rows
        if objects == old["objects"]:
            return parent  # Idempotent content, no artificial new version.
        return self._publish(old["contract"], old["bucket_count"], objects, parent, label, counts,
                             provenance or {}, format)

    def repack(self, snapshot, destination, interrupt=None):
        """Write a lossless compact copy; never replace, activate or prune sources.

        Re-encoding vectors is deliberately excluded. Metadata, tombstones and
        per-observation revisions must survive byte-for-byte canonical comparison.
        The destination is a new independent base; source lineage is recorded in
        provenance, rather than pretending its new object hashes are the old head.
        """
        import shutil
        target = Path(destination).expanduser().resolve()
        source = self.root.resolve()
        if target == source or source in target.parents or target in source.parents:
            raise ValueError("Choose a separate new directory outside the source store")
        if target.exists():
            raise FileExistsError(target)
        old = self.manifest(snapshot)
        target.mkdir(parents=True, exist_ok=False, mode=0o700)
        private_directory(target)
        result = Store(target)
        objects = {}
        counts = dict(observations=0, vectors=0, tombstones=0, source_records=0)
        source_bytes = self.manifest_path(snapshot).stat().st_size
        for index in sorted(map(int, old["objects"])):
            if interrupt:
                interrupt()
            rows = self.read_bucket(old, index)
            raw = pack(rows, "xz")
            disk = shutil.disk_usage(target)
            if disk.free - len(raw) - MAX_MANIFEST_BYTES < disk.total * .2:
                raise OSError("VRS repack would cross the 20% free disk reserve")
            oid = digest(raw)
            write_once(result.object_path(oid), raw)
            # Read the written file back before publishing anything. Verify the
            # hash and every field, including audit strings and vector bytes.
            written = _bounded_read(result.object_path(oid))
            if digest(written) != oid or canonical(unpack(written)) != canonical(rows):
                raise ValueError("Lossless bucket verification failed")
            objects[str(index)] = oid
            source_bytes += self.object_path(old["objects"][str(index)]).stat().st_size
            for entry in rows:
                if entry.get("deleted"):
                    counts["tombstones"] += 1
                else:
                    counts["observations"] += 1
                    counts["vectors"] += entry["vector"] is not None
                    counts["source_records"] += len(entry["audit"])
        if counts != old["counts"]:
            raise ValueError("Source snapshot counts mismatch")
        if interrupt:
            interrupt()
        provenance = dict(old["provenance"], repacked_from={
            "snapshot": snapshot, "format": old["format"], "lossless": True})
        new = result._publish(old["contract"], old["bucket_count"], objects, None,
                              old["label"], counts, provenance, COMPACT_FORMAT)
        return {**result.verify(new), "source_snapshot": snapshot,
                "source_snapshot_bytes": source_bytes, "store": str(target),
                "active": None, "compression": "xz", "lossless": True}

    def export(self, snapshot, destination, base=None):
        report = self.verify(snapshot)
        manifest = self.manifest(snapshot)
        if base is not None and manifest["parent"] != base:
            raise ValueError("Delta requires its exact immediate parent")
        known = set(self.manifest(base)["objects"].values()) if base else set()
        needed = sorted(set(manifest["objects"].values()) - known)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        private_directory(destination)
        package = Store(destination)
        for object_id in needed:
            write_once(package.object_path(object_id), self.object_path(object_id).read_bytes())
        write_once(package.manifest_path(snapshot), self.manifest_path(snapshot).read_bytes())
        write_once(destination / "package.json", canonical({"format": manifest["format"], "snapshot": snapshot,
                    "required_parent": base, "objects": needed}))
        return {**report, "transferred_objects": len(needed), "package_bytes": sum(
            p.stat().st_size for p in destination.rglob("*") if p.is_file())}

    def install(self, source):
        source = Path(source)
        package = json.loads(_bounded_read(source / "package.json", MAX_MANIFEST_BYTES))
        if package["format"] not in SUPPORTED_FORMATS:
            raise ValueError("Unsupported package format")
        snapshot, parent = package["snapshot"], package["required_parent"]
        incoming = Store(source)
        manifest = incoming.manifest(snapshot)
        if manifest["format"] != package["format"]:
            raise ValueError("Package and snapshot format mismatch")
        required = set(manifest["objects"].values())
        if parent is not None:
            existing = self.manifest(parent)
            if manifest["parent"] != parent or manifest["contract"] != existing["contract"] or manifest["bucket_count"] != existing["bucket_count"]:
                raise ValueError("Delta base or compatibility mismatch")
            self.verify(parent)
            required -= set(existing["objects"].values())
        if set(package["objects"]) != required or len(package["objects"]) != len(required):
            raise ValueError("Incomplete or unexpected package object inventory")
        for object_id in sorted(required):
            raw = _bounded_read(incoming.object_path(object_id))
            if digest(raw) != object_id:
                raise ValueError("Corrupt transfer object")
        # Validate the prospective snapshot before publishing its manifest.
        # Unchanged buckets come from the verified local parent.
        counts = {"observations": 0, "vectors": 0, "tombstones": 0, "source_records": 0}
        for index, object_id in manifest["objects"].items():
            owner = incoming if object_id in required else self
            for entry in owner.read_bucket(manifest, int(index)):
                if entry.get("deleted"):
                    counts["tombstones"] += 1
                else:
                    counts["observations"] += 1
                    counts["vectors"] += entry["vector"] is not None
                    counts["source_records"] += len(entry["audit"])
        if counts != manifest["counts"]:
            raise ValueError("Incoming snapshot counts mismatch")
        private_directory(self.root)
        for object_id in sorted(required):
            raw = _bounded_read(incoming.object_path(object_id))
            if digest(raw) != object_id:
                raise ValueError("Transfer changed during installation")
            write_once(self.object_path(object_id), raw)
        raw = _bounded_read(incoming.manifest_path(snapshot), MAX_MANIFEST_BYTES)
        if digest(raw) != snapshot:
            raise ValueError("Transfer manifest changed during installation")
        write_once(self.manifest_path(snapshot), raw)
        self.verify(snapshot)
        return snapshot  # Installation never changes the active pointer.

    def current(self):
        path = self.root / "CURRENT"
        return check_id(path.read_text().strip()) if path.exists() else None

    def activate(self, snapshot, expected_current):
        """Explicit compare-and-swap, also used for rollback to a retained base."""
        self.verify(snapshot)
        private_directory(self.root)
        lock = self.root / ".activation.lock"
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.close(fd)
            if self.current() != expected_current:
                raise ValueError("Active snapshot changed; re-read before activation")
            fd, temporary = tempfile.mkstemp(prefix=".current-", dir=self.root)
            try:
                with os.fdopen(fd, "w") as stream:
                    stream.write(snapshot + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.root / "CURRENT")
                _sync_directory(self.root)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        finally:
            lock.unlink()
