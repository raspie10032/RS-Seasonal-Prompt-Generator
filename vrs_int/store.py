"""Immutable hash-addressed snapshots and explicit, atomic local activation.

A snapshot is a flat map of stable key buckets, not a replay-dependent delta
chain. Packages carry only new compressed buckets. Old snapshots remain readable.
Hashes detect changes; they do NOT authenticate a publisher.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import zlib

from .vectors import decode_vector

FORMAT = "vrs-int/1"
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
    decoder = zlib.decompressobj()
    value = decoder.decompress(raw, MAX_OBJECT_BYTES + 1)
    if len(value) > MAX_OBJECT_BYTES or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid or oversized compressed object")
    return value


def pack(entries):
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
    return zlib.compress(raw, level=6)


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
    if not re.fullmatch(r"danbooru:[1-9][0-9]*", entry.get("key", "")):
        raise ValueError("Invalid observation key")
    if entry.get("deleted"):
        if set(entry) != {"key", "deleted", "previous_revision", "reason"} or not entry["reason"]:
            raise ValueError("Invalid tombstone")
        check_id(entry["previous_revision"])
        return
    if set(entry) != {"key", "document", "categories", "vector", "audit"}:
        raise ValueError("Invalid observation schema")
    doc = entry["document"]
    if entry["key"] != f"danbooru:{doc['id']}" or doc["rating"] not in ("g", "s", "q", "e"):
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

    def put_object(self, entries):
        raw = pack(entries)
        object_id = digest(raw)
        write_once(self.object_path(object_id), raw)
        return object_id

    def manifest(self, snapshot):
        raw = _bounded_read(self.manifest_path(snapshot), MAX_MANIFEST_BYTES)
        if digest(raw) != snapshot:
            raise ValueError("Snapshot hash mismatch")
        manifest = json.loads(raw)
        if manifest["format"] != FORMAT:
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

    def _publish(self, contract, count, objects, parent, label, counts, provenance):
        manifest = {"format": FORMAT, "contract": contract, "bucket_count": count,
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
        self.root.mkdir(parents=True, exist_ok=True)
        # Spool metadata, never original images. Only one bucket is packed in
        # RAM at a time, including at corpus sizes beyond this initial trial.
        with tempfile.TemporaryDirectory(prefix=".build-", dir=self.root) as staging:
            with sqlite3.connect(str(Path(staging) / "spool.sqlite3")) as db:
                db.execute("CREATE TABLE records (key TEXT PRIMARY KEY, bucket INTEGER, value BLOB)")
                for entry in entries:
                    validate_entry(entry, contract)
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
            db.close()
        provenance = dict(provenance or {}, canonical_input_jsonl_bytes=input_bytes)
        return self._publish(contract, bucket_count, objects, None, label, counts, provenance)

    def update(self, parent, operations, label, contract=None, provenance=None):
        """Compare revisions before replacing observations, never add extra votes.

        Expected revision None means a previously unseen key. Restoring a
        tombstone requires its own current revision. Any conflict aborts before
        publishing a snapshot. A changed encoder needs a new base.
        """
        old = self.manifest(parent)
        if contract is not None and contract != old["contract"]:
            raise ValueError("Incompatible update; re-encode into a separate base")
        groups, touched = {}, set()
        counts = dict(old["counts"])
        for op in operations:
            key = op["key"]
            if key in touched or "expected_revision" not in op:
                raise ValueError("Duplicate operation or missing expected revision")
            touched.add(key)
            index = bucket(key, old["bucket_count"])
            if index not in groups:
                groups[index] = {e["key"]: e for e in self.read_bucket(old, index)}
            previous = groups[index].get(key)
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
            for value, sign in ((previous, -1), (entry, 1)):
                if value is None:
                    continue
                if value.get("deleted"):
                    counts["tombstones"] += sign
                else:
                    counts["observations"] += sign
                    counts["vectors"] += sign * (value["vector"] is not None)
                    counts["source_records"] += sign * len(value["audit"])
            groups[index][key] = entry
        objects = dict(old["objects"])
        for index, rows in sorted(groups.items()):
            objects[str(index)] = self.put_object(list(rows.values()))
        if objects == old["objects"]:
            return parent  # Idempotent content, no artificial new version.
        return self._publish(old["contract"], old["bucket_count"], objects, parent, label, counts,
                             provenance or {})

    def export(self, snapshot, destination, base=None):
        report = self.verify(snapshot)
        manifest = self.manifest(snapshot)
        if base is not None and manifest["parent"] != base:
            raise ValueError("Delta requires its exact immediate parent")
        known = set(self.manifest(base)["objects"].values()) if base else set()
        needed = sorted(set(manifest["objects"].values()) - known)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        package = Store(destination)
        for object_id in needed:
            write_once(package.object_path(object_id), self.object_path(object_id).read_bytes())
        write_once(package.manifest_path(snapshot), self.manifest_path(snapshot).read_bytes())
        write_once(destination / "package.json", canonical({"format": FORMAT, "snapshot": snapshot,
                    "required_parent": base, "objects": needed}))
        return {**report, "transferred_objects": len(needed), "package_bytes": sum(
            p.stat().st_size for p in destination.rglob("*") if p.is_file())}

    def install(self, source):
        source = Path(source)
        package = json.loads(_bounded_read(source / "package.json", MAX_MANIFEST_BYTES))
        if package["format"] != FORMAT:
            raise ValueError("Unsupported package format")
        snapshot, parent = package["snapshot"], package["required_parent"]
        incoming = Store(source)
        manifest = incoming.manifest(snapshot)
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
        for object_id in sorted(required):
            raw = _bounded_read(incoming.object_path(object_id))
            if digest(raw) != object_id:
                raise ValueError("Transfer changed during installation")
            write_once(self.object_path(object_id), raw)
        write_once(self.manifest_path(snapshot), incoming.manifest_path(snapshot).read_bytes())
        self.verify(snapshot)
        return snapshot  # Installation never changes the active pointer.

    def current(self):
        path = self.root / "CURRENT"
        return check_id(path.read_text().strip()) if path.exists() else None

    def activate(self, snapshot, expected_current):
        """Explicit compare-and-swap, also used for rollback to a retained base."""
        self.verify(snapshot)
        self.root.mkdir(parents=True, exist_ok=True)
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
