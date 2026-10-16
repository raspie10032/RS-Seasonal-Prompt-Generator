"""Private, incremental folder catalog. No GPU work or workflow execution here."""
from contextlib import contextmanager
from itertools import islice
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .generated import EXTENSIONS, _scan_image, compact, expand
from .generation_metadata import VERSION
from .store import Store


def collection_path(root, collection_id):
    if str(uuid.UUID(collection_id)) != collection_id:
        raise ValueError("Invalid collection ID")
    return Path(root) / collection_id


class FolderCatalog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        self.memory = self.directory / "memory"
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,size INTEGER,mtime INTEGER,status TEXT,sha TEXT,detail TEXT)")
            db.execute("CREATE INDEX IF NOT EXISTS files_by_status ON files(status)")
            db.execute("CREATE TABLE IF NOT EXISTS records(sha TEXT PRIMARY KEY,path TEXT,record BLOB,state TEXT,detail TEXT)")
            db.execute("CREATE INDEX IF NOT EXISTS by_state ON records(state)")
            db.execute("CREATE TABLE IF NOT EXISTS staged(sha TEXT PRIMARY KEY,entry BLOB,created REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)")
            parser = db.execute("SELECT value FROM meta WHERE key='parser'").fetchone()
            if parser and parser[0] != VERSION:
                raise ValueError("Metadata parser changed; create a new collection to preserve the old one")
            db.execute("INSERT OR IGNORE INTO meta VALUES('parser',?)", (VERSION,))

    @classmethod
    def create(cls, root, roots, model_name, *, label="Personal images", batch_size=4):
        paths = sorted({str(Path(p).expanduser().resolve(strict=True)) for p in roots if str(p).strip()})
        if not paths or any(not Path(p).is_dir() for p in paths):
            raise ValueError("Select at least one existing server-side image folder")
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("Batch size must be 1..32")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Select the native Gemma vision model")
        # Collapse nested roots so output inside a selected parent is scanned once.
        paths = [p for p in paths if not any(Path(q) in Path(p).parents for q in paths if q != p)]
        cid = str(uuid.uuid4())
        directory = collection_path(root, cid)
        directory.mkdir(parents=True)
        config = dict(id=cid, roots=paths, model_name=model_name, label=str(label)[:120], batch_size=batch_size)
        (directory / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return cls(directory)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.directory / "catalog.sqlite3", timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _images(self, stopped):
        """Discover regular image paths without following directory symlinks."""
        for root in self.config["roots"]:
            if not Path(root).is_dir():
                raise FileNotFoundError(f"Image folder unavailable: {root}")
            def walk_error(error):
                raise error
            for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
                if stopped():
                    return
                dirs[:] = sorted(d for d in dirs if not Path(directory, d).is_symlink())
                for name in sorted(names):
                    if stopped():
                        return
                    image = Path(directory, name)
                    if image.suffix.lower() in EXTENSIONS and not image.is_symlink():
                        yield image

    def scan_steps(self, stopped=lambda: False, settle_seconds=5):
        """Read/commit at most 32 files per connection, then yield progress.

        Polling avoids platform-specific watchers, watches recursively, and never
        follows symlinks. Source disappearance never deletes existing experience.
        No connection or transaction survives a yield: asyncio.to_thread may
        resume this generator on a different executor thread. Parsing happens
        before the short write transaction, allowing concurrent GPU commits.
        """
        images = self._images(stopped)
        while paths := list(islice(images, 32)):
            progress, changes = [], []
            with self.db() as db:
                for image in paths:
                    if stopped():
                        break
                    try:
                        before = image.stat()
                    except FileNotFoundError:
                        continue
                    if time.time() - before.st_mtime < settle_seconds:
                        progress.append(None)
                        continue
                    old = db.execute("SELECT size,mtime,status FROM files WHERE path=?", (str(image),)).fetchone()
                    if old and old[:2] == (before.st_size, before.st_mtime_ns) and old[2] != "error":
                        progress.append(None)
                        continue
                    try:
                        change = _scan_image(str(image))
                    except FileNotFoundError:
                        continue
                    changes.append(change)
                    progress.append(change[0])
                for path, size, mtime, status, sha, detail, payload in changes:
                    if payload:
                        added = db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,'pending',NULL)",
                                           (sha, path, payload)).rowcount
                        if not added:
                            status = "duplicate-bytes"
                            # A newly found surviving copy can recover a failed source.
                            db.execute("UPDATE records SET path=?,state='pending',detail=NULL "
                                       "WHERE sha=? AND state IN ('pending','error')", (path, sha))
                    db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?)",
                               (path, size, mtime, status, sha, detail))
            yield from progress
            if stopped():
                return

    def has_pending(self):
        """Check the indexed state without decompressing an observation."""
        with self.db() as db:
            return db.execute("SELECT 1 FROM records WHERE state='pending' LIMIT 1").fetchone() is not None

    def pending(self, limit):
        with self.db() as db:
            return [(sha, Path(path), expand(record)) for sha, path, record in db.execute(
                "SELECT sha,path,record FROM records WHERE state='pending' ORDER BY rowid LIMIT ?", (limit,))]

    def failed(self, sha, error):
        with self.db() as db:
            db.execute("UPDATE records SET state='error',detail=? WHERE sha=?", (str(error)[:500], sha))

    def retry(self):
        with self.db() as db:
            db.execute("UPDATE records SET state='pending',detail=NULL WHERE state='error'")

    def snapshot(self):
        with self.db() as db:
            row = db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()
        return row[0] if row else None

    def search_snapshot(self):
        """Last fully prepared head; old collections fall back to lazy preparation."""
        with self.db() as db:
            row = db.execute("SELECT value FROM meta WHERE key='search_snapshot'").fetchone()
        return row[0] if row else self.snapshot()

    def search_ready(self, snapshot):
        # A newer publisher must never be moved backwards by a late preparer.
        manifest = Store(self.memory).manifest(snapshot)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            head = db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()
            if not head or head[0] != snapshot:
                return False
            db.execute("INSERT OR REPLACE INTO meta VALUES('search_snapshot',?)", (snapshot,))
            db.execute("INSERT OR REPLACE INTO meta VALUES('search_count',?)",
                       (str(manifest['counts']['observations']),))
        return True

    def stage(self, entries, shas, contract):
        """Durably checkpoint encoded images without publishing a whole snapshot."""
        if len(entries) != len(shas):
            raise ValueError("Encoded image count mismatch")
        text = json.dumps(contract, sort_keys=True)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT value FROM meta WHERE key='encoder'").fetchone()
            if previous and previous[0] != text:
                raise ValueError("Vision model/runtime changed; create a new collection")
            db.execute("INSERT OR IGNORE INTO meta VALUES('encoder',?)", (text,))
            for sha, entry in zip(shas, entries):
                row = db.execute("SELECT state FROM records WHERE sha=?", (sha,)).fetchone()
                if not row or row[0] != "pending":
                    raise ValueError("Image is no longer pending")
                db.execute("INSERT INTO staged VALUES(?,?,?)", (sha, compact(entry), time.time()))
                db.execute("UPDATE records SET state='staged',detail=NULL WHERE sha=?", (sha,))

    def publish(self, *, force=False):
        """Publish the first result, 128 images, 60 seconds, or a final drain.

        Only one FolderService owns this catalog. CAS in commit still rejects
        competing publishers. Interrupted publication retains durable staging;
        old snapshots and their referenced buckets remain untouched.
        """
        base = self.snapshot()
        with self.db() as db:
            count, oldest = db.execute("SELECT count(*),min(created) FROM staged").fetchone()
            if not count or (base and not force and count < 128 and time.time() - oldest < 60):
                return None
            contract = json.loads(db.execute("SELECT value FROM meta WHERE key='encoder'").fetchone()[0])
            rows = list(db.execute("SELECT sha,entry FROM staged ORDER BY rowid"))
        store = Store(self.memory)
        entries = [expand(entry) for _, entry in rows]
        provenance = {"collection": "comfyui-folder-import", "images_included": False}
        if base:
            operations = (dict(op="upsert", key=entry["key"], entry=entry, expected_revision=None)
                          for entry in entries)
            snapshot = store.update(base, operations, self.config["label"], contract, provenance)
        else:
            snapshot = store.build(entries, contract, self.config["label"], provenance)
        self.commit(base, snapshot, [sha for sha, _ in rows])
        return snapshot

    def commit(self, previous, snapshot, shas):
        # Immutable buckets are written first; this single transaction advances
        # the head and processed ledger together. A crash leaves only orphan
        # objects, never 'processed' records missing from the selected snapshot.
        Store(self.memory).manifest(snapshot)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()
            if (row[0] if row else None) != previous:
                raise ValueError("Collection advanced concurrently; retry from its new snapshot")
            db.execute("INSERT OR REPLACE INTO meta VALUES('snapshot',?)", (snapshot,))
            db.executemany("UPDATE records SET state='encoded',detail=NULL WHERE sha=?", ((s,) for s in shas))
            db.executemany("DELETE FROM staged WHERE sha=?", ((s,) for s in shas))

    def status(self):
        with self.db() as db:
            files = dict(db.execute("SELECT status,count(*) FROM files GROUP BY status"))
            records = dict(db.execute("SELECT state,count(*) FROM records GROUP BY state"))
            errors = [dict(path=p, detail=d) for p, d in db.execute(
                "SELECT path,detail FROM records WHERE state='error' LIMIT 10")]
            search = dict(db.execute("SELECT key,value FROM meta WHERE key IN ('search_snapshot','search_count')"))
        return dict(config=self.config, store_path=str(self.memory), snapshot=self.snapshot(),
                    search_snapshot=search.get("search_snapshot"), search_count=int(search.get("search_count", 0)),
                    files=files, records=records, errors=errors)
