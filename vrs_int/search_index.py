"""Exact snapshot row selection over a persistent, shared vector pool."""
from contextlib import closing
import json
import mmap
import os
import sqlite3
import zlib

import numpy as np

from .search_cache import META, PersistentCache, cache_root

USE_MMAP = os.name != "nt"


class MappedVectors:
    """Snapshot-local order over append-only physical rows; copies only a batch."""
    def __init__(self, path, rows, count, dimension):
        self.path, self.rows, self.dimension = path, rows, dimension
        self.closed = False
        # Windows forbids truncating a mapped file, including rollback after a
        # cancelled append. Read only requested rows there, without retaining a
        # mapping/handle that prevents another snapshot from being prepared.
        self.array = np.memmap(path, mode="r", dtype="<f4", shape=(count, dimension)) if USE_MMAP else None

    def __getitem__(self, selected):
        if self.closed:
            raise ValueError("Vector view is closed")
        physical = np.asarray(self.rows[selected])
        if self.array is not None:
            return self.array[physical]
        unique, inverse = np.unique(physical.reshape(-1), return_inverse=True)
        values = np.empty((len(unique), self.dimension), dtype="<f4")
        # Coalesce contiguous rows, then restore requested order and duplicates.
        # Memory and file reads are bounded by this batch, never the whole pool.
        boundaries = np.r_[0, np.flatnonzero(np.diff(unique) != 1) + 1, len(unique)]
        row_bytes = self.dimension * 4
        with self.path.open("rb") as stream:
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                if begin == end:
                    continue
                stream.seek(int(unique[begin]) * row_bytes)
                raw = stream.read(int(end - begin) * row_bytes)
                if len(raw) != (end - begin) * row_bytes:
                    raise OSError("Truncated VRS vector cache")
                values[begin:end] = np.frombuffer(raw, dtype="<f4").reshape(-1, self.dimension)
        return values[inverse].reshape(physical.shape + (self.dimension,))

    def release_pages(self):
        if self.array is not None and hasattr(mmap, "MADV_DONTNEED") and hasattr(self.array._mmap, "madvise"):
            try:
                self.array._mmap.madvise(mmap.MADV_DONTNEED)
            except OSError:
                pass

    def close(self):
        if self.array is not None:
            self.array._mmap.close()
            self.array = None
        self.closed = True


class SearchIndex:
    def __init__(self, store, snapshot, interrupt=None):
        self._cache = None
        self.vectors = self.meta = None
        self.count = 0
        self.duplicates = np.empty(0, dtype=np.int64)
        manifest = store.manifest(snapshot)
        try:
            self._cache = PersistentCache(store, manifest, interrupt)
            self.path = self._cache.path
            self.stats = self._cache.stats
            self._rows = np.asarray(self._cache.rows, dtype=np.int64)
            self._cache.rows = []
            self.count = len(self._rows)
            self.meta = np.empty(self.count, dtype=META)
            keys = []
            with closing(sqlite3.connect(self.path / "votes.sqlite3")) as db:
                for start in range(0, self.count, 512):
                    if interrupt:
                        interrupt()
                    ids = self._rows[start:start + 512].tolist()
                    records = {row: (key, raw) for row, key, raw in db.execute(
                        "SELECT id,key,meta FROM records WHERE id IN (" + ",".join("?" * len(ids)) + ")", ids)}
                    for offset, row in enumerate(ids):
                        key, raw = records[row]
                        self.meta[start + offset] = np.frombuffer(raw, dtype=META)[0]
                        keys.append(key)
            # Exact lexical order, including arbitrarily long IDs, is local to
            # this snapshot. New/deleted rows never change an already-open view.
            order = sorted(range(self.count), key=keys.__getitem__)
            self.meta["order"][order] = np.arange(self.count)
            self.meta.flags.writeable = False
            if self.count:
                self.vectors = MappedVectors(self.path / "vectors.f32", self._rows,
                                             self._cache.pool_count, manifest["contract"]["dimension"])
                _, inverse, counts = np.unique(self.meta["image"], return_inverse=True, return_counts=True)
                self.duplicates = np.flatnonzero(counts[inverse] > 1)
            if interrupt:
                interrupt()
        except BaseException:
            self.close()
            raise

    def eligible(self, image_sha256, pixel_sha256, ratings):
        if not self.count:
            return np.empty(0, dtype=np.int64)
        mask = np.ones(self.count, dtype=bool)
        # Avoid NumPy's silent fixed-width truncation of caller filter values.
        if isinstance(image_sha256, str) and len(image_sha256) == 64 and image_sha256.isascii():
            mask &= self.meta["image"] != image_sha256.encode("ascii")
        if isinstance(pixel_sha256, str) and len(pixel_sha256) == 64 and pixel_sha256.isascii():
            mask &= self.meta["pixel"] != pixel_sha256.encode("ascii")
        if ratings:
            mask &= np.isin(self.meta["rating"], [r.encode("ascii") for r in ratings])
        # Deduplicate AFTER query/rating/pixel filters, retaining the first
        # eligible source in the original bucket/key traversal order.
        duplicate_rows = self.duplicates[mask[self.duplicates]]
        if len(duplicate_rows):
            _, first = np.unique(self.meta["image"][duplicate_rows], return_index=True)
            mask[duplicate_rows] = False
            mask[duplicate_rows[first]] = True
        return np.flatnonzero(mask)

    def documents(self, rows):
        ids = [int(self._rows[row]) for row in rows]
        documents = {}
        with closing(sqlite3.connect(self.path / "votes.sqlite3")) as db:
            # Stay below SQLite's older 999-variable limit. Restore the caller's
            # exact order and duplicates after fetching only requested winners.
            unique = list(dict.fromkeys(ids))
            for start in range(0, len(unique), 512):
                batch = unique[start:start + 512]
                for row, raw in db.execute(
                        "SELECT id,value FROM records WHERE id IN (" + ",".join("?" * len(batch)) + ")", batch):
                    documents[row] = json.loads(zlib.decompress(raw))
        return [documents[row] for row in ids]

    def release_pages(self):
        if self.vectors is not None:
            self.vectors.release_pages()

    def close(self):
        if self.vectors is not None:
            self.vectors.close()
            self.vectors = None
        self.meta = None
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __del__(self):
        self.close()


def best_in_block(scores, keys, count):
    """Exact top-k, including ALL cutoff ties before lexical key selection."""
    if len(scores) <= count:
        selected = np.arange(len(scores))
    else:
        cutoff = np.partition(scores, len(scores) - count)[len(scores) - count]
        selected = np.flatnonzero(scores >= cutoff)
    return selected[np.lexsort((keys[selected], -scores[selected]))[:count]]
