"""Full local corpus, sparse GPU association memory. No downloading or model training.

Snapshots retain all source records, exact image hashes, tag categories and filter
decisions. Sparse GPU multiplication invokes the existing VRS CUDA judgment core.
The reader requires CUDA; no automatic CPU inference fallback.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:  # Windows can still import the pure data helpers.
    resource = None
import shutil
import sqlite3
import time

from experiments.tag_policy import POLICY_VERSION, filter_tags, typed_tags
from experiments.vrs_memory import canonical, contained, digest_file, row_digest


def emit(stage, **fields):
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_digest(path):
    before = path.stat()
    result = digest_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"File changed during hashing: {path}")
    return result


def source_rows(dataset):
    manifest = dataset / "manifest.jsonl"
    split_path = dataset / "processed_v1/manifest_visual_expand.jsonl"
    hashes = {str(p): stable_digest(p) for p in (manifest, split_path)}
    splits = {}
    with split_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = int(row["id"])
            if key in splits and splits[key] != row["split"]:
                raise ValueError(f"Conflicting split for {key}")
            splits[key] = row["split"]
    rows, hints = {}, defaultdict(set)
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = int(row["id"])
            if key in rows and row_digest(rows[key]) != row_digest(row):
                raise ValueError(f"Conflicting source ID {key}")
            rows[key] = row
            for category, tags in typed_tags(row).items():
                for tag in tags:
                    hints[tag].add(category)
    for path, expected in hashes.items():
        if digest_file(path) != expected:
            raise ValueError(f"Source changed while reading: {path}")
    return rows, splits, {t: sorted(c) for t, c in hints.items()}, hashes


def assigned_split(post_id, splits):
    if post_id in splits:
        return splits[post_id]
    # The 159 records omitted by the old caption policy still belong in memory.
    bucket = int(hashlib.sha256(f"rs-vrs-full-v1:{post_id}".encode()).hexdigest(), 16) % 100
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def ingest_full(dataset, output):
    rows, splits, hints, hashes = source_rows(dataset)
    write_json(output / "tag_categories.json", hints)
    conn = sqlite3.connect(output / "observations.sqlite3")
    conn.execute("""CREATE TABLE observations (
        id INTEGER PRIMARY KEY, canonical_id INTEGER NOT NULL, split TEXT NOT NULL,
        image TEXT NOT NULL, image_sha256 TEXT NOT NULL, source_sha256 TEXT NOT NULL,
        source_record TEXT NOT NULL, kept_tags TEXT NOT NULL, removed_tags TEXT NOT NULL,
        category_fields_present INTEGER NOT NULL)""")
    counts, removed_counts = Counter(), Counter()
    docs, representatives, by_representative = [], {}, {}
    ordered = sorted(rows, key=lambda i: ({"train": 0, "val": 1, "test": 2}[assigned_split(i, splits)], i))

    def inspect(key):
        row = rows[key]
        image = contained(dataset, row["image"])
        if not image.is_file():
            raise FileNotFoundError(f"Missing source image for {key}: {image}")
        return key, stable_digest(image)

    try:
        # Only two low-priority I/O readers, irrespective of system CPU count.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for batch_start in range(0, len(ordered), 256):
                for key, image_sha in pool.map(inspect, ordered[batch_start:batch_start + 256]):
                    row = rows[key]
                    kept, removed = filter_tags(row, hints)
                    split = assigned_split(key, splits)
                    representative = representatives.setdefault(image_sha, key)
                    conn.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)", (
                        key, representative, split, row["image"], image_sha, row_digest(row),
                        json.dumps(row, ensure_ascii=False, sort_keys=True),
                        json.dumps(kept, ensure_ascii=False), json.dumps(removed, ensure_ascii=False),
                        int("tags_artist" in row)))
                    counts["source_records"] += 1
                    counts["records_with_explicit_category_fields"] += int("tags_artist" in row)
                    counts["legacy_split_missing"] += int(key not in splits)
                    removed_counts.update(removed.keys())
                    if representative != key:
                        counts["duplicate_image_records"] += 1
                        # Multiple annotations of identical image bytes get one
                        # vote, while retaining rare identities from every record.
                        doc = by_representative[representative]
                        doc["tags"] = sorted(set(doc["tags"]) | set(kept))
                        doc["source_ids"].append(key)
                        doc["rating"] = max((doc["rating"], row["rating"]), key="gsqe".index)
                        continue
                    if not kept:
                        raise ValueError(f"No scene/core tags remain for {key}")
                    doc = {"id": key, "image": row["image"], "image_sha256": image_sha,
                           "split": split, "rating": row["rating"], "tags": kept, "source_ids": [key]}
                    docs.append(doc)
                    by_representative[key] = doc
                conn.commit()
                if batch_start % 4096 == 0:
                    emit("image_hash_and_ingest", processed=counts["source_records"], total=len(rows))
        conn.execute("CREATE INDEX image_digest ON observations(image_sha256)")
        conn.execute("CREATE INDEX representative ON observations(canonical_id)")
        conn.commit()
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Observation ledger integrity check failed")
    finally:
        conn.close()
    docs.sort(key=lambda r: r["id"])
    with (output / "documents.jsonl").open("w", encoding="utf-8") as f:
        for row in docs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {**counts, "documents": len(docs), "source_files_sha256": hashes,
              "unique_removed_tags": len(removed_counts), "removed_occurrences": sum(removed_counts.values()),
              "removed_tags": dict(removed_counts.most_common()), "filter_policy": POLICY_VERSION,
              "ratings": dict(Counter(r["rating"] for r in docs)),
              "splits": dict(Counter(r["split"] for r in docs)),
              "deduplication": "exact image SHA256; train representative preferred; union labels across identical image annotations; strictest rating; all original records retained",
              "legacy_missing_split": "stable ID hash: 98% train, 1% val, 1% test"}
    write_json(output / "ingest.json", report)
    return docs, hints, report


def membership_matrix(docs, output):
    import numpy as np
    from scipy import sparse
    vocab = sorted({tag for row in docs for tag in row["tags"]})
    index = {tag: i for i, tag in enumerate(vocab)}
    indices, ptr = [], [0]
    for row in docs:
        indices.extend(sorted(index[t] for t in row["tags"]))
        ptr.append(len(indices))
    x = sparse.csr_matrix((np.ones(len(indices), dtype=np.float32),
                           np.array(indices, dtype=np.int32), np.array(ptr, dtype=np.int32)),
                          shape=(len(docs), len(vocab)))
    sparse.save_npz(output / "membership.npz", x, compressed=False)
    write_json(output / "vocabulary.json", vocab)
    np.save(output / "ids.npy", np.array([r["id"] for r in docs], dtype=np.int64))
    return x, vocab


def gpu_csr(x):
    import torch
    return torch.sparse_csr_tensor(torch.from_numpy(x.indptr).cuda(),
                                   torch.from_numpy(x.indices).cuda(), torch.from_numpy(x.data).cuda(),
                                   size=x.shape, check_invariants=True)


def build_pairs(x, output, core, block_rows=128):
    import numpy as np
    import torch
    from experiments.vrs_cuda import AssociationCUDA
    if not 1 <= x.shape[0] < 2 ** 24:
        raise ValueError("FP32 exact counting requires fewer than 2**24 binary documents")
    target = output / "pairs"
    target.mkdir()
    tx = x.T.tocsr()
    gpu_x = gpu_csr(x)
    torch.backends.cuda.matmul.allow_tf32 = False
    pair_count, shard_rows = 0, []
    started = time.perf_counter()
    with AssociationCUDA(core) as judge:
        core_receipt = {"headers_sha256": judge.headers_sha256, "ptx_sha256": judge.ptx_sha256}
        for start in range(0, x.shape[1], block_rows):
            stop = min(start + block_rows, x.shape[1])
            a = gpu_csr(tx[start:stop])
            co = torch.sparse.mm(a, gpu_x)
            values = co.values().to(torch.int64)
            status, reason = judge.judge(values, torch.zeros_like(values))
            if not bool((status == 1).all()):
                raise ValueError("Positive recorded co-presence did not receive VRS A")
            # Diagonal stores frequency, never offered as a new candidate.
            name = f"pairs/{start:07d}.npz"
            np.savez(output / name, indptr=co.crow_indices().cpu().numpy(),
                     indices=co.col_indices().cpu().numpy(), counts=values.to(torch.int32).cpu().numpy(),
                     status=status.cpu().numpy(), reason=reason.cpu().numpy(),
                     row_start=start, row_stop=stop)
            nnz = values.numel()
            pair_count += nnz
            shard_rows.append({"file": name, "start": start, "stop": stop, "nonzero": nnz})
            del a, co, values, status, reason
            if start % (block_rows * 16) == 0:
                emit("gpu_pairs", tags_done=stop, tags_total=x.shape[1], nonzero=pair_count)
    del gpu_x
    torch.cuda.empty_cache()
    result = {"core": core_receipt, "shards": shard_rows,
              "directed_pairs_excluding_self": pair_count - x.shape[1],
              "seconds": time.perf_counter() - started,
              "unobserved_pairs": "implicit U; no negative visual evidence inferred"}
    write_json(output / "pairs.json", result)
    return result


def pool_features(docs, dataset, output):
    import numpy as np
    import torch
    target = np.lib.format.open_memmap(output / "vectors.npy", mode="w+", dtype=np.float16,
                                       shape=(len(docs), 768))
    mask = np.zeros(len(docs), dtype=np.bool_)
    counts = Counter()
    started = time.perf_counter()

    def read(i):
        path = contained(dataset, f"img_embeds_pre/{docs[i]['id']}.pt")
        if not path.is_file():
            return i, None, None
        # Hash and deserialize the same bytes; never load arbitrary pickle objects.
        import io
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        feature = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(feature, torch.Tensor) or feature.ndim != 2 or feature.shape[1] != 768:
            raise ValueError(f"Invalid feature shape: {path}")
        return i, feature, sha

    with (output / "features.jsonl").open("w") as log, ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, len(docs), 64):
            items = list(pool.map(read, range(start, min(start + 64, len(docs)))))
            present = [(i, f, sha) for i, f, sha in items if f is not None]
            if present:
                # Variable token lengths: pad only this bounded batch on the GPU.
                length = max(f.shape[0] for _, f, _ in present)
                if length < 1 or length > 4096:
                    raise ValueError("Unbounded or empty feature token count")
                batch = torch.zeros((len(present), length, 768), device="cuda", dtype=torch.float32)
                lengths = []
                for j, (_, f, _) in enumerate(present):
                    batch[j, :f.shape[0]] = f.to(device="cuda", dtype=torch.float32)
                    lengths.append(f.shape[0])
                pooled = batch.sum(dim=1) / torch.tensor(lengths, device="cuda")[:, None]
                if not bool(torch.isfinite(pooled).all()) or bool((pooled.norm(dim=1) == 0).any()):
                    raise ValueError("Nonfinite or zero visual feature")
                vectors = torch.nn.functional.normalize(pooled, dim=1).to(torch.float16).cpu().numpy()
                for j, (i, f, sha) in enumerate(present):
                    target[i] = vectors[j]
                    mask[i] = True
                    log.write(json.dumps({"id": docs[i]["id"], "cache_sha256": sha,
                                          "shape": list(f.shape)}) + "\n")
                del batch, pooled, vectors
            counts["pooled"] += len(present)
            counts["missing_cache_kept_as_tag_observation"] += len(items) - len(present)
            if start % 4096 == 0:
                emit("gpu_feature_pool", processed=start + len(items), total=len(docs), **counts)
    target.flush()
    del target
    np.save(output / "vector_mask.npy", mask)
    torch.cuda.empty_cache()
    return {**counts, "seconds": time.perf_counter() - started,
            "description": "GPU mean + normalize, FP16 768-dimensional vectors; legacy encoder provenance unverified; not text aligned"}


class FullMemory:
    def __init__(self, directory, verify=True, split=None, building_audit=False):
        import numpy as np
        import torch
        from scipy import sparse
        self.path = Path(directory)
        self.receipt = json.loads((self.path / "receipt.json").read_text())
        if self.receipt.get("state") != "complete" and not building_audit:
            raise ValueError("Incomplete snapshot")
        if verify:
            for name, digest in self.receipt["files_sha256"].items():
                if digest_file(self.path / name) != digest:
                    raise ValueError(f"Snapshot digest mismatch: {name}")
        self.rows = [json.loads(l) for l in (self.path / "documents.jsonl").open()]
        self.vocab = json.loads((self.path / "vocabulary.json").read_text())
        self.categories = json.loads((self.path / "tag_categories.json").read_text())
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.x = sparse.load_npz(self.path / "membership.npz")
        if split:
            selected = np.array([r["split"] == split for r in self.rows])
            self.rows = [r for r in self.rows if r["split"] == split]
            self.x = self.x[selected]
        self.xt = self.x.T.tocsr()
        self.gpu_xt = gpu_csr(self.xt)
        self.freq = torch.from_numpy(np.diff(self.xt.indptr)).cuda()

    def match(self, seeds, bans=(), ratings=()):
        import torch
        match = torch.ones(len(self.rows), device="cuda", dtype=torch.bool)
        for tag in seeds:
            i = self.index[tag]
            present = torch.zeros_like(match)
            present[torch.from_numpy(self.xt.indices[self.xt.indptr[i]:self.xt.indptr[i + 1]]).cuda().long()] = True
            match &= present
        for tag in bans:
            if tag in self.index:
                i = self.index[tag]
                match[torch.from_numpy(self.xt.indices[self.xt.indptr[i]:self.xt.indptr[i + 1]]).cuda().long()] = False
        if ratings:
            match &= torch.tensor([r["rating"] in ratings for r in self.rows], device="cuda")
        return match

    def query(self, tags, excluded=(), limit=10, min_support=2, ratings=(), judge=None):
        import torch
        if not 1 <= min_support:
            raise ValueError("min_support must be positive")
        seeds = sorted({canonical(t) for t in tags if t.strip()})
        bans = {canonical(t) for t in excluded if t.strip()}
        if set(seeds) & bans:
            raise ValueError("A tag cannot be required and excluded")
        unknown = [t for t in seeds if t not in self.index]
        result = {"input_tags": seeds, "excluded": sorted(bans), "ratings": list(ratings),
                  "unknown_tags": unknown, "verdict": "U", "matching_records": 0,
                  "candidates": [], "examples": [],
                  "scope": "recorded co-presence only; not permission to add a tag; absent labels are not visual refutations"}
        if not seeds or unknown:
            return result
        match = self.match(seeds, bans, ratings)
        positions = torch.nonzero(match).flatten().cpu().tolist()
        result["matching_records"] = len(positions)
        if not positions:
            return result
        support = torch.sparse.mm(self.gpu_xt, match.float()[:, None]).flatten().to(torch.int64)
        support[[self.index[t] for t in set(seeds) | bans if t in self.index]] = 0
        top = torch.argsort(support, descending=True, stable=True)[:max(0, min(limit, 50))]
        if judge is None:
            from experiments.vrs_cuda import AssociationCUDA
            with AssociationCUDA(self.receipt["core_root"]) as own:
                status, _ = own.judge(support[top], torch.zeros_like(support[top]))
        else:
            status, _ = judge.judge(support[top], torch.zeros_like(support[top]))
        for j, count, state in zip(top.cpu().tolist(), support[top].cpu().tolist(), status.cpu().tolist()):
            if count < min_support or state != 1:
                continue
            tag = self.vocab[j]
            evidence = [self.rows[i]["id"] for i in positions if tag in self.rows[i]["tags"]][:3]
            result["candidates"].append({"tag": tag, "support": count,
                "conditional_rate": count / len(positions), "image_ids": evidence,
                "categories": self.categories.get(tag, ["unknown"]), "vrs": "A"})
        result["examples"] = [{k: self.rows[i][k] for k in ("id", "image", "rating")} for i in positions[:3]]
        if result["candidates"]:
            result["verdict"] = "A: recorded co-presence"
        return result


def evaluate_train(memory, held_out):
    import numpy as np
    import torch
    train_hashes = {r["image_sha256"] for r in memory.rows}
    results, skipped = [], Counter()
    for row in held_out:
        if row["image_sha256"] in train_hashes:
            raise ValueError("Exact-image test leakage")
        known = [t for t in row["tags"] if memory.freq[memory.index[t]].item() > 0]
        known.sort(key=lambda t: hashlib.sha256(f"{row['id']}:{t}".encode()).digest())
        if len(known) < 8:
            skipped["under_eight_known_tags"] += 1
            continue
        seeds, target = known[:3], set(known[3:])
        ix = [memory.index[t] for t in seeds]
        # Pairwise counts recomputed from TRAIN rows only, never the full graph.
        masks = []
        for tag in seeds:
            masks.append(memory.match([tag]).float())
        seed_masks = torch.stack(masks, dim=1)
        co = torch.sparse.mm(memory.gpu_xt, seed_masks)
        assoc = (co / memory.freq[ix].float()[None, :]).mean(dim=1)
        strict = torch.sparse.mm(memory.gpu_xt, seed_masks.bool().all(dim=1).float()[:, None]).flatten()
        baseline = memory.freq.float().clone()
        methods = {}
        for name, scores in (("frequency", baseline), ("association", assoc), ("strict_query", strict)):
            scores[ix] = -1
            scores[memory.freq == 0] = -1
            if name == "strict_query":
                scores[scores < 2] = -1
            top = torch.argsort(scores, descending=True, stable=True)[:10]
            pred = [memory.vocab[i] for i in top.cpu().tolist() if scores[i].item() >= 0]
            hits = len(set(pred) & target)
            methods[name] = {"precision_at_10": hits / 10, "recall_at_10": hits / len(target), "tags": pred}
        results.append({"id": row["id"], "rating": row["rating"], "seeds": seeds, "target_count": len(target),
                        "oov": len(row["tags"]) - len(known), "total": len(row["tags"]), **methods})
        if len(results) % 128 == 0:
            emit("held_out_evaluation", cases=len(results), total=len(held_out))
    def summarize(selected):
        if not selected:
            return {"cases": 0}
        return {"cases": len(selected), "strict_query_coverage": sum(bool(r["strict_query"]["tags"]) for r in selected) / len(selected),
                "oov_fraction": sum(r["oov"] for r in selected) / sum(r["total"] for r in selected),
                "means": {m: {key: float(np.mean([r[m][key] for r in selected]))
                               for key in ("precision_at_10", "recall_at_10")}
                          for m in ("frequency", "association", "strict_query")}}
    return {"scope": "Train-only, exact image SHA disjoint test; three seeds; in-vocabulary masked tag recovery, not prompt quality; near duplicates not audited",
            **summarize(results), "by_rating": {r: summarize([x for x in results if x["rating"] == r]) for r in "gsqe"},
            "skipped": dict(skipped), "results": results}


def audit_snapshot(output, x, vocab, docs, hints):
    import numpy as np
    from scipy import sparse
    rng = np.random.default_rng(20261011)
    manifest = json.loads((output / "pairs.json").read_text())
    checked = 0
    for shard in manifest["shards"][::max(1, len(manifest["shards"]) // 32)]:
        with np.load(output / shard["file"], allow_pickle=False) as d:
            co = sparse.csr_matrix((d["counts"], d["indices"], d["indptr"]),
                                    shape=(shard["stop"] - shard["start"], x.shape[1]))
            for _ in range(4):
                row = int(rng.integers(co.shape[0]))
                start, stop = co.indptr[row:row + 2]
                j = int(co.indices[int(rng.integers(start, stop))])
                i = shard["start"] + row
                expected = int(x[:, i].multiply(x[:, j]).sum())
                if int(co[row, j]) != expected:
                    raise ValueError(f"Independent pair count failed: {i},{j}")
                checked += 1
    counts = np.asarray(x.sum(axis=0)).flatten()
    identity = {}
    for cat in ("artist", "character", "copyright"):
        indices = [i for i, t in enumerate(vocab) if cat in hints.get(t, ())]
        identity[cat] = {"tags": len(indices), "singletons": sum(int(counts[i] == 1) for i in indices),
                         "documents": sum(any(cat in hints.get(t, ()) for t in r["tags"]) for r in docs)}
    return {"independent_pair_counts_checked": checked, "all_passed": True,
            "identity_categories": identity, "singleton_tags_kept": int((counts == 1).sum()),
            "vocabulary_cap": None, "minimum_document_frequency": 1}


def run_build(args):
    import torch
    from experiments.run_vrs_trial import Monitor, parity, setup_gpu
    setup_gpu()
    output, dataset = Path(args.output).resolve(), Path(args.dataset).resolve()
    disk = shutil.disk_usage(output.parent)
    if disk.free / disk.total < .20:
        raise RuntimeError("Keep at least 20% disk free")
    output.mkdir(exist_ok=False)
    started, cpu_start = time.perf_counter(), time.process_time()
    source_hashes = {p.name: digest_file(p) for p in Path(__file__).parent.glob("*.py")}
    report = {"schema": "rs-vrs-full-v1", "state": "building", "core_root": str(Path(args.vrs_core).resolve()),
              "dataset": str(dataset), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "code_sha256": source_hashes, "gpu": torch.cuda.get_device_name(),
              "scope": "Full local unified manifest; GPU association core + exact observations + legacy feature pooling; no VRS resident runtime or new neural training"}
    write_json(output / "receipt.json", report)
    with Monitor() as monitor:
        report["core_parity"] = parity(args.vrs_core, output)
        docs, hints, report["ingest"] = ingest_full(dataset, output)
        x, vocab = membership_matrix(docs, output)
        emit("matrix_ready", documents=x.shape[0], vocabulary=x.shape[1], tag_memberships=x.nnz)
        report["pairs"] = build_pairs(x, output, args.vrs_core)
        report["features"] = pool_features(docs, dataset, output)
        report["audit"] = audit_snapshot(output, x, vocab, docs, hints)
        report["documents"], report["vocabulary"] = x.shape
        write_json(output / "receipt.json", report)
        try:
            memory = FullMemory(output, verify=False, split="train", building_audit=True)
            evaluation = evaluate_train(memory, [r for r in docs if r["split"] == "test"])
            write_json(output / "evaluation.json", evaluation)
            report["evaluation_summary"] = {k: v for k, v in evaluation.items() if k != "results"}
            del memory
            torch.cuda.empty_cache()
            memory = FullMemory(output, verify=False, building_audit=True)
            queries = [memory.query(["hatsune miku"], ratings=("g", "s")),
                       memory.query(["blue archive"], ratings=("g", "s")),
                       memory.query(["umbrella", "subway platform"], ratings=("g",)),
                       memory.query(["nonexistent rs character 2099"])]
            write_json(output / "queries.json", queries)
            refs = 0
            by_id = {r["id"]: r for r in docs}
            for q in queries:
                for c in q["candidates"]:
                    for key in c["image_ids"]:
                        row = by_id[key]
                        if not set(q["input_tags"] + [c["tag"]]).issubset(row["tags"]):
                            raise ValueError("Invalid query evidence reference")
                        if q["ratings"] and row["rating"] not in q["ratings"]:
                            raise ValueError("Query rating constraint failed")
                        refs += 1
            report["audit"]["query_evidence_references_checked"] = refs
            del memory
        finally:
            write_json(output / "receipt.json", {**report, "state": "building"})
        report["resources"] = {"wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_start,
            "rss_max_mib": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource else None),
            "torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "nvidia_smi_peak_sample_mib": max((s["mib"] for s in monitor.samples), default=None),
            "gpu_uuids": sorted({s["uuid"] for s in monitor.samples}),
            "cpu_affinity": (sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None), "cpu_threads": torch.get_num_threads(),
            "torch_memory_fraction_limit": .20, "gpu_sample_interval_seconds": .5}
        write_json(output / "gpu_samples.json", monitor.samples)
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "receipt.json")
    report["files_sha256"] = {str(p.relative_to(output)): digest_file(p) for p in files}
    report["artifact_bytes"] = sum(p.stat().st_size for p in files)
    report["state"] = "complete"
    write_json(output / "receipt.json", report)
    emit("complete", receipt=str(output / "receipt.json"), resources=report["resources"],
         documents=report["documents"], vocabulary=report["vocabulary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--dataset", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--vrs-core", required=True)
    query = sub.add_parser("query")
    query.add_argument("--memory", required=True)
    query.add_argument("--tags", required=True)
    query.add_argument("--exclude", default="")
    query.add_argument("--ratings", default="g,s", help="Comma-separated g,s,q,e; empty means all")
    query.add_argument("--min-support", type=int, default=2)
    args = parser.parse_args()
    if args.command == "build":
        run_build(args)
    else:
        from experiments.run_vrs_trial import setup_gpu
        setup_gpu()
        result = FullMemory(args.memory).query(args.tags.split(","), args.exclude.split(","),
                                                min_support=args.min_support,
                                                ratings=tuple(t for t in args.ratings.split(",") if t))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
