"""Small, local Danbooru association experiment. No downloader or scheduler.

The immutable observations are recorded tag membership, not visual truth.
CUDA builds exact small integer co-counts and invokes the existing VRS core.
Pooled legacy visual features support image-neighbor inspection only: they are
NOT text-aligned embeddings and their original encoder is still unverified.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(tag):
    return " ".join(tag.lower().replace("_", " ").split())


def row_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def selection_key(row):
    return hashlib.sha256(f"rs-vrs-trial-v1:{row['id']}".encode()).digest()


def read_rows(manifest, split="train", rating="g"):
    rows = {}
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != split or row.get("rating") != rating:
                continue
            post_id = int(row["id"])
            if post_id in rows and row_digest(row) != row_digest(rows[post_id]):
                raise ValueError(f"Conflicting manifest records for post {post_id}")
            rows[post_id] = row
    return sorted(rows.values(), key=selection_key)


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path escapes its root")
    return path


def open_ledger(path):
    conn = sqlite3.connect(path)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY, record_sha256 TEXT NOT NULL,
            image_sha256 TEXT NOT NULL UNIQUE, cache_sha256 TEXT NOT NULL,
            record TEXT NOT NULL, vector BLOB NOT NULL);
    ''')
    return conn


def record_observation(conn, row, image_sha, cache_sha, vector):
    """An identical replay adds nothing; revisions need explicit replacement policy."""
    identity = row_digest(row)
    old = conn.execute("SELECT record_sha256,image_sha256,cache_sha256 FROM observations WHERE id=?", (row["id"],)).fetchone()
    if old:
        if old != (identity, image_sha, cache_sha):
            raise ValueError(f"Changed observation {row['id']}; create a new version explicitly")
        return "replayed"
    if conn.execute("SELECT 1 FROM observations WHERE image_sha256=?", (image_sha,)).fetchone():
        return "duplicate_image"
    conn.execute("INSERT INTO observations VALUES (?,?,?,?,?,?)", (
        row["id"], identity, image_sha, cache_sha,
        json.dumps(row, ensure_ascii=False, sort_keys=True), vector))
    return "added"


def ingest(conn, rows, dataset):
    import torch
    counts = Counter()
    start = time.perf_counter()
    for i, row in enumerate(rows):
        image = contained(dataset, row["image"])
        cache = contained(dataset, f"img_embeds_pre/{int(row['id'])}.pt")
        if not image.is_file() or not cache.is_file():
            counts["missing_pair"] += 1
            continue
        image_sha, cache_sha = digest_file(image), digest_file(cache)
        # Fail on revisions even during replay. Never reload executable pickle objects.
        old = conn.execute("SELECT record_sha256,image_sha256,cache_sha256 FROM observations WHERE id=?", (row["id"],)).fetchone()
        if old:
            if old != (row_digest(row), image_sha, cache_sha):
                raise ValueError(f"Changed observation {row['id']}; explicit versioning required")
            counts["replayed"] += 1
            continue
        feature = torch.load(cache, map_location="cpu", weights_only=True)
        if not isinstance(feature, torch.Tensor) or feature.ndim != 2 or feature.shape[1] != 768:
            raise ValueError(f"Unexpected legacy feature shape for post {row['id']}")
        gpu_feature = feature.to(device="cuda", dtype=torch.float32)
        pooled = gpu_feature.mean(dim=0)
        if not bool(torch.isfinite(pooled).all()) or float(pooled.norm()) == 0:
            raise ValueError(f"Invalid visual feature for post {row['id']}")
        vector = torch.nn.functional.normalize(pooled, dim=0).cpu().numpy().tobytes()
        counts[record_observation(conn, row, image_sha, cache_sha, vector)] += 1
        del feature, gpu_feature, pooled
        if (i + 1) % 256 == 0:
            print(json.dumps({"ingested": i + 1, "counts": dict(counts)}, ensure_ascii=False), flush=True)
    conn.commit()
    return {"counts": dict(counts), "seconds": time.perf_counter() - start}


def observation_tags(row):
    return sorted({canonical(t) for t in row.get("flat_tags", []) + row.get("identity_tags", []) if t.strip()})


def build_snapshot(conn, output, core, manifest, max_vocab=3072, min_docs=2):
    import numpy as np
    import torch
    from experiments.vrs_cuda import AssociationCUDA
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = conn.execute("SELECT id,record,image_sha256,cache_sha256,vector FROM observations ORDER BY id").fetchall()
    if not records or len(records) > 16384:
        raise ValueError("This bounded trial requires 1..16384 distinct observations")
    rows = [json.loads(r[1]) for r in records]
    docs = [observation_tags(r) for r in rows]
    counts = Counter(tag for doc in docs for tag in doc)
    vocab = [tag for tag, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if count >= min_docs][:max_vocab]
    if not vocab:
        raise ValueError("No tags meet the trial vocabulary threshold")
    vocab_index = {tag: i for i, tag in enumerate(vocab)}
    membership = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, tags in enumerate(docs):
        for tag in tags:
            if tag in vocab_index:
                membership[i, vocab_index[tag]] = 1
    x = torch.from_numpy(membership).to("cuda")
    # Binary inputs and <2**24 summands: float32 produces exact integer counts.
    torch.backends.cuda.matmul.allow_tf32 = False
    co = x.T @ x
    flat = co.flatten().to(torch.int64)
    zeros = torch.zeros_like(flat)
    with AssociationCUDA(core) as judge:
        status, reason = judge.judge(flat, zeros)
        core_receipt = {"headers_sha256": judge.headers_sha256, "ptx_sha256": judge.ptx_sha256}
        status = status.reshape(co.shape).cpu().numpy()
        reason = reason.reshape(co.shape).cpu().numpy()
    # Co-presence asserts only that some observed records contain both labels.
    # Missing labels are never evidence that an image lacks the visual property.
    co_cpu = co.to(torch.int32).cpu().numpy()
    np.fill_diagonal(status, 0)
    np.fill_diagonal(reason, 1)
    frequencies = np.diag(co_cpu).copy()
    vectors = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in records])
    np.savez(output / "memory.npz", cooccurrence=co_cpu, frequency=frequencies,
             status=status, reason=reason, membership=membership.astype(np.uint8),
             vectors=vectors, vocabulary=np.array(vocab), ids=np.array([r[0] for r in records]))
    with (output / "observations.jsonl").open("w", encoding="utf-8") as f:
        for raw, row in zip(records, rows):
            f.write(json.dumps({"id": raw[0], "image": row["image"], "tags": observation_tags(row),
                                "image_sha256": raw[2], "cache_sha256": raw[3],
                                "identity_tags": row.get("identity_tags", [])}, ensure_ascii=False) + "\n")
    result = {"schema": "rs-vrs-trial-v1", "documents": len(rows), "vocabulary": len(vocab),
              "directed_observed_pairs": int(np.count_nonzero(status == 1)),
              "source_manifest": str(Path(manifest).resolve()), "source_sha256": digest_file(manifest),
              "core": core_receipt, "gpu": torch.cuda.get_device_name(),
              "selection": "general rating, official train split, SHA256 ID ordering; exact-image deduplication",
              "visual_features": "mean of legacy 768-dimensional tokens; encoder provenance unverified; NOT text-aligned",
              "claim": "recorded co-presence only; A is not semantic truth or permission to add a tag",
              "absent_tag": "unknown visual meaning; not a refutation",
              "max_vocab": max_vocab, "min_docs": min_docs, "seconds": time.perf_counter() - start}
    result["files_sha256"] = {name: digest_file(output / name) for name in ("memory.npz", "observations.jsonl")}
    (output / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del x, co, flat, zeros
    return result


class Memory:
    def __init__(self, directory):
        import numpy as np
        directory = Path(directory)
        self.receipt = json.loads((directory / "receipt.json").read_text())
        for name in ("memory.npz", "observations.jsonl"):
            if digest_file(directory / name) != self.receipt["files_sha256"][name]:
                raise ValueError(f"Snapshot digest mismatch: {name}")
        with np.load(directory / "memory.npz", allow_pickle=False) as data:
            self.data = {name: data[name] for name in data.files}
        self.vocab = self.data["vocabulary"].tolist()
        self.index = {tag: i for i, tag in enumerate(self.vocab)}
        self.rows = [json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()]

    def query(self, tags, excluded=(), limit=8):
        import torch
        limit = max(0, min(int(limit), 50))
        seeds = sorted({canonical(t) for t in tags if t.strip()})
        bans = {canonical(t) for t in excluded if t.strip()}
        if set(seeds) & bans:
            raise ValueError("A tag cannot be both required and excluded")
        known = [self.index[t] for t in seeds if t in self.index]
        unknown = [t for t in seeds if t not in self.index]
        base = {"input_tags": seeds, "excluded": sorted(bans), "unknown_tags": unknown,
                "verdict": "U", "candidates": [], "examples": [], "matching_records": 0,
                "exclusion_scope": "records with explicit excluded labels are removed; missing labels do not prove visual absence",
                "scope": "tag membership evidence; suggestions require user/creativity permission"}
        if not known or unknown:
            return base
        x = torch.from_numpy(self.data["membership"]).to("cuda")
        match = x[:, known].bool().all(dim=1)
        # Excludes records explicitly containing a banned label. Its absence does
        # not prove the property is visually absent in the image.
        banned_indices = [self.index[t] for t in bans if t in self.index]
        if bans:
            # Rare excluded labels may be outside the truncated vocabulary;
            # still honor their presence in the complete observation record.
            match &= torch.tensor([not bans.intersection(r["tags"]) for r in self.rows], device="cuda")
        witnesses = torch.nonzero(match).flatten()
        base["matching_records"] = witnesses.numel()
        if not witnesses.numel():
            return base
        support = x[match].sum(dim=0).to(torch.float32)
        # The saved VRS judgments gate observed pairs. This does not promote
        # an observational association into a semantic/causal judgment.
        judgments = torch.from_numpy(self.data["status"][known]).to("cuda")
        support[~(judgments == 1).all(dim=0)] = 0
        support[known + banned_indices] = 0
        # Rank observed candidates using conditional rate; neither probability of
        # visual truth nor a learned causal relation is claimed.
        top = torch.argsort(support, descending=True, stable=True).cpu().tolist()
        counts = support.cpu().tolist()
        ids = witnesses.cpu().tolist()
        for j in top:
            if len(base["candidates"]) >= limit or counts[j] < 2:
                break
            evidence_ids = [self.rows[i]["id"] for i in ids if self.data["membership"][i, j]][:3]
            base["candidates"].append({"tag": self.vocab[j], "support": int(counts[j]),
                                        "conditional_rate": counts[j] / len(ids), "image_ids": evidence_ids})
        base["examples"] = [{"id": self.rows[i]["id"], "image": self.rows[i]["image"]} for i in ids[:3]]
        base["verdict"] = "A: recorded co-presence" if base["candidates"] else "U"
        return base

    def visual_neighbors(self, image_id, limit=3):
        import torch
        ids = self.data["ids"].tolist()
        if image_id not in ids:
            raise ValueError("Image ID is absent from this snapshot")
        vectors = torch.from_numpy(self.data["vectors"]).to("cuda")
        scores = vectors @ vectors[ids.index(image_id)]
        scores[ids.index(image_id)] = -2
        scores, indices = torch.topk(scores, min(limit, len(ids) - 1))
        return [{"id": ids[i], "cosine": score, "tags": self.rows[i]["tags"]}
                for score, i in zip(scores.cpu().tolist(), indices.cpu().tolist())]

    def evaluate(self, held_out, fixed_vocabulary=None):
        """Fixed masking; compare association ranking against frequency only.

        Tags are incomplete labels. These metrics measure recovery of recorded
        labels, not human-rated relevance, Korean understanding, or image quality.
        """
        import torch
        selected = []
        train_ids = set(self.data["ids"].tolist())
        for row in held_out:
            if row["id"] in train_ids:
                raise ValueError("Evaluation ID leaked into training")
            tags = sorted(observation_tags(row), key=lambda t: hashlib.sha256(f"{row['id']}:{t}".encode()).digest())
            known = [t for t in tags if t in self.index and (fixed_vocabulary is None or t in fixed_vocabulary)]
            if len(known) >= 8:
                selected.append((row["id"], known[:3], set(known[3:]), len(tags) - len(known), len(tags)))
        if not selected:
            raise ValueError("No evaluable held-out rows")
        co = torch.from_numpy(self.data["cooccurrence"]).to(device="cuda", dtype=torch.float32)
        membership = torch.from_numpy(self.data["membership"]).to("cuda")
        freq = torch.from_numpy(self.data["frequency"]).to(device="cuda", dtype=torch.float32)
        eligible = torch.tensor([fixed_vocabulary is None or tag in fixed_vocabulary for tag in self.vocab], device="cuda")
        results = []
        for post_id, seeds, target, oov, total in selected:
            indices = [self.index[t] for t in seeds]
            association = (co[indices] / freq[indices, None]).mean(dim=0)
            strict = membership[membership[:, indices].bool().all(dim=1)].sum(dim=0).to(torch.float32)
            strict[strict < 2] = -1
            baseline = freq.clone()
            association[indices] = -1
            strict[indices] = -1
            baseline[indices] = -1
            association[~eligible] = -1
            strict[~eligible] = -1
            baseline[~eligible] = -1
            methods = {}
            for name, scores in (("frequency", baseline), ("association", association), ("strict_query", strict)):
                ix = torch.argsort(scores, descending=True, stable=True)[:10].cpu().tolist()
                if name == "strict_query":
                    ix = [i for i in ix if float(scores[i]) >= 2]
                predictions = [self.vocab[i] for i in ix]
                hits = len(set(predictions) & target)
                methods[name] = {"precision_at_10": hits / 10, "recall_at_10": hits / len(target), "tags": predictions}
            results.append({"id": post_id, "seeds": seeds, "target_count": len(target), "oov": oov, "total": total, **methods})
        return {"scope": "official general-rating test IDs; three recorded seed tags; in-vocabulary masked-label recovery, not prompt quality; near-duplicate leakage not audited",
                "methods": {"association": "mean pairwise conditional rates; exploratory scoring, permits partial-seed matches",
                            "strict_query": "all seed tags required, minimum two witnesses per suggestion; unknown/no support abstains; empty slots count as misses"},
                "cases": len(results), "oov_fraction": sum(r["oov"] for r in results) / sum(r["total"] for r in results),
                "strict_query_coverage": sum(bool(r["strict_query"]["tags"]) for r in results) / len(results),
                "means": {m: {metric: sum(r[m][metric] for r in results) / len(results)
                               for metric in ("precision_at_10", "recall_at_10")}
                          for m in ("frequency", "association", "strict_query")}, "results": results}
