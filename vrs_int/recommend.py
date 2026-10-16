"""Independent CUDA neighbor retrieval. No corpus image access or LLM loading."""
from collections import defaultdict
import math
import threading

from .store import Store, canonical, digest
from .generation_metadata import VERSION as TAG_VIEW_VERSION
from .tag_policy import RECOMMENDATION_POLICY_VERSION, filter_recommendation_tags


class Recommender:
    def __init__(self, store, snapshot):
        self.store = Store(store)
        self.snapshot = snapshot
        self.manifest = self.store.manifest(snapshot)
        self._index = None
        self._lock = threading.RLock()

    def recommend(self, vector, image_sha256, neighbors=32, limit=20, ratings=(), batch_size=2048,
                  dictionary=None, device=None, pixel_sha256=None, interrupt=None):
        # Serialize cache preparation/close with queries, including cancellation.
        with self._lock:
            return self._recommend(vector, image_sha256, neighbors, limit, ratings, batch_size,
                                   dictionary, device, pixel_sha256, interrupt)

    def close(self):
        """Release maps/lease; retain the rebuildable disk cache for the next open."""
        with self._lock:
            if self._index is not None:
                self._index.close()
                self._index = None

    def _recommend(self, vector, image_sha256, neighbors, limit, ratings, batch_size,
                   dictionary, device, pixel_sha256, interrupt):
        import torch
        from .search_index import SearchIndex, best_in_block
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        if not 1 <= neighbors <= 256 or not 1 <= limit <= 100 or not 1 <= batch_size <= 4096:
            raise ValueError("Invalid retrieval limits")
        if any(r not in ("g", "s", "q", "e", "u") for r in ratings):
            raise ValueError("Invalid ratings")
        device = torch.device(device if device is not None else "cuda")
        if device.type != "cuda":
            raise RuntimeError("CUDA search required; no CPU fallback")
        query = torch.as_tensor(vector, device=device, dtype=torch.float32).flatten()
        if query.numel() != self.manifest["contract"]["dimension"] or not torch.isfinite(query).all() or query.norm() <= 0:
            raise ValueError("Invalid query vector")
        query = query / query.norm()
        if interrupt is not None:
            interrupt()
        if self._index is None:
            # Assign only after every bucket is verified and preparation finishes.
            self._index = SearchIndex(self.store, self.snapshot, interrupt)
        index = self._index
        best = []
        try:
            rows = index.eligible(image_sha256, pixel_sha256, ratings)
            for start in range(0, len(rows), batch_size):
                if interrupt is not None:
                    interrupt()
                selected = rows[start:start + batch_size]
                # Advanced indexing makes a bounded, writable copy. Neither the
                # full matrix nor a persistent model/cache occupies GPU memory.
                features = index.vectors[selected]
                tensor = torch.from_numpy(features).to(device, torch.float32)
                tensor = torch.nn.functional.normalize(tensor, dim=1)
                scores = (tensor @ query).cpu().numpy()
                keys = index.meta["order"][selected]
                local = best_in_block(scores, keys, neighbors)
                best.extend((float(scores[i]), int(keys[i]), int(selected[i])) for i in local)
                best.sort(key=lambda item: (-item[0], item[1]))
                del best[neighbors:]
            if interrupt is not None:
                interrupt()
            documents = index.documents([row for _, _, row in best]) if best else []
            best = [(score, doc["key"], doc) for (score, _, _), doc in zip(best, documents)]
        finally:
            index.release_pages()
        if not best:
            return {"snapshot": self.snapshot, "visual_verdict": "U", "candidates": [], "identities": {}, "neighbors": [],
                    "recommendation_policy": RECOMMENDATION_POLICY_VERSION, "excluded_tags": {}}
        weights = [math.exp((s - best[0][0]) / .07) for s, _, _ in best]
        total = sum(weights)
        votes, evidence, categories = defaultdict(float), defaultdict(list), defaultdict(set)
        for (_, key, entry), weight in zip(best, weights):
            for tag in entry["document"]["tags"]:
                votes[tag] += weight / total
                evidence[tag].append(key)
                categories[tag].update(entry["categories"].get(tag, ["unknown"]))

        annotations = {}
        if dictionary is not None:
            for tag in votes:
                annotations[tag] = dictionary.annotate({"tag": tag, "categories": sorted(categories[tag])})
                categories[tag] = set(annotations[tag]["categories"])

        def candidates(tags):
            return [{**annotations.get(tag, {}), "tag": tag, "score": votes[tag], "support": len(evidence[tag]),
                     "observation_keys": evidence[tag], "image_ids": [
                         int(k.split(":", 1)[1]) if k.startswith("danbooru:") else k.split(":", 1)[1]
                         for k in evidence[tag]],
                     "categories": sorted(categories[tag]), "visual_verdict": "U"} for tag in tags]

        ordered = sorted(votes, key=lambda t: (-votes[t], t))
        ordered, excluded = filter_recommendation_tags(ordered, categories)
        result = {"snapshot": self.snapshot, "contract_sha256": digest(canonical(self.manifest["contract"])),
                "generated_tag_view": TAG_VIEW_VERSION,
                "recommendation_policy": RECOMMENDATION_POLICY_VERSION, "excluded_tags": excluded,
                "visual_verdict": "U", "candidates": candidates(ordered[:limit]),
                "identities": {kind: candidates([t for t in ordered if kind in categories[t]][:3])
                               for kind in ("artist", "character", "copyright")},
                "neighbors": [{"key": key, "id": e["document"]["id"], "similarity": score,
                               "rating": e["document"]["rating"], "image_sha256": e["document"]["image_sha256"]}
                              for score, key, e in best],
                "scope": "Creative reference tags. Scores are neighbor votes, not identity probabilities. "
                         "No native VRS judgment core is invoked by this standalone storage/retrieval path."}
        if dictionary is not None:
            result["tag_dictionary"] = dict(dictionary.provenance)
        return result


def image_recommend(store, snapshot, encoder_path, image, **options):
    from pathlib import Path
    from .encoder import ImageEncoder
    from .import_snapshot import file_digest
    memory = Recommender(store, snapshot)
    expected = memory.manifest["contract"]["encoder"]
    supported = {"provenance_sha256": file_digest(Path(encoder_path) / "provenance.json"),
                 "preprocess": "gemma4-float-pixels-v1", "backend": "pil", "quantization": "nf4",
                 "pooling": "mean-tokens-l2-v1"}
    if expected != supported or memory.manifest["contract"]["dimension"] != 768:
        raise ValueError("Query encoder incompatible with this snapshot")
    image_before = file_digest(image)
    encoder = ImageEncoder(encoder_path)
    vector = encoder.encode(image)
    # Return only the query vector to the caller's search phase.
    import gc
    import torch
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    if file_digest(image) != image_before:
        raise ValueError("Query image changed during encoding")
    try:
        result = memory.recommend(vector, image_before, **options)
    finally:
        memory.close()
    result["query_image_sha256"] = image_before
    return result
