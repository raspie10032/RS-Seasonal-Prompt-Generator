"""Independent CUDA neighbor retrieval. No corpus image access or LLM loading."""
from collections import defaultdict
import math

from .store import Store, canonical, digest
from .vectors import decode_vector
from .generation_metadata import VERSION as TAG_VIEW_VERSION, recommendation_view
from .tag_policy import RECOMMENDATION_POLICY_VERSION, filter_recommendation_tags


class Recommender:
    def __init__(self, store, snapshot):
        self.store = Store(store)
        self.snapshot = snapshot
        self.manifest = self.store.manifest(snapshot)
        self._visual_entries = None

    def recommend(self, vector, image_sha256, neighbors=32, limit=20, ratings=(), batch_size=2048,
                  dictionary=None):
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        if not 1 <= neighbors <= 256 or not 1 <= limit <= 100 or not 1 <= batch_size <= 4096:
            raise ValueError("Invalid retrieval limits")
        if any(r not in ("g", "s", "q", "e", "u") for r in ratings):
            raise ValueError("Invalid ratings")
        query = torch.as_tensor(vector, device="cuda", dtype=torch.float32).flatten()
        if query.numel() != self.manifest["contract"]["dimension"] or not torch.isfinite(query).all() or query.norm() <= 0:
            raise ValueError("Invalid query vector")
        query = query / query.norm()
        best, pending, seen_images = [], [], set()

        def search_batch(entries):
            features = np.stack([decode_vector(e["vector"], query.numel()) for e in entries])
            tensor = torch.from_numpy(features).to("cuda", torch.float32)
            tensor = torch.nn.functional.normalize(tensor, dim=1)
            similarities = (tensor @ query).cpu().tolist()
            # Stable key tie break, including across bucket boundaries.
            best.extend((s, e["key"], e) for s, e in zip(similarities, entries))
            best.sort(key=lambda item: (-item[0], item[1]))
            del best[neighbors:]

        # Only a bounded vector block resides on GPU. This is exact streaming
        # search, not a sublinear ANN index; million-scale latency is unproven.
        if self._visual_entries is None:
            self._visual_entries = []
            for entry in self.store.entries(self.snapshot):
                if entry["vector"] is not None and entry["document"]["split"] == "train":
                    self._visual_entries.append(dict(recommendation_view(entry), audit=[]))
        for entry in self._visual_entries:
            doc = entry["document"]
            if entry["vector"] is None or doc["split"] != "train" or doc["image_sha256"] == image_sha256:
                continue
            if ratings and doc["rating"] not in ratings:
                continue
            if doc["image_sha256"] in seen_images:
                continue
            seen_images.add(doc["image_sha256"])
            # Audit metadata is preserved in storage, not needed in the vote.
            pending.append(entry)
            if len(pending) >= batch_size:
                search_batch(pending)
                pending.clear()
        if pending:
            search_batch(pending)
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
    result = memory.recommend(vector, image_before, **options)
    result["query_image_sha256"] = image_before
    return result
