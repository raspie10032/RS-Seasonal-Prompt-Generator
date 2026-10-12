"""Image-only retrieval over the immutable VRS observation snapshot.

Returns suggestions with neighbor evidence, never a claim that a retrieved
artist/character identity has been visually established. Only train images are
searchable; caller metadata and target tags are not accepted by recommend().
"""
import json
from pathlib import Path

from experiments.vrs_memory import digest_file
from vrs_int.import_snapshot import corrected_index_metadata


def eligible_indices(rows, mask, excluded_sha256=None, ratings=()):
    return [i for i, (row, valid) in enumerate(zip(rows, mask))
            if valid and row["split"] == "train"
            and row["image_sha256"] != excluded_sha256
            and (not ratings or row["rating"] in ratings)]


class ImageEncoder:
    def __init__(self, directory, backend="pil", quantization="nf4", legacy_uint8=False):
        import torch
        from transformers import Gemma4VisionModel
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        self.path = Path(directory)
        self.provenance = json.loads((self.path / "provenance.json").read_text())
        for name, digest in self.provenance["files"].items():
            if digest_file(self.path / name) != digest:
                raise ValueError(f"Encoder digest mismatch: {name}")
        if backend == "pil":
            from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
            processor_type = Gemma4ImageProcessorPil
        elif backend == "torchvision":
            from transformers import Gemma4ImageProcessor
            processor_type = Gemma4ImageProcessor
        else:
            raise ValueError("Unsupported image processor")
        self.processor = processor_type.from_pretrained(self.path, local_files_only=True)
        extra = {}
        if quantization == "nf4":
            from transformers import BitsAndBytesConfig
            extra = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=[]), "device_map": {"": "cuda:0"}}
        elif quantization != "bf16":
            raise ValueError("Unsupported encoder quantization")
        self.model = Gemma4VisionModel.from_pretrained(
            self.path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", **extra)
        if quantization == "bf16":
            self.model.to("cuda:0")
        self.model.eval()
        self.backend = backend
        self.quantization = quantization
        self.legacy_uint8 = legacy_uint8
        if legacy_uint8:
            if quantization != "nf4":
                raise ValueError("Legacy uint8 behavior requires NF4 weights")
            # Reproduce the historical v02 cache ONLY, not a correct vision input.
            # Its old Transformers patch embedder cast rescaled pixels to the
            # quantized weight dtype (uint8), discarding much of their information.
            from types import MethodType

            def legacy_patch(module, pixel_values, pixel_position_ids, padding_positions):
                pixels = 2 * (pixel_values - 0.5)
                projected = module.input_proj(pixels.to(module.input_proj.weight.dtype))
                return projected + module._position_embeddings(pixel_position_ids, padding_positions)

            self.model.patch_embedder.forward = MethodType(legacy_patch, self.model.patch_embedder)

    def encode(self, image):
        import torch
        from PIL import Image
        with Image.open(image) as source:
            pixels = source.convert("RGB")
        inputs = self.processor(images=pixels, return_tensors="pt")
        with torch.inference_mode():
            features = self.model(
                pixel_values=inputs["pixel_values"].to("cuda", torch.bfloat16),
                pixel_position_ids=inputs["image_position_ids"].to("cuda"),
            ).last_hidden_state
            if not torch.isfinite(features).all():
                raise ValueError("Nonfinite image features")
            vector = features.float().reshape(-1, features.shape[-1]).mean(dim=0)
            norm = torch.linalg.vector_norm(vector)
            if norm <= 0:
                raise ValueError("Zero image vector")
            return vector / norm


class ImageMemory:
    def __init__(self, directory, corrected_index=None):
        import numpy as np
        import torch
        from scipy import sparse
        self.path = Path(directory)
        self.receipt = json.loads((self.path / "receipt.json").read_text())
        if self.receipt.get("state") != "complete":
            raise ValueError("Incomplete VRS snapshot")
        for name in ("documents.jsonl", "vocabulary.json", "tag_categories.json",
                     "membership.npz", "vectors.npy", "vector_mask.npy"):
            if digest_file(self.path / name) != self.receipt["files_sha256"][name]:
                raise ValueError(f"Snapshot digest mismatch: {name}")
        self.rows = [json.loads(line) for line in (self.path / "documents.jsonl").open()]
        self.vocab = json.loads((self.path / "vocabulary.json").read_text())
        self.categories = json.loads((self.path / "tag_categories.json").read_text())
        if corrected_index is not None:
            _, selected = corrected_index_metadata(corrected_index, self.path / "receipt.json")
            lookup = {row["id"]: i for i, row in enumerate(self.rows)}
            self.positions = []
            for row in selected:
                pos = lookup.get(row["id"])
                if pos is None or self.rows[pos] != row:
                    raise ValueError("Corrected index observation mismatch")
                self.positions.append(pos)
            vectors = np.load(Path(corrected_index) / "corrected_vectors.npy", allow_pickle=False)
            if vectors.shape != (len(selected), 768) or not np.isfinite(vectors).all():
                raise ValueError("Invalid corrected vectors")
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.01):
                raise ValueError("Corrected vectors must be normalized")
        else:
            mask = np.load(self.path / "vector_mask.npy")
            self.positions = eligible_indices(self.rows, mask)
            vectors = np.load(self.path / "vectors.npy", mmap_mode="r")[self.positions]
        self.rows = [self.rows[i] for i in self.positions]
        self.x = sparse.load_npz(self.path / "membership.npz")[self.positions].tocsr()
        self.frequency = np.asarray(self.x.sum(axis=0)).ravel()
        self.vectors = torch.from_numpy(vectors).to("cuda", torch.float32)
        self.vectors = torch.nn.functional.normalize(self.vectors, dim=1)
        self.baseline = np.argsort(-self.frequency, kind="stable")[:20].tolist()
        self.group_indices = {
            category: torch.tensor([i for i, tag in enumerate(self.vocab)
                                    if category in self.categories.get(tag, [])], device="cuda")
            for category in ("artist", "character", "copyright")}

    def recommend(self, vector, image_sha256, neighbors=32, limit=20, ratings=(), judge=None):
        import torch
        if neighbors < 1 or not 1 <= limit <= 100:
            raise ValueError("Invalid neighbors or limit")
        vector = vector.to("cuda", torch.float32).flatten()
        if vector.numel() != self.vectors.shape[1] or not torch.isfinite(vector).all():
            raise ValueError("Invalid query vector")
        norm = torch.linalg.vector_norm(vector)
        if norm <= 0:
            raise ValueError("Zero query vector")
        similarities = self.vectors @ (vector / norm)
        # Exact duplicates are excluded even for a user-supplied known image.
        eligible = eligible_indices(self.rows, [True] * len(self.rows), image_sha256, ratings)
        allowed = torch.zeros(len(self.rows), device="cuda", dtype=torch.bool)
        allowed[torch.tensor(eligible, device="cuda", dtype=torch.long)] = True
        similarities.masked_fill_(~allowed, -torch.inf)
        k = min(neighbors, len(eligible))
        if not k:
            return {"visual_verdict": "U", "candidates": [], "identities": {}, "neighbors": []}
        values, indices = torch.topk(similarities, k)
        positions = indices.cpu().tolist()
        weights = torch.softmax(values / 0.07, dim=0)
        # Only retrieved observations supply tag votes. Missing tags are not negatives.
        owner, tag_indices = [], []
        for j, pos in enumerate(positions):
            tags = self.x.indices[self.x.indptr[pos]:self.x.indptr[pos + 1]].tolist()
            owner.extend([j] * len(tags))
            tag_indices.extend(tags)
        tags_gpu = torch.tensor(tag_indices, device="cuda", dtype=torch.long)
        owners_gpu = torch.tensor(owner, device="cuda", dtype=torch.long)
        scores = torch.bincount(tags_gpu, weights=weights[owners_gpu], minlength=len(self.vocab))
        support = torch.bincount(tags_gpu, minlength=len(self.vocab))
        ranked = torch.argsort(scores, descending=True, stable=True)[:limit]
        groups = {group: ix[torch.argsort(scores[ix], descending=True, stable=True)[:3]]
                  for group, ix in self.group_indices.items()}
        union = torch.unique(torch.cat([ranked, *groups.values()]))
        states = {}
        if judge is not None:
            status, _ = judge.judge(support[union], torch.zeros_like(support[union]))
            states = dict(zip(union.cpu().tolist(), status.cpu().tolist()))

        def candidates(ids):
            result = []
            for index in ids.cpu().tolist():
                count = int(support[index].item())
                if not count:
                    continue
                tag = self.vocab[index]
                evidence = [self.rows[p]["id"] for p in positions if tag in self.rows[p]["tags"]]
                if len(evidence) != count:
                    raise ValueError("Evidence count mismatch")
                result.append({"tag": tag, "score": float(scores[index].item()), "support": count,
                    "image_ids": evidence, "categories": self.categories.get(tag, ["unknown"]),
                    "record_vrs": "A" if states.get(index) == 1 else "U",
                    "visual_verdict": "U"})
            return result

        return {"visual_verdict": "U", "candidates": candidates(ranked),
                "identities": {group: candidates(ids) for group, ids in groups.items()},
                "neighbors": [{"id": self.rows[p]["id"], "image": self.rows[p]["image"],
                               "rating": self.rows[p]["rating"], "similarity": float(value),
                               "image_sha256": self.rows[p]["image_sha256"]}
                              for p, value in zip(positions, values.cpu().tolist())],
                "scope": "Visual-neighbor suggestions; scores are vote weights, not calibrated probabilities. "
                         "VRS A describes recorded neighbor tags, not visual truth of the query image."}
