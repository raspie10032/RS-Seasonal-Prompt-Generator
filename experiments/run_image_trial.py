"""Held-out pixels -> VRS neighbor tags. Fixed selection before predictions.

The index contains train observations only, with exact byte duplicates excluded.
Legacy-cache compatibility is tested on train images before any test scoring.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from experiments.run_vrs_trial import Monitor, peak_rss_mib, setup_gpu
from experiments.vrs_full import write_json
from experiments.vrs_memory import contained, digest_file


def selection_key(row):
    return hashlib.sha256(f"rs-vrs-image-holdout-v1:{row['id']}".encode()).digest()


def select_tests(rows, per_rating):
    return [row for rating in "gsqe" for row in sorted(
        (r for r in rows if r["split"] == "test" and r["rating"] == rating),
        key=selection_key)[:per_rating]]


def score(predicted, target, k):
    correct = len(set(predicted[:k]) & set(target))
    return {f"p_at_{k}": correct / k, f"r_at_{k}": correct / len(target) if target else 0}


def summarize(rows):
    def group(items):
        if not items:
            return {"count": 0}
        result = {"count": len(items)}
        for method in ("image", "frequency"):
            result[method] = {key: sum(r[method][key] for r in items) / len(items)
                              for key in items[0][method]}
        result["identities"] = {}
        for kind in ("artist", "character", "copyright"):
            known = [r["identity_metrics"][kind] for r in items if r["identity_metrics"][kind]["targets"]]
            result["identities"][kind] = {
                "with_recorded_target": len(known),
                "with_train_known_target": sum(r["any_train_known"] for r in known),
                "hit_at_1": sum(r["hit_at_1"] for r in known) / len(known) if known else None,
                "hit_at_3": sum(r["hit_at_3"] for r in known) / len(known) if known else None,
                "without_recorded_target": len(items) - len(known),
            }
        return result
    return {"balanced_aggregate": group(rows),
            "by_rating": {rating: group([r for r in rows if r["rating"] == rating]) for rating in "gsqe"}}


def run(args):
    from experiments.vrs_image import ImageEncoder, ImageMemory
    from experiments.vrs_cuda import AssociationCUDA
    torch = setup_gpu()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = [json.loads(line) for line in (args.memory / "documents.jsonl").open()]
    selected = select_tests(rows, args.per_rating)
    if not selected:
        raise ValueError("Empty test set")
    train_hashes = {r["image_sha256"] for r in rows if r["split"] == "train"}
    if any(r["image_sha256"] in train_hashes for r in selected):
        raise ValueError("Test image byte overlap with train")
    write_json(output / "selection.json", [{k: row[k] for k in ("id", "image", "rating", "image_sha256")}
                                            for row in selected])
    results = []
    with Monitor() as monitor:
        encoder = ImageEncoder(args.encoder, backend=args.backend, legacy_uint8=args.legacy_uint8)
        memory = ImageMemory(args.memory)
        lookup = {r["id"]: i for i, r in enumerate(memory.rows)}
        calibration = sorted(memory.rows, key=selection_key)[:8]
        checks = []
        for row in calibration:
            q = encoder.encode(contained(args.dataset, row["image"]))
            similarities = memory.vectors @ q
            checks.append({"id": row["id"], "same_image_cosine": float(similarities[lookup[row['id']]]),
                           "top1_id": memory.rows[int(similarities.argmax())]["id"]})
        compatible = all(c["same_image_cosine"] >= 0.98 and c["top1_id"] == c["id"] for c in checks)
        write_json(output / "compatibility.json", {"passed": compatible, "samples": checks,
                   "criterion": "8 predetermined train images: same-image cosine >= .98 and rank 1"})
        if not compatible:
            raise RuntimeError("Legacy image cache is incompatible; no test scoring performed")
        print(json.dumps({"stage": "compatibility_passed", "samples": checks}), flush=True)
        baseline = [memory.vocab[i] for i in memory.baseline]
        index = {tag: i for i, tag in enumerate(memory.vocab)}
        with AssociationCUDA(memory.receipt["core_root"]) as judge, (output / "results.jsonl").open("w") as log:
            for n, row in enumerate(selected):
                before = time.perf_counter()
                image_path = contained(args.dataset, row["image"])
                image_hash = digest_file(image_path)
                if image_hash != row["image_sha256"]:
                    raise ValueError("Test image changed since snapshot")
                vector = encoder.encode(image_path)
                recommendation = memory.recommend(vector, image_hash, judge=judge)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                if any(r["image_sha256"] == image_hash for r in recommendation["neighbors"]):
                    raise ValueError("Exact duplicate reached recommendation")
                predicted = [r["tag"] for r in recommendation["candidates"]]
                target = row["tags"]  # Evaluation only, AFTER recommend() has returned.
                result = {"id": row["id"], "image_path": str(image_path), "rating": row["rating"],
                          "image_sha256": image_hash, "seconds": elapsed,
                          "recommendation": recommendation, "reference_tags": target,
                          "image": {**score(predicted, target, 10), **score(predicted, target, 20)},
                          "frequency": {**score(baseline, target, 10), **score(baseline, target, 20)},
                          "identity_metrics": {}}
                for kind in ("artist", "character", "copyright"):
                    expected = [tag for tag in target if kind in memory.categories.get(tag, [])]
                    names = [r["tag"] for r in recommendation["identities"][kind]]
                    result["identity_metrics"][kind] = {
                        "targets": expected, "predictions": names,
                        "any_train_known": any(memory.frequency[index[t]] > 0 for t in expected),
                        "hit_at_1": bool(set(names[:1]) & set(expected)),
                        "hit_at_3": bool(set(names[:3]) & set(expected)),
                    }
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()
                results.append(result)
                if (n + 1) % 16 == 0:
                    print(json.dumps({"stage": "image_test", "done": n + 1, "total": len(selected),
                                      "mean_p10": sum(r["image"]["p_at_10"] for r in results) / len(results)}), flush=True)
        import numpy as np
        report = {"state": "complete", "memory": str(args.memory.resolve()),
                  "memory_receipt_sha256": digest_file(args.memory / "receipt.json"),
                  "encoder": encoder.provenance, "processor_backend": args.backend,
                  "legacy_uint8_reproduction": args.legacy_uint8,
                  "quantization": encoder.quantization, "compatibility": checks,
                  "index_count": len(memory.rows), "index_splits": dict(Counter(r["split"] for r in memory.rows)),
                  "search_ratings": "gsqe", "test_input": "Image pixels only; no tags, identity, rating, prompt or filename text is encoded",
                  "summary": summarize(results), "frequency_baseline": baseline,
                  "timing_seconds": {"median_per_image": float(np.median([r["seconds"] for r in results])),
                                     "p95_per_image": float(np.percentile([r["seconds"] for r in results], 95)),
                                     "total_including_load_and_checks": time.perf_counter() - started},
                  "resources": {"gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_threads": torch.get_num_threads(),
                                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                                "process_gpu_peak_sampled_mib": max((r["mib"] for r in monitor.samples), default=None),
                                "host_peak_rss_mib": peak_rss_mib()},
                  "versions": {name: importlib.metadata.version(name) for name in
                               ("torch", "transformers", "bitsandbytes", "accelerate", "Pillow")},
                  "code_sha256": {str(p): digest_file(p) for p in
                                  (Path(__file__), Path(__file__).with_name("vrs_image.py"))},
                  "limits": ["Legacy uint8 reproduction is diagnostic only: old cache discarded pixel precision",
                             "Unseen to the VRS train index; foundation model pretraining overlap unknown",
                             "Exact byte dedup only; similar crops, recompressions and near duplicates are not removed",
                             "Reference tags are incomplete annotations, not exhaustive visual truth",
                             "Identity category hints are incomplete for legacy records",
                             "Balanced 32 per rating sample is not corpus-weighted performance",
                             "Candidate scores are uncalibrated; unknown-character rejection is not established",
                             "No new neural training or ComfyUI integration"]}
        write_json(output / "gpu_samples.json", monitor.samples)
        report["output_sha256"] = {p.name: digest_file(p) for p in output.iterdir() if p.is_file()}
        write_json(output / "report.json", report)
        print(json.dumps({"stage": "complete", "report": str(output / "report.json"),
                          "summary": report["summary"], "resources": report["resources"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-rating", type=int, default=32)
    parser.add_argument("--backend", choices=("pil", "torchvision"), default="pil")
    parser.add_argument("--legacy-uint8", action="store_true", help="Diagnostic reproduction of the defective historical cache")
    args = parser.parse_args()
    if args.per_rating < 1:
        parser.error("--per-rating must be positive")
    run(args)
