"""Matched 2048-image diagnostic: legacy uint8 versus corrected NF4 vision.

Only the pixel casting behavior changes between methods. Neither the full VRS
snapshot nor the old image cache is modified. The same 128 held-out images are
used for a diagnostic comparison, not an independent confirmation after tuning.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from experiments.run_image_trial import score, select_tests, summarize
from experiments.run_vrs_trial import Monitor, setup_gpu
from experiments.vrs_full import write_json
from experiments.vrs_memory import contained, digest_file


def run(args):
    import numpy as np
    from experiments.vrs_cuda import AssociationCUDA
    from experiments.vrs_image import ImageEncoder, ImageMemory
    torch = setup_gpu()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.memory / "documents.jsonl").open()]
    tests = select_tests(rows, 32)
    train_hashes = {r["image_sha256"] for r in rows if r["split"] == "train"}
    assert not any(r["image_sha256"] in train_hashes for r in tests)
    del rows
    with Monitor() as monitor:
        memory = ImageMemory(args.memory)
        positions = [i for rating in "gsqe" for i in sorted(
            [i for i, row in enumerate(memory.rows) if row["rating"] == rating],
            key=lambda i: hashlib.sha256(f"rs-vrs-image-repair-v1:{memory.rows[i]['id']}".encode()).digest())[:512]]
        memory.rows = [memory.rows[i] for i in positions]
        memory.x = memory.x[positions].tocsr()
        memory.vectors = memory.vectors[positions].contiguous()
        memory.frequency = np.asarray(memory.x.sum(axis=0)).ravel()
        memory.baseline = np.argsort(-memory.frequency, kind="stable")[:20].tolist()
        torch.cuda.empty_cache()
        write_json(output / "train_selection.json", memory.rows)
        write_json(output / "test_selection.json", [{k: row[k] for k in ("id", "image", "rating", "image_sha256")} for row in tests])
        baseline = [memory.vocab[i] for i in memory.baseline]
        vocabulary = {tag: i for i, tag in enumerate(memory.vocab)}
        summaries, timing = {}, {}
        for method in ("legacy_uint8", "corrected_float"):
            encoder = ImageEncoder(args.encoder, legacy_uint8=(method == "legacy_uint8"))
            if method == "corrected_float":
                new_vectors = []
                for n, row in enumerate(memory.rows):
                    path = contained(args.dataset, row["image"])
                    if digest_file(path) != row["image_sha256"]:
                        raise ValueError("Training image bytes changed")
                    new_vectors.append(encoder.encode(path).cpu())
                    if (n + 1) % 128 == 0:
                        print(json.dumps({"stage": "reencode", "done": n + 1, "total": len(memory.rows)}), flush=True)
                memory.vectors = torch.stack(new_vectors).to("cuda")
                np.save(output / "corrected_vectors.npy", memory.vectors.cpu().numpy())
            results = []
            with AssociationCUDA(memory.receipt["core_root"]) as judge, (output / f"{method}.jsonl").open("w") as log:
                for n, row in enumerate(tests):
                    before = time.perf_counter()
                    path = contained(args.dataset, row["image"])
                    image_hash = digest_file(path)
                    if image_hash != row["image_sha256"]:
                        raise ValueError("Test image bytes changed")
                    recommendation = memory.recommend(encoder.encode(path), image_hash, judge=judge)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - before
                    assert not any(r["image_sha256"] == image_hash for r in recommendation["neighbors"])
                    target = row["tags"]
                    predicted = [r["tag"] for r in recommendation["candidates"]]
                    result = {"id": row["id"], "image_path": str(path), "rating": row["rating"],
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
                            "any_train_known": any(memory.frequency[vocabulary[t]] > 0 for t in expected),
                            "hit_at_1": bool(set(names[:1]) & set(expected)),
                            "hit_at_3": bool(set(names[:3]) & set(expected))}
                    log.write(json.dumps(result, ensure_ascii=False) + "\n")
                    log.flush()
                    results.append(result)
                    if (n + 1) % 32 == 0:
                        print(json.dumps({"stage": method, "done": n + 1, "total": len(tests)}), flush=True)
            summaries[method] = summarize(results)
            timing[method] = {"median": float(np.median([r["seconds"] for r in results])),
                              "p95": float(np.percentile([r["seconds"] for r in results], 95))}
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
        report = {"state": "complete", "index_count": len(memory.rows), "test_count": len(tests),
                  "memory_receipt_sha256": digest_file(args.memory / "receipt.json"),
                  "encoder_provenance_sha256": digest_file(args.encoder / "provenance.json"),
                  "methods": summaries, "timing_seconds": timing,
                  "total_seconds": time.perf_counter() - started,
                  "resources": {"gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_threads": torch.get_num_threads(),
                                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                                "process_gpu_peak_sampled_mib": max((r["mib"] for r in monitor.samples), default=None),
                                "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
                  "scope": "Matched diagnostic; no test tags or ratings are search inputs; same NF4 encoder and processor",
                  "limits": ["2048 balanced train observations, not the full index",
                             "Same 128 holdouts as the first test; not independent confirmation",
                             "Near duplicates and foundation pretraining overlap uncontrolled",
                             "Ground truth and identity category annotations incomplete",
                             "VRS A applies to recorded neighbor tags, all query visual verdicts remain U"],
                  "code_sha256": {str(p): digest_file(p) for p in
                                  (Path(__file__), Path(__file__).with_name("vrs_image.py"),
                                   Path(__file__).with_name("run_image_trial.py"))}}
        write_json(output / "gpu_samples.json", monitor.samples)
        report["output_sha256"] = {p.name: digest_file(p) for p in output.iterdir() if p.is_file()}
        write_json(output / "report.json", report)
        print(json.dumps({"stage": "complete", "report": str(output / "report.json"),
                          "resources": report["resources"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("memory", "dataset", "encoder", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
