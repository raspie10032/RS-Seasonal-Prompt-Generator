"""Run in a network-isolated mount namespace with all old datasets hidden.

Uses only vrs_int/, two new stores, an encoder and one held-out query image.
Measures compression stability, not recognition accuracy or prompt quality.
"""
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time

os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
os.nice(15)

from vrs_int.encoder import ImageEncoder
from vrs_int.import_snapshot import file_digest
from vrs_int.recommend import Recommender
from vrs_int.vectors import decode_vector
import numpy as np
import torch


def run():
    hidden = Path("/mnt/storage/Projects")
    assert hidden.is_dir() and not list(hidden.iterdir()), "Old datasets must be masked"
    output = Path("/tmp/output")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20)
    torch.cuda.reset_peak_memory_stats()
    samples, stop = [], threading.Event()

    def monitor():
        while not stop.is_set():
            text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                "--format=csv,noheader,nounits"], text=True)
            for line in text.splitlines():
                pid, mib, uuid = [v.strip() for v in line.split(",")]
                if int(pid) == os.getpid():
                    samples.append({"seconds": time.perf_counter(), "mib": int(mib), "gpu": uuid})
            stop.wait(.5)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    start = time.perf_counter()
    f32 = Recommender("/tmp/base", os.environ["VRS_INT_BASE"])
    i8 = Recommender("/tmp/compact", os.environ["VRS_INT_COMPACT"])
    assert f32.manifest["contract"]["encoder"]["provenance_sha256"] == file_digest("/tmp/encoder/provenance.json")
    encoder = ImageEncoder("/tmp/encoder")
    query = encoder.encode("/tmp/query.png").cpu().numpy()
    np.save(output / "query_vector.npy", query, allow_pickle=False)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    query_hash = file_digest("/tmp/query.png")
    cold = time.perf_counter()
    first = f32.recommend(query, query_hash)
    cold_seconds = time.perf_counter() - cold
    assert first["neighbors"] and all(r["image_sha256"] != query_hash for r in first["neighbors"])
    trials = [("held-out-image", query, query_hash)]
    counts = Counter()
    for entry in f32._visual_entries:
        rating = entry["document"]["rating"]
        if counts[rating] < 8:
            trials.append((entry["key"], decode_vector(entry["vector"], 768), entry["document"]["image_sha256"]))
            counts[rating] += 1
    ordered = sorted(f32._visual_entries, key=lambda e: e["key"])
    matrix = torch.from_numpy(np.stack([decode_vector(e["vector"], 768) for e in ordered])).cuda()
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    results = []
    started_probes = time.perf_counter()
    for label, vector, image_hash in trials:
        left = first if label == "held-out-image" else f32.recommend(vector, image_hash)
        right = i8.recommend(vector, image_hash)
        gpu_query = torch.as_tensor(vector, dtype=torch.float32, device="cuda")
        scores = (matrix @ torch.nn.functional.normalize(gpu_query, dim=0)).cpu().tolist()
        exact = sorted([(s, e["key"]) for s, e in zip(scores, ordered)
                        if e["document"]["image_sha256"] != image_hash], key=lambda x: (-x[0], x[1]))[:32]
        observed = [r["key"] for r in left["neighbors"]]
        assert observed == [k for _, k in exact], "FP32 streaming search changed exact neighbors"
        expected_tags = {r["tag"] for r in left["candidates"]}
        compact_tags = {r["tag"] for r in right["candidates"]}
        expected_neighbors = set(observed)
        compact_neighbors = {r["key"] for r in right["neighbors"]}
        results.append({"query": label, "neighbors_at_32_overlap": len(expected_neighbors & compact_neighbors) / len(expected_neighbors),
                        "tags_at_20_overlap": len(expected_tags & compact_tags) / len(expected_tags),
                        "identity_at_3_overlap": {kind: len({r["tag"] for r in left["identities"][kind]} &
                            {r["tag"] for r in right["identities"][kind]}) / max(1, len(left["identities"][kind]))
                            for kind in ("artist", "character", "copyright")}})
        if label == "held-out-image":
            (output / "recommendation-f32.json").write_text(json.dumps(left, ensure_ascii=False, indent=2) + "\n")
            (output / "recommendation-i8.json").write_text(json.dumps(right, ensure_ascii=False, indent=2) + "\n")
    stop.set()
    watcher.join(4)
    report = {"state": "complete", "old_datasets_hidden": True, "network_namespace_isolated": True,
        "base": f32.snapshot, "compact": i8.snapshot, "live_query_sha256": query_hash,
        "stored_query_probes_by_rating": dict(counts), "query_count": len(trials),
        "fp32_exact_neighbor_matches": len(trials),
        "mean_neighbor_overlap_at_32": sum(r["neighbors_at_32_overlap"] for r in results) / len(results),
        "mean_tag_overlap_at_20": sum(r["tags_at_20_overlap"] for r in results) / len(results),
        "mean_identity_overlap_at_3": {kind: sum(r["identity_at_3_overlap"][kind] for r in results) / len(results)
                                       for kind in ("artist", "character", "copyright")},
        "cold_f32_query_seconds": cold_seconds, "probe_seconds": time.perf_counter() - started_probes,
        "elapsed_seconds": time.perf_counter() - start,
        "gpu_process_mib_max_sample": max([r["mib"] for r in samples], default=None),
        "gpu_sample_count": len(samples), "gpu_sample_interval_seconds": .5,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "results": results,
        "scope": "One new held-out image plus 32 stored-vector probes; tests compression stability, not tag accuracy."}
    (output / "gpu_samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
