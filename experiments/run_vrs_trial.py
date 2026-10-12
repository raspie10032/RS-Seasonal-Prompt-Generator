"""Run with python -m experiments.run_vrs_trial (CUDA PyTorch environment).

Use a short-lived process so all CUDA allocations/context are released on exit.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import threading
import time


class Monitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                for line in result.stdout.splitlines():
                    pid, mib, uuid = (v.strip() for v in line.split(","))
                    if int(pid) == os.getpid():
                        self.samples.append({"monotonic": time.perf_counter(), "mib": int(mib), "uuid": uuid})
            except (ValueError, OSError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=4)


def setup_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.20)
    torch.cuda.reset_peak_memory_stats()
    return torch


def parity(core, out):
    from experiments.vrs_cuda import AssociationCUDA
    import torch
    source = r'''
#include <cstdio>
#include "swegca_architecture/association_scalar.hpp"
int main() {
    using namespace swegca::architecture::kernel;
    for (unsigned long s : {0UL, 1UL, 17UL, 65536UL})
        for (unsigned long r : {0UL, 1UL, 23UL}) {
            auto j = judge_association({s,r});
            std::printf("%u %u\n", (unsigned)j.status(), (unsigned)j.reason());
        }
}
'''
    (out / "parity.cpp").write_text(source)
    subprocess.run(["g++", "-std=c++17", "-O1", "-I", str(core), str(out / "parity.cpp"), "-o", str(out / "parity")], check=True, capture_output=True, text=True)
    expected = [tuple(map(int, line.split())) for line in subprocess.check_output([str(out / "parity")], text=True).splitlines()]
    pairs = [(s, r) for s in (0, 1, 17, 65536) for r in (0, 1, 23)]
    with AssociationCUDA(core) as judge:
        status, reason = judge.judge(torch.tensor([s for s, _ in pairs], device="cuda"),
                                     torch.tensor([r for _, r in pairs], device="cuda"))
        empty = torch.empty(0, device="cuda", dtype=torch.int64)
        if judge.judge(empty, empty)[0].numel() != 0:
            raise RuntimeError("Empty association batch failed")
        try:
            judge.judge(torch.tensor([-1], device="cuda"), torch.tensor([0], device="cuda"))
        except ValueError:
            pass
        else:
            raise RuntimeError("Negative association counts were accepted")
    actual = list(zip(status.cpu().tolist(), reason.cpu().tolist()))
    if expected != actual:
        raise RuntimeError("Native VRS and CUDA judgments differ")
    return {"cases": len(pairs), "passed": True, "inputs": pairs, "status_reason": actual,
            "empty_batch_passed": True, "negative_counts_rejected": True}


def trial(args):
    from experiments.vrs_memory import (Memory, build_snapshot, contained, digest_file, ingest,
                                         open_ledger, read_rows)
    torch = setup_gpu()
    out = Path(args.output).resolve()
    if args.initial < 1 or args.append < 0 or args.initial + args.append > 16384:
        raise ValueError("Use 1..16384 total records for this bounded trial")
    if shutil.disk_usage(out.parent).free / shutil.disk_usage(out.parent).total < 0.20:
        raise RuntimeError("Keep at least 20% disk free")
    out.mkdir(exist_ok=False)
    start, cpu_start = time.perf_counter(), time.process_time()
    manifest = Path(args.dataset) / "processed_v1/manifest_visual_expand.jsonl"
    rows = read_rows(manifest)
    test = read_rows(manifest, "test")
    if len(rows) < args.initial + args.append:
        raise ValueError("Insufficient eligible records")
    report = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "data_root": str(Path(args.dataset).resolve()), "core_root": str(Path(args.vrs_core).resolve()),
              "source_sha256": {p.name: digest_file(p) for p in Path(__file__).parent.glob("*.py")}}
    with Monitor() as monitor, open_ledger(out / "ledger.sqlite3") as db:
        report["core_parity"] = parity(args.vrs_core, out)
        report["initial_ingest"] = ingest(db, rows[:args.initial], args.dataset)
        report["initial_snapshot"] = build_snapshot(db, out / "initial", args.vrs_core, manifest)
        initial = Memory(out / "initial")
        report["append_ingest"] = ingest(db, rows[args.initial:args.initial + args.append], args.dataset)
        report["final_snapshot"] = build_snapshot(db, out / "final", args.vrs_core, manifest)
        memory = Memory(out / "final")
        # Audit exact bytes across official splits. Perceptual near duplicates
        # and alternate image encodings remain an explicit limitation.
        train_hashes = {r["image_sha256"] for r in memory.rows}
        test_rows = []
        overlap = []
        missing = []
        for row in test:
            path = contained(args.dataset, row["image"])
            if not path.is_file():
                missing.append(row["id"])
            elif digest_file(path) in train_hashes:
                overlap.append(row["id"])
            else:
                test_rows.append(row)
        report["test_split_audit"] = {"official_records": len(test), "exact_image_overlap_excluded": overlap,
                                      "missing_images_excluded": missing, "evaluated_records": len(test_rows)}
        # Freeze both evaluation vocabulary and masking before comparing an
        # update: growing the vocabulary must not silently change the task.
        fixed_vocabulary = set(initial.vocab) & set(memory.vocab)
        report["evaluation_vocabulary"] = len(fixed_vocabulary)
        report["initial_evaluation"] = initial.evaluate(test_rows, fixed_vocabulary)
        count_before = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        report["replay_ingest"] = ingest(db, rows[:32], args.dataset)
        count_after = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        if count_before != count_after or report["replay_ingest"]["counts"].get("added", 0):
            raise RuntimeError("Replay incorrectly added observations")
        report["replay_count_unchanged"] = count_before == count_after
        report["final_evaluation"] = memory.evaluate(test_rows, fixed_vocabulary)
        # Independent Python set intersection checks for exact CUDA co-counts.
        pairs = [(0, i) for i in range(min(24, len(memory.vocab)))]
        for a, b in pairs:
            count = sum(memory.vocab[a] in row["tags"] and memory.vocab[b] in row["tags"] for row in memory.rows)
            if int(memory.data["cooccurrence"][a, b]) != count:
                raise RuntimeError("GPU co-count differs from independent record intersection")
        report["co_count_checks"] = {"pairs": len(pairs), "passed": True}
        report["queries"] = [memory.query(tags, excluded) for tags, excluded in (
            (["hatsune miku"], []), (["white dress"], []), (["school uniform"], []),
            (["umbrella"], ["rain", "rainy weather"]), (["red umbrella", "subway platform"], []),
            (["rs_unknown_character_2099"], []))]
        excluded_query = memory.query(["hatsune miku"], ["twintails"])
        if any(c["tag"] == "twintails" for c in excluded_query["candidates"]):
            raise RuntimeError("Excluded tag leaked into recommendations")
        unknown_query = memory.query(["hatsune miku", "rs_unknown_character_2099"])
        if unknown_query["verdict"] != "U" or unknown_query["candidates"]:
            raise RuntimeError("Unknown constraints were silently discarded")
        try:
            memory.query(["white dress"], ["white_dress"])
        except ValueError:
            pass
        else:
            raise RuntimeError("Conflicting query constraints were accepted")
        by_id = {r["id"]: set(r["tags"]) for r in memory.rows}
        checked = 0
        for query in report["queries"] + [excluded_query]:
            for candidate in query["candidates"]:
                for post_id in candidate["image_ids"]:
                    if not set(query["input_tags"] + [candidate["tag"]]).issubset(by_id[post_id]):
                        raise RuntimeError("Reported witness does not support the claimed relation")
                    if set(query["excluded"]) & by_id[post_id]:
                        raise RuntimeError("Excluded witness leaked into recommendations")
                    checked += 1
        report["retrieval_checks"] = {"witnesses_checked": checked, "excluded_tags_passed": True,
                                       "unknown_constraint_abstains": True, "conflicting_constraints_rejected": True,
                                       "exclusion_query": excluded_query}
        report["visual_neighbors"] = {"query_id": memory.rows[0]["id"],
                                      "neighbors": memory.visual_neighbors(memory.rows[0]["id"])}
        report["scope"] = "existing VRS association scalar on GPU + local observation ledger; not the full resident VRS runtime, no new encoder/LLM training, no online collection"
    report["resources"] = {"gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
                           "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                           "wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu_start,
                           "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                           "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                           "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                           "sampled_process_peak_mib": max((s["mib"] for s in monitor.samples), default=None),
                           "sample_interval_seconds": 0.5, "gpu_samples": len(monitor.samples),
                           "gpu_uuids": sorted({s["uuid"] for s in monitor.samples}),
                           "torch_memory_fraction_limit": 0.20}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out / "report.json"), "resources": report["resources"],
                      "initial_evaluation": report["initial_evaluation"]["means"],
                      "final_evaluation": report["final_evaluation"]["means"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("trial")
    build.add_argument("--dataset", required=True)
    build.add_argument("--vrs-core", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--initial", type=int, default=2048)
    build.add_argument("--append", type=int, default=256)
    query = sub.add_parser("query")
    query.add_argument("--memory", required=True)
    query.add_argument("--tags", required=True, help="Comma-separated exact English tags (not free text)")
    query.add_argument("--exclude", default="")
    args = parser.parse_args()
    if args.command == "trial":
        trial(args)
    else:
        setup_gpu()
        from experiments.vrs_memory import Memory
        print(json.dumps(Memory(args.memory).query(args.tags.split(","), args.exclude.split(",")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
