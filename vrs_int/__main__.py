"""Local VRS-INT builder, package manager and independent image recommender."""
import argparse
import json
import os
from pathlib import Path
import shutil

from .store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-input", help="Validate image/tag JSONL without a model or GPU")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--images", type=Path, required=True)
    build = commands.add_parser("build", help="Build personal memory from local images + JSONL")
    for name in ("manifest", "images", "encoder", "store"):
        build.add_argument("--" + name, type=Path, required=True)
    build.add_argument("--gpu", required=True, help="Explicit CUDA GPU UUID")
    build.add_argument("--codec", choices=("f32le", "i8"), default="f32le")
    build.add_argument("--label", default="personal-initial")
    build.add_argument("--buckets", type=int, default=256)
    build.add_argument("--base", help="Exact existing snapshot to append/revise; never auto-activates")
    migrate = commands.add_parser("migrate", help="Import verified metadata and corrected features")
    for name in ("memory", "corrected", "encoder", "store"):
        migrate.add_argument("--" + name, type=Path, required=True)
    migrate.add_argument("--codec", choices=("f32le", "i8"), default="f32le")
    migrate.add_argument("--label", default="local-initial")
    migrate.add_argument("--buckets", type=int, default=256)
    for command in ("verify", "inspect", "export", "activate", "update", "recommend", "repack"):
        sub = commands.add_parser(command)
        sub.add_argument("--store", type=Path, required=True)
        sub.add_argument("--snapshot", required=True, help="Exact content hash; no moving latest alias")
        if command == "repack":
            sub.add_argument("--output", type=Path, required=True,
                             help="New independent local store; original snapshots remain untouched")
        if command == "export":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--base", help="Exact immediate parent for a delta package")
        if command == "activate":
            sub.add_argument("--expected-current", required=True, help="Exact current hash, or 'none'")
        if command == "update":
            sub.add_argument("--operations", type=Path, required=True, help="JSONL revision-checked operations")
            sub.add_argument("--contract", type=Path, required=True, help="Producer's encoder/format contract JSON")
            sub.add_argument("--label", required=True)
        if command == "recommend":
            sub.add_argument("--encoder", type=Path, required=True)
            sub.add_argument("--image", type=Path, required=True)
            sub.add_argument("--gpu", required=True, help="CUDA GPU UUID, explicitly selected")
            sub.add_argument("--ratings", default="g,s,q,e,u", help="Include u for unrated personal generations")
            sub.add_argument("--neighbors", type=int, default=32)
            sub.add_argument("--limit", type=int, default=20)
            sub.add_argument("--tag-dictionary", type=Path, help="Optional local tag semantics CSV")
            sub.add_argument("--output", type=Path, required=True)
    install = commands.add_parser("install", help="Verify and install; never activates automatically")
    install.add_argument("--store", type=Path, required=True)
    install.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    if hasattr(os, "nice"):
        os.nice(15)
    if args.command == "check-input":
        from .manifest import check_manifest
        print(json.dumps(check_manifest(args.manifest, args.images), ensure_ascii=False, indent=2))
        return
    store = Store(args.store)
    if args.command in ("migrate", "build"):
        ancestor = args.store.resolve()
        while not ancestor.exists():
            ancestor = ancestor.parent
        disk = shutil.disk_usage(ancestor)
        if disk.free / disk.total < .20:
            raise RuntimeError("Keep at least 20% disk free")
    if args.command in ("build", "recommend"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        torch.cuda.set_per_process_memory_fraction(.20)
    if args.command == "build":
        from .build import build_personal
        result = build_personal(args.manifest, args.images, args.encoder, args.store,
                                codec=args.codec, label=args.label, bucket_count=args.buckets,
                                base=args.base)
    elif args.command == "migrate":
        from .import_snapshot import migrate
        result = migrate(args.memory, args.corrected, args.encoder, args.store,
                         args.codec, args.label, args.buckets)
    elif args.command == "verify":
        result = store.verify(args.snapshot)
    elif args.command == "inspect":
        result = store.manifest(args.snapshot)
    elif args.command == "repack":
        result = store.repack(args.snapshot, args.output)
    elif args.command == "export":
        result = store.export(args.snapshot, args.output, args.base)
    elif args.command == "install":
        result = {"installed": store.install(args.package), "active": store.current()}
    elif args.command == "activate":
        store.activate(args.snapshot, None if args.expected_current == "none" else args.expected_current)
        result = {"active": store.current()}
    elif args.command == "update":
        with args.operations.open(encoding="utf-8") as stream:
            snapshot = store.update(args.snapshot, (json.loads(line) for line in stream if line.strip()),
                                    args.label, json.loads(args.contract.read_text(encoding="utf-8")))
        result = store.verify(snapshot)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        from .recommend import image_recommend
        from .tag_dictionary import TagDictionary
        dictionary = TagDictionary(args.tag_dictionary) if args.tag_dictionary else None
        result = image_recommend(args.store, args.snapshot, args.encoder, args.image,
                                 ratings=tuple(args.ratings.split(",")), neighbors=args.neighbors, limit=args.limit,
                                 dictionary=dictionary)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
