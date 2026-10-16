"""Check indexed Git blobs; never print corpus contents or inspect private files.

Only reviewed fictional pair fixtures and existing documentation screenshots are
allowed as data. This is a publication guard, not a substitute for diff review.
"""
import hashlib
from pathlib import PurePosixPath
import subprocess
import sys


# Hashes pin reviewed fixtures; a name alone must not allow replaced contents.
REVIEWED = {
    "examples/gemma-image-30k.api.json": "cc2a4ad61e3e88dc122c371c2419affecc8d6518e14ab270755f4f963b2d1c39",
    "examples/gemma-q4-native.api.json": "3d59a96f5f7a0e834f42fac9d3404fc46399a0348c35a22f7fdb26518ff1c772",
    "examples/gemma-prompt-expand-native.api.json": "41a9a4c317e8912be297210f50a08ffcef05dfdd679fc06fbac0cfd46e055e33",
    "examples/gemma-tipo-native.api.json": "a1852ca607e97f6d74fdc8c52252d0664226eda79a3559904c2f5841a31e9e11",
    "examples/vrs-native-build.api.json": "80e38c6cc608e3fdc40ada0391fbb22c37b74a572f7a2fb257ef36e2691ea56c",
    "examples/vrs-native-recall.api.json": "fba7db890c6df75572013da64dedde4cc3fb3cae6dc9e59ccce2140d2d48ad31",
    "examples/vrs-native-state.json": "18ee07175c3d2e9c724e873f22f7d32fd31e73b12c2189d363fc61d0f80bd54b",
    "examples/korean-ai-writer.api.json": "5727b152f343a062452d1a41cd4cef371b70074e61f57d034b1ca09e07da791a",
    "examples/seasonal-prompt.api.json": "bd803209a454c11b941831a5ad7f8b94b3c7a4043dd01fe4f01d10bb9702528b",
    "examples/seasonal-to-tipo.api.json": "5b889482a98b95585bc65f98f3034717da88207d779a90e03745de121b0c8e97",
    "examples/seasonal-to-writer.api.json": "05a32e28d783c5127b0e7f6edb670e43c26c65b7bf26722a1b1033c93dde9f13",
    "examples/vrs-int/batch-01.jsonl": "33ed587c60fd550058ad849716b7cc284d21e949c693a44cd3488b2377ae2a1e",
    "examples/vrs-int/batch-02.jsonl": "36be159b44751600b8acf36289b983e71c66a4d5e2b3c57c611bba43c99621d3",
    "examples/vrs-int/pairs/1001.png.json": "39395f0a477a29504ce3cdb8c7ab00df6032ddc1125dde8a7131e01c0877b246",
    "examples/vrs-int/pairs/1002.jpg.json": "8687899a349eef09fb04d2960f285d62a430e32bab554fa3068c025ae26fd01b",
    "assets/gemma-tipo-example.png": "2a0c70ac315ed15b7343a7029d1481a1434a842f28dd2608b439cb6ec460de64",
    "assets/gemma-tipo-vision-example.png": "4970a83403373d789d1eed1162c7d22f7ab60e34757128e255f5ed503efb23a1",
}
BLOCKED_SUFFIXES = {
    ".jsonl", ".json", ".sqlite", ".sqlite3", ".db", ".npy", ".npz", ".pt", ".pth",
    ".safetensors", ".gguf", ".onnx", ".z", ".zip", ".tar", ".gz", ".zst",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm",
}
BLOCKED_DIRS = {"vrs-int-data", "vrs-int-models", "models", "training", "objects", "snapshots"}


def check_blob(name, mode, raw):
    if mode not in ("100644", "100755"):
        return "links, submodules and unresolved index entries require review"
    if name in REVIEWED:
        if hashlib.sha256(raw).hexdigest() != REVIEWED[name]:
            return "reviewed example or screenshot content changed"
        return None
    path = PurePosixPath(name)
    if BLOCKED_DIRS.intersection(path.parts) or path.suffix.lower() in BLOCKED_SUFFIXES:
        return "corpus/model/output artifact is not part of code distribution"
    if path.name in ("CURRENT", ".env") or path.name.startswith(".env."):
        return "private local state"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "unreviewed binary file"
    if "\0" in text:
        return "binary content in a text file"
    # Data disguised as a text file, including JSONL and renamed store objects.
    if path.suffix.lower() in (".txt", ".csv") and text.lstrip().startswith(("{", "[")):
        return "structured data must be reviewed as an explicit fictional fixture"
    return None


def check_index():
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    failures, count = [], 0
    for item in entries.split(b"\0"):
        if not item:
            continue
        metadata, name = item.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        name = name.decode("utf-8")
        raw = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root)
        reason = "unmerged index entry" if stage != "0" else check_blob(name, mode, raw)
        if reason:
            failures.append(f"{name}: {reason}")
        count += 1
    return count, failures


def main():
    count, failures = check_index()
    if failures:
        print("Code-only publication check failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Code-only check passed: {count} indexed files; no unapproved data artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
