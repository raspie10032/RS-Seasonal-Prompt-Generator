"""Build personal memory from user-selected local Danbooru images and metadata.

Never downloads, deletes or packages images. Metadata arrives as JSONL; one image
is encoded at a time. Each successful batch creates an immutable snapshot.
"""
from pathlib import Path

from .import_snapshot import file_digest
from .manifest import read_pairs
from .identity import observation_key
from .store import Store, bucket, revision, validate_entry
from .tag_policy import POLICY_VERSION, filter_tags, typed_tags
from .vectors import encode_vector


def encoder_contract(directory, codec="f32le"):
    return {"encoder": {
        "provenance_sha256": file_digest(Path(directory) / "provenance.json"),
        "preprocess": "gemma4-float-pixels-v1", "backend": "pil", "quantization": "nf4",
        "pooling": "mean-tokens-l2-v1"}, "dimension": 768,
        "codec": codec, "tag_policy": POLICY_VERSION}


def local_entries(manifest, images, contract, encode):
    for row, image, sha in read_pairs(manifest, images):
        post_id = row["id"]
        kept, removed = filter_tags(row)
        typed = typed_tags(row)
        entry = {"key": observation_key(row), "document": {
            "id": post_id, "image": Path(row["image"]).as_posix(), "image_sha256": sha,
            "rating": row["rating"], "split": row.get("split", "train"),
            "source_ids": [post_id], "tags": kept},
            "categories": {tag: sorted(c for c, names in typed.items() if tag in names)
                           or ["unknown"] for tag in kept},
            "vector": None,
            "audit": [{"source_record": row, "removed_tags": removed}]}
        if row.get("source") == "generated":
            entry["document"]["source"] = "generated"
        validate_entry(entry, contract)
        vector = encode(image)
        if hasattr(vector, "detach"):
            vector = vector.detach().cpu().numpy()
        entry["vector"] = encode_vector(vector, contract["codec"])
        if file_digest(image) != sha:
            raise ValueError("Source image changed during encoding")
        validate_entry(entry, contract)
        yield entry


def upsert_operations(store, base, entries):
    """Bind incoming observations to an immutable base without repeated decoding.

    Cache only keys and revision hashes for each touched bucket, not source
    documents or vectors. Store.update still validates each expected revision
    independently before publishing the new snapshot.
    """
    manifest = store.manifest(base)
    revisions = {}
    for entry in entries:
        index = bucket(entry["key"], manifest["bucket_count"])
        if index not in revisions:
            revisions[index] = {previous["key"]: revision(previous)
                                for previous in store.read_bucket(manifest, index)}
        yield {"op": "upsert", "key": entry["key"], "entry": entry,
               "expected_revision": revisions[index].get(entry["key"])}


def build_personal(manifest, images, encoder_path, destination, *, label="personal-initial",
                   codec="f32le", bucket_count=256, base=None, encoder_factory=None):
    contract = encoder_contract(encoder_path, codec)
    store = Store(destination)
    if base and store.manifest(base)["contract"] != contract:
        raise ValueError("Incompatible update; re-encode into a separate base")
    if encoder_factory is None:
        from .encoder import ImageEncoder
        encoder_factory = ImageEncoder
    manifest_sha = file_digest(manifest)
    encoder = encoder_factory(encoder_path)

    def entries():
        yield from local_entries(manifest, images, contract, encoder.encode)
        if file_digest(manifest) != manifest_sha:
            raise ValueError("Input manifest changed during build")

    provenance = {"input_manifest_sha256": manifest_sha, "images_included": False,
                  "collection": "user-selected-local-images", "automatic_downloads": False}
    try:
        if base:
            snapshot = store.update(base, upsert_operations(store, base, entries()),
                                    label, contract, provenance)
        else:
            snapshot = store.build(entries(), contract, label, provenance, bucket_count)
        return store.verify(snapshot)
    finally:
        # CLI exits after each batch; release any live model immediately for
        # callers that build several batches within one Python process too.
        del encoder
        import gc
        gc.collect()
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
