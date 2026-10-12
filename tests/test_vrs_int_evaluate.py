"""Fictional in-memory labels/features only; no torch, CUDA, images or weights."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import math

from test_vrs_int import contract, entry
from vrs_int.evaluate import evaluate
from vrs_int.recommend import Recommender
from vrs_int.store import Store, digest, revision
from vrs_int.vectors import encode_vector

ROOT = Path(__file__).resolve().parents[1]


def observation(number, tags, vector, split="train", categories=None):
    row = entry(number)
    row["document"].update(tags=tags, split=split)
    row["categories"] = categories or {}
    row["vector"] = None if vector is None else encode_vector(vector)
    row["audit"] = [{"source_record": {"tags": tags}}]
    return row


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "memory")
        self.kinds = {"a artist": ["artist"], "b character": ["character"], "c work": ["copyright"]}
        self.rows = [
            observation(1, ["a artist", "b character", "c work", "d rare", "z common"],
                        [1, 0, 0, 0], categories=self.kinds),
            observation(2, ["e other", "z common"], [0, 1, 0, 0]),
            observation(10, ["a artist", "b character", "c work", "d rare", "u unseen"],
                        [1, 0, 0, 0], "test", self.kinds),
            observation(20, ["e other"], [0, 1, 0, 0], "val")]
        self.rows[0]["audit"][0]["source_record"]["tags"] = self.rows[0]["document"]["tags"] + ["highres"]
        self.snapshot = self.store.build(self.rows, contract(), "fictional", bucket_count=8)

    def tearDown(self):
        self.temp.cleanup()

    def run_eval(self, **options):
        config = dict(ks=(3, 5, 10), neighbors=1, rare_max=1)
        config.update(options)
        return evaluate(self.store.root, self.snapshot, **config)

    def test_hand_computed_metrics_categories_rare_preservation_and_errors(self):
        r = self.run_eval(include_cases=True)
        self.assertEqual(r["top_k"]["3"]["precision_micro"], 1)
        self.assertEqual(r["top_k"]["3"]["recall_micro"], 3 / 5)
        self.assertEqual(r["top_k"]["5"]["precision_macro"], 4 / 5)
        self.assertEqual(r["top_k"]["10"]["precision_micro"], 4 / 10)
        self.assertEqual(r["top_k"]["10"]["recall_micro"], 4 / 5)
        for kind in self.kinds.values():
            c = kind[0]
            self.assertEqual(r["category_top_k"][c]["3"]["recall_micro"], 1)
            self.assertEqual(r["identity_list_at_3"][c]["recall_micro"], 1)
        self.assertEqual(r["rare"]["vocabulary_preservation"], 5 / 6)
        self.assertEqual(r["rare"]["occurrence_preservation"], 5 / 6)
        self.assertEqual(r["rare"]["top_k"]["3"]["recall_micro"], 3 / 4)
        self.assertEqual(r["rare"]["top_k"]["5"]["recall_micro"], 1)
        self.assertEqual(r["unseen_top_k"]["5"]["truth"], 1)
        self.assertEqual(r["unseen_top_k"]["5"]["recall_micro"], 0)
        case = r["cases"][0]["top_k"]["5"]
        self.assertEqual(case["false_positives"], ["z common"])
        self.assertEqual(case["false_negatives"], ["u unseen"])
        self.assertEqual(r["errors"]["5"]["queries_with_false_positives"], 1)
        self.assertEqual(r["coverage"]["train_audit_tag_coverage"], 2)

    def test_val_only_unknown_categories_and_empty_denominators(self):
        r = self.run_eval(split="val")
        self.assertEqual(r["top_k"]["3"]["hits"], 1)
        self.assertIsNone(r["category_top_k"]["artist"]["3"]["recall_micro"])
        self.assertIsNone(r["unseen_top_k"]["3"]["recall_macro"])
        self.assertNotIn("cases", r)
        self.assertNotIn("per_tag", r)

    def test_train_hash_and_source_aliases_of_both_holdouts_are_blocked(self):
        rows = copy.deepcopy(self.rows)
        leaked = observation(100, ["gold leak"], [1, 0, 0, 0])
        leaked["document"]["image_sha256"] = rows[2]["document"]["image_sha256"]
        alias = observation(101, ["alias leak"], [1, 0, 0, 0])
        alias["document"]["source_ids"] = [101, 20]  # val aliases blocked during test too
        duplicate = observation(102, ["duplicate vote"], [1, 0, 0, 0])
        duplicate["document"]["image_sha256"] = rows[0]["document"]["image_sha256"]
        self.snapshot = self.store.build(rows + [leaked, alias, duplicate], contract(), "leaks", bucket_count=8)
        r = self.run_eval(include_cases=True)
        self.assertEqual(r["coverage"]["eligible_train_images"], 2)
        self.assertEqual(r["coverage"]["excluded_or_duplicate_train_observations"], 3)
        self.assertEqual(r["cases"][0]["neighbor_keys"], ["danbooru:1"])
        self.assertFalse({"gold leak", "alias leak", "duplicate vote"} & set(r["cases"][0]["top_k"]["5"]["predicted"]))

    def test_gold_changes_metrics_but_never_retrieval(self):
        before = self.run_eval(include_cases=True)
        changed = copy.deepcopy(self.rows[2])
        changed["document"]["tags"] = ["different truth"]
        changed["categories"] = {}
        old = self.snapshot
        self.snapshot = self.store.update(old, [{"op": "upsert", "key": changed["key"],
                                               "entry": changed, "expected_revision": revision(self.rows[2])}], "gold")
        after = self.run_eval(include_cases=True)
        self.assertEqual(before["cases"][0]["candidates"], after["cases"][0]["candidates"])
        self.assertEqual(after["top_k"]["5"]["hits"], 0)
        self.assertEqual(after["errors"]["5"]["zero_hit_with_recommendations"], 1)
        self.assertEqual(self.store.get(old, changed["key"]), self.rows[2])

    def test_read_only_deterministic_report_and_no_model_imports(self):
        self.store.activate(self.snapshot, None)
        before = {str(p): p.read_bytes() for p in self.store.root.rglob("*") if p.is_file()}
        with patch.dict(sys.modules, {"torch": None, "transformers": None, "vrs_int.encoder": None}):
            a = self.run_eval(include_cases=True)
            b = self.run_eval(include_cases=True)
        self.assertEqual(a, b)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.store.root.rglob("*") if p.is_file()})
        checksum = a.pop("report_sha256")
        from vrs_int.store import canonical
        self.assertEqual(digest(canonical(a)), checksum)

    def test_missing_vectors_duplicate_queries_and_no_evidence(self):
        rows = copy.deepcopy(self.rows)
        rows.append(observation(11, ["missing"], None, "test"))
        duplicate = observation(12, ["duplicate"], [1, 0, 0, 0], "test")
        duplicate["document"]["image_sha256"] = rows[2]["document"]["image_sha256"]
        self.snapshot = self.store.build(rows + [duplicate], contract(), "coverage", bucket_count=8)
        r = self.run_eval()
        self.assertEqual(r["coverage"]["evaluated"], 1)
        self.assertEqual({s["reason"] for s in r["coverage"]["skipped"]}, {"missing_vector", "duplicate_holdout_image"})
        self.snapshot = self.store.build(rows[2:4], contract(), "no train")
        r = self.run_eval()
        self.assertEqual(r["top_k"]["5"]["precision_micro"], 0)
        self.assertEqual(r["top_k"]["5"]["recall_micro"], 0)
        self.assertEqual(r["errors"]["5"]["no_recommendation"], 1)

    def test_weighted_votes_and_stable_ties_across_batch_boundaries(self):
        m = Recommender(self.store.root, self.snapshot)
        a = m.recommend_offline([1, 0, 0, 0], "f" * 64, neighbors=2, limit=10, batch_size=1)
        b = m.recommend_offline([1, 0, 0, 0], "f" * 64, neighbors=2, limit=10, batch_size=8)
        self.assertEqual(a, b)
        candidates = {c["tag"]: c for c in a["candidates"]}
        self.assertEqual(candidates["z common"]["support"], 2)
        self.assertAlmostEqual(candidates["z common"]["score"], 1)
        self.assertAlmostEqual(candidates["a artist"]["score"], 1 / (1 + math.exp(-1 / .07)))
        rows = [observation(i, ["tie"], [1, 0, 0, 0]) for i in (3, 2, 1)]
        snapshot = self.store.build(rows, contract(), "tie", bucket_count=8)
        result = Recommender(self.store.root, snapshot).recommend_offline([1, 0, 0, 0], "f" * 64, neighbors=2, batch_size=1)
        self.assertEqual([n["key"] for n in result["neighbors"]], ["danbooru:1", "danbooru:2"])

    def test_bad_options_empty_split_missing_features_and_corruption_fail(self):
        for options in ({"split": "train"}, {"backend": "auto"}, {"rare_max": 0},
                        {"ratings": ("x",)}, {"neighbors": 0}, {"batch_size": 5000}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_eval(**options)
        for ks in ((), (0,), (101,), (True,)):
            with self.assertRaises(ValueError):
                evaluate(self.store.root, self.snapshot, ks=ks)
        rows = [observation(50, ["missing"], None, "test")]
        snapshot = self.store.build(rows, contract(), "missing")
        with self.assertRaisesRegex(ValueError, "No evaluable"):
            evaluate(self.store.root, snapshot)
        with self.assertRaisesRegex(ValueError, "No holdout"):
            evaluate(self.store.root, snapshot, split="val")
        object_id = next(iter(self.store.manifest(self.snapshot)["objects"].values()))
        self.store.object_path(object_id).write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "Bucket hash mismatch"):
            self.run_eval()

    def test_int8_snapshot_metrics_and_audit_fallback(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            from vrs_int.vectors import decode_vector
            row["vector"] = encode_vector(decode_vector(row["vector"], 4), "i8")
            row["audit"] = []
        self.snapshot = self.store.build(rows, contract("i8"), "i8")
        r = self.run_eval()
        self.assertEqual(r["top_k"]["5"]["hits"], 4)
        self.assertEqual(r["coverage"]["train_audit_tag_coverage"], 0)

    def test_macro_micro_rating_filter_and_metadata_only_storage(self):
        rows = copy.deepcopy(self.rows)
        rows.append(observation(30, ["e other"], [0, 1, 0, 0], "test"))
        rows.append(observation(40, ["vectorless singleton"], None))
        self.snapshot = self.store.build(rows, contract(), "averages")
        r = self.run_eval()
        self.assertEqual(r["top_k"]["5"]["recall_micro"], 5 / 6)
        self.assertEqual(r["top_k"]["5"]["recall_macro"], .9)
        self.assertEqual(r["top_k"]["5"]["precision_micro"], .5)
        self.assertEqual(r["coverage"]["train_without_vectors"], 1)
        self.assertEqual(r["storage_preservation"]["train_observations"], 3)
        self.assertEqual(r["storage_preservation"]["original_rare_vocabulary"], 7)
        rows[1]["document"]["rating"] = "e"
        self.snapshot = self.store.build(rows, contract(), "ratings")
        filtered = self.run_eval(ratings=("g",))
        self.assertEqual(filtered["coverage"]["eligible_train_images"], 1)

    def test_production_remains_cuda_only_and_invalid_queries_fail(self):
        import types
        m = Recommender(self.store.root, self.snapshot)
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake}), self.assertRaisesRegex(RuntimeError, "CUDA required"):
            m.recommend([1, 0, 0, 0], "f" * 64)
        for vector in ([0, 0, 0, 0], [1, 0], [float("nan"), 0, 0, 0]):
            with self.assertRaises(ValueError):
                m.recommend_offline(vector, "f" * 64)

    def test_cli_numpy_and_output_guards(self):
        output = self.root / "evaluation-test.json"
        command = [sys.executable, "-m", "vrs_int", "evaluate", "--store", str(self.store.root),
                   "--snapshot", self.snapshot, "--ks", "3,5", "--neighbors", "1", "--output", str(output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        r = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(r["top_k"]["5"]["hits"], 4)
        self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)
        command[-1] = str(self.store.root / "objects" / "evaluation-new.json")
        self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)
        self.assertFalse(Path(command[-1]).exists())
        command[-1] = str(self.root / "cuda.json")
        result = subprocess.run(command + ["--backend", "cuda"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --gpu", result.stderr)
        self.assertFalse(Path(command[-1]).exists())


if __name__ == "__main__":
    unittest.main()
