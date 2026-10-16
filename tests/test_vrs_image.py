"""Evaluation leakage boundaries; no encoder load or CUDA initialization."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.run_image_trial import score, select_tests, selection_key
from experiments.run_vrs_trial import peak_rss_mib
from experiments.vrs_image import eligible_indices


class ImageBoundaryTests(unittest.TestCase):
    def test_peak_memory_units_and_unavailable_accounting(self):
        with patch.dict("sys.modules", {"resource": None}):
            self.assertIsNone(peak_rss_mib())
        resource = SimpleNamespace(RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=1024 ** 2))
        with patch.dict("sys.modules", {"resource": resource}):
            for platform, expected in (("darwin", 1), ("linux", 1024)):
                with self.subTest(platform=platform), patch("sys.platform", platform):
                    self.assertEqual(peak_rss_mib(), expected)

    def test_train_only_and_exact_duplicate_exclusion(self):
        rows = [
            {"split": "train", "image_sha256": "same", "rating": "g"},
            {"split": "test", "image_sha256": "other", "rating": "g"},
            {"split": "val", "image_sha256": "third", "rating": "g"},
            {"split": "train", "image_sha256": "valid", "rating": "s"},
            {"split": "train", "image_sha256": "missing-vector", "rating": "g"},
        ]
        self.assertEqual(eligible_indices(rows, [1, 1, 1, 1, 0], "same"), [3])

    def test_optional_rating_filter_does_not_change_storage(self):
        rows = [{"split": "train", "image_sha256": str(i), "rating": rating}
                for i, rating in enumerate("gsqe")]
        self.assertEqual(eligible_indices(rows, [1] * 4), [0, 1, 2, 3])
        self.assertEqual(eligible_indices(rows, [1] * 4, ratings=("g", "s")), [0, 1])

    def test_fixed_selection_ignores_tags_and_input_order(self):
        rows = [{"id": i, "split": "test", "rating": rating, "tags": ["hidden"]}
                for i, rating in enumerate("ggssqqee")]
        first = select_tests(rows, 1)
        changed = [dict(row, tags=["different"]) for row in reversed(rows)]
        self.assertEqual([r["id"] for r in first], [r["id"] for r in select_tests(changed, 1)])
        self.assertEqual({r["rating"] for r in first}, set("gsqe"))
        self.assertEqual(selection_key(rows[0]), selection_key(dict(rows[0], tags=[])))

    def test_test_selection_never_uses_train_or_validation(self):
        rows = [{"id": i, "split": split, "rating": "g"}
                for i, split in enumerate(("train", "val", "test"))]
        self.assertEqual([r["id"] for r in select_tests(rows, 32)], [2])

    def test_missing_recommendations_count_against_precision(self):
        self.assertEqual(score(["a", "a"], ["a", "b"], 10), {"p_at_10": .1, "r_at_10": .5})
        self.assertEqual(score([], [], 10), {"p_at_10": 0, "r_at_10": 0})


if __name__ == "__main__":
    unittest.main()
