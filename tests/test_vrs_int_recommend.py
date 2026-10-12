"""Recommendation filtering keeps scene material and immutable source records."""
from copy import deepcopy
import unittest

from vrs_int.tag_policy import filter_recommendation_tags


class RecommendationFilterTests(unittest.TestCase):
    def test_quality_and_years_removed_without_changing_order_or_sources(self):
        tags = ["amazing quality", "year 2024", "red hair", "very_aesthetic", "newest",
                "YEAR:2025", "2023", "score_9", "highres", "greenhouse", "apron"]
        categories = {tag: ["unknown"] for tag in tags}
        before = deepcopy((tags, categories))
        kept, removed = filter_recommendation_tags(tags, categories)
        self.assertEqual(kept, ["red hair", "greenhouse", "apron"])
        self.assertEqual(set(removed), set(tags) - set(kept))
        self.assertEqual((tags, categories), before)

    def test_names_take_priority_even_with_conflicting_categories(self):
        tags = ["masterpiece", "year 2024", "newest", "best quality"]
        categories = {"masterpiece": ["meta", "artist"], "year 2024": ["copyright"],
                      "newest": ["character"], "best quality": ["unknown"]}
        kept, removed = filter_recommendation_tags(tags, categories)
        self.assertEqual(kept, tags[:3])
        self.assertEqual(removed, {"best quality": "quality or recency boilerplate"})

    def test_scene_words_and_unclassified_names_are_not_guessed_away(self):
        tags = ["new year", "1980s (style)", "old man", "volumetric lighting",
                "detailed background", "colorful theme", "sample artist", "year 2024 festival"]
        self.assertEqual(filter_recommendation_tags(tags, {}), (tags, {}))


if __name__ == "__main__":
    unittest.main()
