"""Custom CSV rows must be usable by both the dropdowns and generation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import seasonal_fashion_prompt_node as seasonal


class SeasonalCSVTests(unittest.TestCase):
    def test_malformed_weather_and_time_rows_are_skipped_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.csv'
            path.write_text('season,value\nspring\nspring,\nspring,   \n# comment\n'
                            'spring, sunny \n', encoding='utf-8')
            rows = seasonal.load_data_from_csv(path, min_columns=2)
        self.assertEqual(rows, {'spring': [['sunny']]})
        with patch.object(seasonal, 'seasonal_weather', rows), \
                patch.object(seasonal, 'seasonal_times', rows), \
                patch.object(seasonal, 'seasonal_backgrounds', {'spring': [['outdoors', 'park']]}):
            node = seasonal.SeasonalFashionPromptNode()
            self.assertIn('sunny', node.INPUT_TYPES()['required']['weather_toggle'][0])
            self.assertEqual(node.get_background_and_situation('spring', 'random', 'random', 'off'),
                             ('outdoors, park, sunny, sunny', ''))

    def test_structured_rows_require_values_but_single_column_lists_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fashion.csv'
            path.write_text('season,category,item\nspring\nspring,top\nspring,top,\n'
                            'spring,,shirt\nspring, top , shirt \n', encoding='utf-8')
            self.assertEqual(seasonal.load_data_from_csv(path, min_columns=3),
                             {'spring': [['top', 'shirt']]})
            path.write_text('pose\nstanding\n\n# comment\nsitting\n', encoding='utf-8')
            self.assertEqual(seasonal.load_list_from_csv(path), ['standing', 'sitting'])


if __name__ == '__main__':
    unittest.main()
