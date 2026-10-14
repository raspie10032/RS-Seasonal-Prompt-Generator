"""Training coverage must not favor adjacent scene/format blocks."""
import random
import unittest

from experiments.train_gemma4_tipo import training_order


class TrainingOrderTests(unittest.TestCase):
    def test_each_epoch_covers_all_rows_once_without_global_rng_mutation(self):
        state = random.getstate()
        order = training_order(320, 650)
        self.assertEqual(len(order), 650)
        self.assertEqual(set(order[:320]), set(range(320)))
        self.assertEqual(set(order[320:640]), set(range(320)))
        self.assertNotEqual(order[:320], order[320:640])
        self.assertEqual(order, training_order(320, 650))
        self.assertEqual(state, random.getstate())

    def test_partial_epoch_and_invalid_sizes(self):
        self.assertEqual(len(set(training_order(320, 17))), 17)
        for size, count in ((0, 3), (4, 0), (-1, 1)):
            with self.assertRaises(ValueError):
                training_order(size, count)


if __name__ == '__main__':
    unittest.main()
