import unittest

from elo import update_ratings


class EloTests(unittest.TestCase):
    def test_equal_ratings_and_a_win(self):
        self.assertEqual(update_ratings(1000, 1000, 1), (1016, 984))

    def test_tie_preserves_equal_ratings(self):
        self.assertEqual(update_ratings(1000, 1000, 0.5), (1000, 1000))

    def test_b_win_is_symmetric(self):
        self.assertEqual(update_ratings(1000, 1000, 0), (984, 1016))

    def test_invalid_result_is_rejected(self):
        with self.assertRaises(ValueError):
            update_ratings(1000, 1000, 0.25)


if __name__ == "__main__":
    unittest.main()