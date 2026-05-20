import unittest

from groundwater.backtest import build_origin_indices, parse_horizons


class BacktestTests(unittest.TestCase):
    def test_parse_horizons_sorts_and_deduplicates(self):
        self.assertEqual(parse_horizons("7,1,3,3"), [1, 3, 7])

    def test_build_origin_indices_respects_min_origin_index(self):
        self.assertEqual(
            build_origin_indices(
                n=30,
                max_h=3,
                min_train_size=5,
                stride=5,
                min_origin_index=12,
            ),
            [12, 17, 22],
        )


if __name__ == "__main__":
    unittest.main()
