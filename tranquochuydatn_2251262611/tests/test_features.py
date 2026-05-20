import unittest

from groundwater.features import build_feature_vector_from_history, parse_int_list


class FeatureTests(unittest.TestCase):
    def test_parse_int_list_sorts_deduplicates_and_rejects_non_positive_values(self):
        self.assertEqual(parse_int_list("3, 1, 2, 2, 0, -4"), [1, 2, 3])

    def test_parse_int_list_rejects_empty_result(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            parse_int_list("0,-1")

    def test_build_feature_vector_requires_enough_history(self):
        with self.assertRaisesRegex(ValueError, "Need at least 7 observations"):
            build_feature_vector_from_history(
                history_levels=[1.0, 2.0, 3.0],
                lags=[1, 7],
                rolling_windows=[3],
            )

    def test_build_feature_vector_uses_latest_history_only(self):
        features = build_feature_vector_from_history(
            history_levels=[10, 11, 12, 13, 14],
            lags=[1, 3],
            rolling_windows=[3],
            exogenous_values={"rainfall": 2.5},
            feature_columns=["lag_1", "lag_3", "roll_mean_3", "roll_std_3", "rainfall"],
        )

        self.assertEqual(features["lag_1"], 14.0)
        self.assertEqual(features["lag_3"], 12.0)
        self.assertAlmostEqual(features["roll_mean_3"], 13.0)
        self.assertAlmostEqual(features["roll_std_3"], 1.0)
        self.assertEqual(features["rainfall"], 2.5)


if __name__ == "__main__":
    unittest.main()
