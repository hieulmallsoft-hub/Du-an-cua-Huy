import unittest

from groundwater.inference import GroundwaterService


class InferenceTests(unittest.TestCase):
    def test_naive_last_baseline_predicts_last_history_value(self):
        svc = GroundwaterService(
            {
                "artifact_version": "thesis_v1",
                "model_type": "naive_last_baseline",
                "model_name": "naive_last_baseline",
                "target_col": "groundwater_level",
                "endogenous_cols": ["groundwater_level"],
                "default_target_history": [1.0, 2.0, 3.5],
                "default_endog_history": [[1.0], [2.0], [3.5]],
            }
        )

        self.assertEqual(svc.predict_next(), 3.5)
        self.assertEqual(svc.forecast(steps=3), [3.5, 3.5, 3.5])


if __name__ == "__main__":
    unittest.main()
