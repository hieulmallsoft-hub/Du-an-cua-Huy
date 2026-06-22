import tempfile
import unittest
from pathlib import Path

from groundwater.data import load_series


class DataTests(unittest.TestCase):
    def test_load_series_rejects_duplicate_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.csv"
            path.write_text(
                "date,groundwater_level\n2024-01-01,1.0\n2024-01-01,2.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate dates"):
                load_series(path, date_col="date", target_col="groundwater_level")


if __name__ == "__main__":
    unittest.main()
