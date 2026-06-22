import unittest

from groundwater.fetch_public_data import extract_daily_values
from groundwater.fetch_weather_data import parse_nasa_power_daily


class PublicDataTests(unittest.TestCase):
    def test_usgs_parser_selects_one_statistic_with_most_dates(self):
        payload = {
            "value": {
                "timeSeries": [
                    {
                        "name": "USGS:site:72019:00001",
                        "values": [{"value": [{"dateTime": "2024-01-01T00:00:00", "value": "3"}]}],
                    },
                    {
                        "name": "USGS:site:72019:00002",
                        "values": [
                            {
                                "value": [
                                    {"dateTime": "2024-01-01T00:00:00", "value": "1"},
                                    {"dateTime": "2024-01-02T00:00:00", "value": "2"},
                                ]
                            }
                        ],
                    },
                ]
            }
        }

        rows, statistic = extract_daily_values(payload)

        self.assertEqual(statistic, "00002")
        self.assertEqual(rows, [("2024-01-01", 1.0), ("2024-01-02", 2.0)])

    def test_nasa_parser_maps_daily_weather_columns(self):
        payload = {
            "header": {"fill_value": -999.0},
            "properties": {
                "parameter": {
                    "PRECTOTCORR": {"20240101": 4.5},
                    "T2M": {"20240101": 23.0},
                }
            },
        }

        frame = parse_nasa_power_daily(payload)

        self.assertEqual(frame.loc[0, "rainfall_mm"], 4.5)
        self.assertEqual(frame.loc[0, "temperature_c"], 23.0)


if __name__ == "__main__":
    unittest.main()
