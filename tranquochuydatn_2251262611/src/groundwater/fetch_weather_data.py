from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict

import pandas as pd


NASA_POWER_DAILY_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_nasa_power_daily(
    latitude: float,
    longitude: float,
    start_dt: str,
    end_dt: str,
) -> tuple[pd.DataFrame, str, Dict]:
    query = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_dt.replace("-", ""),
        "end": end_dt.replace("-", ""),
        "format": "JSON",
    }
    source_url = NASA_POWER_DAILY_ENDPOINT + "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(source_url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return parse_nasa_power_daily(payload), source_url, payload


def parse_nasa_power_daily(payload: Dict) -> pd.DataFrame:

    parameters = payload.get("properties", {}).get("parameter", {})
    rainfall = parameters.get("PRECTOTCORR", {})
    temperature = parameters.get("T2M", {})
    dates = sorted(set(rainfall) & set(temperature))
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, format="%Y%m%d", errors="coerce"),
            "rainfall_mm": [rainfall[date] for date in dates],
            "temperature_c": [temperature[date] for date in dates],
        }
    )
    weather["rainfall_mm"] = pd.to_numeric(weather["rainfall_mm"], errors="coerce")
    weather["temperature_c"] = pd.to_numeric(weather["temperature_c"], errors="coerce")
    fill_value = float(payload.get("header", {}).get("fill_value", -999.0))
    weather = weather.replace(fill_value, pd.NA).dropna().sort_values("date").reset_index(drop=True)
    return weather


def run(args: argparse.Namespace) -> None:
    groundwater_path = Path(args.groundwater_csv)
    groundwater_meta_path = Path(args.groundwater_meta)
    out_path = Path(args.out_csv)
    if not groundwater_path.exists():
        raise FileNotFoundError(f"Groundwater CSV not found: {groundwater_path}")
    if not groundwater_meta_path.exists():
        raise FileNotFoundError(f"Groundwater metadata not found: {groundwater_meta_path}")

    groundwater_meta = json.loads(groundwater_meta_path.read_text(encoding="utf-8"))
    latitude = args.latitude if args.latitude is not None else groundwater_meta.get("latitude")
    longitude = args.longitude if args.longitude is not None else groundwater_meta.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError("Latitude/longitude missing. Re-fetch USGS data or provide --latitude and --longitude.")

    groundwater = pd.read_csv(groundwater_path)
    required = {"date", "groundwater_level"}
    missing = required - set(groundwater.columns)
    if missing:
        raise ValueError(f"Missing groundwater columns: {sorted(missing)}")
    groundwater["date"] = pd.to_datetime(groundwater["date"], errors="coerce")
    groundwater["groundwater_level"] = pd.to_numeric(groundwater["groundwater_level"], errors="coerce")
    groundwater = groundwater.dropna(subset=["date", "groundwater_level"]).sort_values("date")
    duplicate_dates = int(groundwater["date"].duplicated().sum())
    if duplicate_dates:
        raise ValueError(
            f"Groundwater data contains {duplicate_dates} duplicate dates. "
            "Re-fetch it with the corrected USGS fetcher before merging weather."
        )

    start_dt = groundwater["date"].min().strftime("%Y-%m-%d")
    end_dt = groundwater["date"].max().strftime("%Y-%m-%d")
    query = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "longitude": float(longitude),
        "latitude": float(latitude),
        "start": start_dt.replace("-", ""),
        "end": end_dt.replace("-", ""),
        "format": "JSON",
    }
    weather_url = NASA_POWER_DAILY_ENDPOINT + "?" + urllib.parse.urlencode(query)
    if args.input_json:
        weather_payload = json.loads(Path(args.input_json).read_text(encoding="utf-8-sig"))
        weather = parse_nasa_power_daily(weather_payload)
    else:
        weather, weather_url, weather_payload = fetch_nasa_power_daily(
            latitude=float(latitude),
            longitude=float(longitude),
            start_dt=start_dt,
            end_dt=end_dt,
        )
    merged = groundwater.merge(weather, on="date", how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No overlapping dates between USGS and NASA POWER data.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    merged.to_csv(out_path, index=False)

    header = weather_payload.get("header", {})
    parameters = weather_payload.get("parameters", {})
    meta = {
        "dataset": "USGS groundwater joined with NASA POWER daily weather",
        "rows": int(len(merged)),
        "unique_dates": int(merged["date"].nunique()),
        "start_dt": str(merged["date"].min()),
        "end_dt": str(merged["date"].max()),
        "join": "inner one-to-one join on calendar date",
        "columns": {
            "groundwater_level": "USGS depth to water level, feet below land surface",
            "rainfall_mm": parameters.get("PRECTOTCORR", {}).get("longname", "Precipitation Corrected")
            + " (mm/day)",
            "temperature_c": parameters.get("T2M", {}).get("longname", "Temperature at 2 Meters") + " (C)",
        },
        "groundwater_source": groundwater_meta,
        "weather_source": {
            "source": "NASA POWER Daily API",
            "source_dataset": ", ".join(header.get("sources", [])),
            "source_url": weather_url,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "parameters": ["PRECTOTCORR", "T2M"],
            "time_standard": header.get("time_standard"),
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Groundwater + weather dataset built")
    print(f"- rows: {len(merged)}")
    print(f"- csv: {out_path}")
    print(f"- meta: {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join USGS groundwater with NASA POWER daily weather.")
    parser.add_argument("--groundwater-csv", default="data/real/groundwater_real.csv")
    parser.add_argument("--groundwater-meta", default="data/real/groundwater_real.meta.json")
    parser.add_argument("--out-csv", default="data/real/groundwater_weather_real.csv")
    parser.add_argument("--input-json", default="", help="Optional previously downloaded NASA POWER JSON")
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
