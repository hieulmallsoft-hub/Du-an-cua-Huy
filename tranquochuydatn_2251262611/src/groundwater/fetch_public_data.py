from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def fetch_site_list(state_cd: str, parameter_cd: str, site_limit: int) -> List[str]:
    site_url = "https://waterservices.usgs.gov/nwis/site/?" + urllib.parse.urlencode(
        {
            "format": "rdb",
            "stateCd": state_cd,
            "siteType": "GW",
            "hasDataTypeCd": "dv",
            "parameterCd": parameter_cd,
            "siteStatus": "active",
        }
    )
    df = pd.read_csv(site_url, sep="\t", comment="#", dtype=str)
    if "agency_cd" not in df.columns or "site_no" not in df.columns:
        raise RuntimeError("Unexpected response format from USGS site service.")

    df = df[df["agency_cd"] == "USGS"].copy()
    sites = [str(x) for x in df["site_no"].dropna().tolist()]
    return sites[:site_limit]


def fetch_daily_values(site_no: str, parameter_cd: str, start_dt: str, end_dt: str) -> List[Tuple[str, float]]:
    url = "https://waterservices.usgs.gov/nwis/dv/?" + urllib.parse.urlencode(
        {
            "format": "json",
            "sites": site_no,
            "parameterCd": parameter_cd,
            "startDT": start_dt,
            "endDT": end_dt,
            "siteStatus": "all",
        }
    )
    with urllib.request.urlopen(url, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))

    series = payload.get("value", {}).get("timeSeries", [])
    rows: List[Tuple[str, float]] = []
    for ts in series:
        values = ts.get("values", [])
        if not values:
            continue
        for item in values[0].get("value", []):
            dt = str(item.get("dateTime", ""))[:10]
            value = item.get("value")
            if not dt or value in ("", None):
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            rows.append((dt, fv))
    return rows


def choose_best_site(
    sites: List[str],
    parameter_cd: str,
    start_dt: str,
    end_dt: str,
    min_rows: int,
) -> Tuple[str, List[Tuple[str, float]]]:
    best_site = ""
    best_rows: List[Tuple[str, float]] = []

    for site in sites:
        try:
            rows = fetch_daily_values(site, parameter_cd, start_dt, end_dt)
        except Exception:
            continue
        if len(rows) < min_rows:
            continue
        # Keep site with most rows
        if len(rows) > len(best_rows):
            best_site = site
            best_rows = rows
    if not best_rows:
        raise RuntimeError("No site found with enough groundwater records. Try another state or broader time range.")
    return best_site, best_rows


def run(args: argparse.Namespace) -> None:
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sites = fetch_site_list(
        state_cd=args.state_cd,
        parameter_cd=args.parameter_cd,
        site_limit=args.site_limit,
    )
    if not sites:
        raise RuntimeError("No USGS groundwater sites found for the given state.")

    site_no, rows = choose_best_site(
        sites=sites,
        parameter_cd=args.parameter_cd,
        start_dt=args.start_dt,
        end_dt=args.end_dt,
        min_rows=args.min_rows,
    )

    rows = sorted(set(rows), key=lambda x: x[0])
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "groundwater_level"])
        writer.writerows(rows)

    meta = {
        "source": "USGS NWIS Daily Values",
        "state_cd": args.state_cd,
        "site_no": site_no,
        "parameter_cd": args.parameter_cd,
        "start_dt": args.start_dt,
        "end_dt": args.end_dt,
        "rows": len(rows),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Public dataset fetched")
    print(f"- site_no: {site_no}")
    print(f"- rows: {len(rows)}")
    print(f"- csv: {out_path}")
    print(f"- meta: {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch public groundwater daily values from USGS.")
    parser.add_argument("--state-cd", type=str, default="CA", help="US state code, for example CA, TX, FL")
    parser.add_argument("--parameter-cd", type=str, default="72019", help="USGS parameter code (72019 = depth to water)")
    parser.add_argument("--start-dt", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-dt", type=str, default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--site-limit", type=int, default=80, help="Max candidate sites to scan")
    parser.add_argument("--min-rows", type=int, default=365, help="Minimum records required")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="data/real/groundwater_real.csv",
        help="Output CSV path",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)

