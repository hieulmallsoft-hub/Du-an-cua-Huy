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


STATISTIC_NAMES = {
    "00001": "maximum",
    "00002": "minimum",
    "00003": "mean",
    "00008": "median",
}


def fetch_daily_values(
    site_no: str,
    parameter_cd: str,
    start_dt: str,
    end_dt: str,
) -> Tuple[List[Tuple[str, float]], str]:
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

    return extract_daily_values(payload)


def extract_daily_values(payload: Dict) -> Tuple[List[Tuple[str, float]], str]:
    series = payload.get("value", {}).get("timeSeries", [])
    candidates: List[Tuple[List[Tuple[str, float]], str]] = []
    for ts in series:
        value_groups = ts.get("values", [])
        if not value_groups:
            continue
        name = str(ts.get("name", ""))
        statistic_cd = name.rsplit(":", 1)[-1] if ":" in name else "unknown"
        rows: List[Tuple[str, float]] = []
        for group in value_groups:
            for item in group.get("value", []):
                dt = str(item.get("dateTime", ""))[:10]
                value = item.get("value")
                if not dt or value in ("", None):
                    continue
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                rows.append((dt, fv))
        if rows:
            candidates.append((sorted(set(rows), key=lambda x: x[0]), statistic_cd))

    if not candidates:
        return [], "unknown"

    # USGS can return maximum/minimum/mean/median as separate time series.
    # Mixing them creates duplicate dates and an invalid training sequence, so
    # select one consistent statistic with the greatest temporal coverage.
    return max(candidates, key=lambda candidate: len(candidate[0]))


def fetch_site_metadata(site_no: str) -> Dict[str, object]:
    url = "https://waterservices.usgs.gov/nwis/site/?" + urllib.parse.urlencode(
        {
            "format": "rdb",
            "sites": site_no,
            "siteOutput": "expanded",
        }
    )
    df = pd.read_csv(url, sep="\t", comment="#", dtype=str)
    if "agency_cd" not in df.columns:
        return {}
    rows = df[df["agency_cd"] == "USGS"]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    result: Dict[str, object] = {
        "station_name": str(row.get("station_nm", "")),
    }
    for source, target in [("dec_lat_va", "latitude"), ("dec_long_va", "longitude")]:
        value = pd.to_numeric(row.get(source), errors="coerce")
        if pd.notna(value):
            result[target] = float(value)
    return result


def choose_best_site(
    sites: List[str],
    parameter_cd: str,
    start_dt: str,
    end_dt: str,
    min_rows: int,
) -> Tuple[str, List[Tuple[str, float]], str]:
    best_site = ""
    best_rows: List[Tuple[str, float]] = []
    best_statistic_cd = "unknown"

    for site in sites:
        try:
            rows, statistic_cd = fetch_daily_values(site, parameter_cd, start_dt, end_dt)
        except Exception:
            continue
        if len(rows) < min_rows:
            continue
        # Keep site with most rows
        if len(rows) > len(best_rows):
            best_site = site
            best_rows = rows
            best_statistic_cd = statistic_cd
    if not best_rows:
        raise RuntimeError("No site found with enough groundwater records. Try another state or broader time range.")
    return best_site, best_rows, best_statistic_cd


def run(args: argparse.Namespace) -> None:
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input_json:
        if not args.site_no:
            raise ValueError("--site-no is required with --input-json")
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8-sig"))
        rows, statistic_cd = extract_daily_values(payload)
        site_no = args.site_no
        if len(rows) < args.min_rows:
            raise RuntimeError(f"USGS JSON only contains {len(rows)} usable rows")
    else:
        sites = [args.site_no] if args.site_no else fetch_site_list(
            state_cd=args.state_cd,
            parameter_cd=args.parameter_cd,
            site_limit=args.site_limit,
        )
        if not sites:
            raise RuntimeError("No USGS groundwater sites found for the given state.")

        site_no, rows, statistic_cd = choose_best_site(
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

    source_url = "https://waterservices.usgs.gov/nwis/dv/?" + urllib.parse.urlencode(
        {
            "format": "json",
            "sites": site_no,
            "parameterCd": args.parameter_cd,
            "startDT": args.start_dt,
            "endDT": args.end_dt,
            "siteStatus": "all",
        }
    )
    supplied_site_meta: Dict[str, object] = {}
    if args.station_name:
        supplied_site_meta["station_name"] = args.station_name
    if args.latitude is not None:
        supplied_site_meta["latitude"] = args.latitude
    if args.longitude is not None:
        supplied_site_meta["longitude"] = args.longitude
    site_meta = supplied_site_meta or fetch_site_metadata(site_no)
    meta = {
        "source": "USGS NWIS Daily Values",
        "source_url": source_url,
        "state_cd": args.state_cd,
        "site_no": site_no,
        **site_meta,
        "parameter_cd": args.parameter_cd,
        "parameter_name": "Depth to water level, feet below land surface",
        "statistic_cd": statistic_cd,
        "statistic_name": STATISTIC_NAMES.get(statistic_cd, "unknown"),
        "start_dt": args.start_dt,
        "end_dt": args.end_dt,
        "rows": len(rows),
        "unique_dates": len({date for date, _ in rows}),
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
    parser.add_argument("--site-no", type=str, default="", help="Optional fixed USGS site number")
    parser.add_argument("--input-json", type=str, default="", help="Optional previously downloaded USGS JSON")
    parser.add_argument("--station-name", type=str, default="", help="Station name for offline JSON input")
    parser.add_argument("--latitude", type=float, default=None, help="Station latitude for offline JSON input")
    parser.add_argument("--longitude", type=float, default=None, help="Station longitude for offline JSON input")
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
