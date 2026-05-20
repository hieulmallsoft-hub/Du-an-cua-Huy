$env:PYTHONPATH = "src"
python -m groundwater.fetch_public_data --state-cd CA --parameter-cd 72019 --start-dt 2018-01-01 --end-dt 2025-12-31 --site-limit 80 --min-rows 365 --out-csv data/real/groundwater_real.csv
