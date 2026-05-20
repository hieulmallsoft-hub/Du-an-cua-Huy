$env:PYTHONPATH = "src"
if (-not (Test-Path "data/real/groundwater_real.csv")) {
  python -m groundwater.fetch_public_data --state-cd CA --parameter-cd 72019 --start-dt 2018-01-01 --end-dt 2025-12-31 --site-limit 80 --min-rows 365 --out-csv data/real/groundwater_real.csv
} else {
  Write-Host "Using existing dataset: data/real/groundwater_real.csv"
}
python -m groundwater.thesis_train --data data/real/groundwater_real.csv --date-col date --target-col groundwater_level --out-dir artifacts --test-size 0.2 --primary-model auto --hybrid-base var --random-state 42
python -m groundwater.backtest --artifact artifacts/model.pkl --data data/real/groundwater_real.csv --date-col date --target-col groundwater_level --horizons 1,3,7,14 --min-train-size 365 --stride 7 --out-dir artifacts --make-plots
python -m groundwater.report --metrics artifacts/tuned_metrics.json --predictions artifacts/test_predictions_tuned.csv --backtest-metrics artifacts/backtest_metrics.csv --out artifacts/report.md
