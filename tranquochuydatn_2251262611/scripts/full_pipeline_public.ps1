$env:PYTHONPATH = "src"
if (-not (Test-Path "data/real/groundwater_weather_real.csv")) {
  & "$PSScriptRoot/fetch_public_data.ps1"
} else {
  Write-Host "Using existing dataset: data/real/groundwater_weather_real.csv"
}
python -m groundwater.thesis_train --data data/real/groundwater_weather_real.csv --date-col date --target-col groundwater_level --feature-cols rainfall_mm,temperature_c --out-dir artifacts --test-size 0.2 --primary-model auto --hybrid-base var --random-state 42
python -m groundwater.thesis_train --data data/real/groundwater_real.csv --date-col date --target-col groundwater_level --out-dir artifacts/ablation_univariate --test-size 0.2 --primary-model auto --hybrid-base var --random-state 42
python -m groundwater.backtest --artifact artifacts/model.pkl --data data/real/groundwater_weather_real.csv --date-col date --target-col groundwater_level --horizons 1,3,7,14 --min-train-size 365 --stride 7 --out-dir artifacts --make-plots
python -m groundwater.report --metrics artifacts/tuned_metrics.json --predictions artifacts/test_predictions_tuned.csv --backtest-metrics artifacts/backtest_metrics.csv --ablation-metrics artifacts/ablation_univariate/tuned_metrics.json --out artifacts/report.md
