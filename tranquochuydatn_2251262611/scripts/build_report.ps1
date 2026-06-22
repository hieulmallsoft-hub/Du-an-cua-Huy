$env:PYTHONPATH = "src"
python -m groundwater.report --metrics artifacts/tuned_metrics.json --predictions artifacts/test_predictions_tuned.csv --backtest-metrics artifacts/backtest_metrics.csv --ablation-metrics artifacts/ablation_univariate/tuned_metrics.json --out artifacts/report.md
