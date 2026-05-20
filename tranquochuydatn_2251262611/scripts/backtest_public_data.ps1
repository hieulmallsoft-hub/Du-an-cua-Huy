$env:PYTHONPATH = "src"
python -m groundwater.backtest --artifact artifacts/model.pkl --data data/real/groundwater_real.csv --date-col date --target-col groundwater_level --horizons 1,3,7,14 --min-train-size 365 --stride 7 --out-dir artifacts --make-plots
