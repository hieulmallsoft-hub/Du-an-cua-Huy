# Groundwater Level Prediction - Graduation Thesis Edition

A complete thesis-ready project for groundwater level forecasting with:
- reproducible public data acquisition
- real daily rainfall and temperature joined by station coordinates
- feature engineering for time-series
- model tuning and benchmark comparison
- leakage-safe post-train rolling-origin evaluation (multi-horizon)
- interactive web demo (UI + API)
- auto-generated experiment report

## 1) What this project predicts
- Target: `groundwater_level`
- Scope: one monitoring site per trained model
- Output:
  - `t+1` next-step prediction
  - `t+N` multi-step forecast

## 2) Core methods (thesis scope)
- Models: `VAR`, `VECM`, `LSTM`, `Hybrid (VAR/VECM + LSTM residual)`, and `naive_last_baseline`
- Diagnostics: ADF stationarity checks, Johansen cointegration rank
- Benchmark: `naive_last_baseline`
- Evaluation:
  - Holdout metrics: MAE, RMSE, MAPE, R2
  - Post-train rolling-origin evaluation for horizons `1,3,7,14`
  - Automatic model selection by holdout RMSE unless a primary model is explicitly requested

## 2.1) Data sources
- Groundwater: USGS NWIS Daily Values, site `323527117050002`, parameter `72019`
  (depth to water, feet below land surface), daily minimum statistic `00002`.
- Rainfall and temperature: NASA POWER Daily API (MERRA-2), parameters
  `PRECTOTCORR` (mm/day) and `T2M` (deg C), sampled at the USGS site coordinates
  `32.59100556, -117.083475`.
- Joined dataset: `data/real/groundwater_weather_real.csv`, one row per date.
- Exact API URLs and column definitions are stored in
  `data/real/groundwater_weather_real.meta.json`.

These are first-party public data services, not Kaggle mirrors:
- USGS Water Services: https://waterservices.usgs.gov/
- NASA POWER Data Access Viewer: https://power.larc.nasa.gov/data-access-viewer/

## 3) Project structure
```text
groundwater-level-prediction/
  app/api/main.py
  src/groundwater/
    data.py
    features.py
    models.py
    thesis_train.py
    backtest.py
    fetch_public_data.py
    report.py
    inference.py
    ts_models.py
    lstm_model.py
    hybrid.py
  scripts/
    fetch_public_data.ps1
    train_public_data.ps1
    backtest_public_data.ps1
    full_pipeline_public.ps1
    run_api.ps1
    build_report.ps1
  artifacts/
  docs/
    THESIS_REPORT_TEMPLATE.md
    DEFENSE_SLIDE_SCRIPT.md
    DEFENSE_CHECKLIST.md
```

## 4) Quick start (full thesis pipeline)
```powershell
cd groundwater-level-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

powershell -ExecutionPolicy Bypass -File .\scripts\full_pipeline_public.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1
```

Open:
- Demo UI: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`

## 5) Step-by-step scripts
1. Fetch and join public real data:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_public_data.ps1
```
2. Train VAR/VECM/LSTM + hybrid:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_public_data.ps1
```
3. Multi-horizon backtest + plots:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\backtest_public_data.ps1
```
4. Build markdown report:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_report.ps1
```

## 6) Key output files for thesis
- `artifacts/tuned_metrics.json`
- `artifacts/ablation_univariate/tuned_metrics.json`
- `artifacts/backtest_metrics.csv`
- `artifacts/backtest_summary.json`
- `artifacts/backtest_rmse.png`
- `artifacts/backtest_curve_last_origin.png`
- `artifacts/report.md`
- `artifacts/test_predictions.csv`
- `artifacts/test_predictions_tuned.csv`

## 7) Demo notes for defense
- UI already explains exactly what is predicted.
- Shows model info + dataset metadata.
- Displays trend (`TANG/GIAM/ON DINH`) and chart for history vs forecast.

## 8) Thesis documents included
- Report chapter template: `docs/THESIS_REPORT_TEMPLATE.md`
- Slide speaking script: `docs/DEFENSE_SLIDE_SCRIPT.md`
- Final-day checklist: `docs/DEFENSE_CHECKLIST.md`

## 9) Quality checks
```powershell
$env:PYTHONPATH = "src"
python -m compileall -q src app
python -m unittest discover -s tests
```

Notes:
- The default training script uses `--primary-model auto`, so the saved artifact is the candidate with the lowest holdout RMSE.
- Backtest starts after the artifact training window by default to avoid evaluating origins whose future was already seen during training.
