# Checklist Truoc Ngay Bao Ve

## 1) Chay lai pipeline
- [ ] powershell -ExecutionPolicy Bypass -File .\scripts\fetch_public_data.ps1
- [ ] powershell -ExecutionPolicy Bypass -File .\scripts\train_public_data.ps1
- [ ] powershell -ExecutionPolicy Bypass -File .\scripts\backtest_public_data.ps1
- [ ] powershell -ExecutionPolicy Bypass -File .\scripts\build_report.ps1

## 2) Kiem tra file ket qua
- [ ] artifacts/model.pkl
- [ ] artifacts/tuned_metrics.json
- [ ] artifacts/backtest_metrics.csv
- [ ] artifacts/backtest_rmse.png
- [ ] artifacts/backtest_curve_last_origin.png
- [ ] artifacts/report.md
- [ ] data/real/groundwater_weather_real.csv
- [ ] data/real/groundwater_weather_real.meta.json (link nguon USGS + NASA POWER)

## 3) Kiem tra demo
- [ ] powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1
- [ ] UI mo duoc tai http://127.0.0.1:8000/
- [ ] /docs mo duoc
- [ ] Predict t+1 chay duoc
- [ ] Forecast N buoc chay duoc

## 4) Du phong
- [ ] Chup anh man hinh ket qua metrics
- [ ] Chup anh bieu do backtest
- [ ] Xuat PDF bao cao va slide
- [ ] Co ban offline cua source code
