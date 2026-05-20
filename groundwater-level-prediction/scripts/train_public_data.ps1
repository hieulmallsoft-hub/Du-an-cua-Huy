$env:PYTHONPATH = "src"
python -m groundwater.thesis_train --data data/real/groundwater_real.csv --date-col date --target-col groundwater_level --out-dir artifacts --test-size 0.2 --primary-model auto --hybrid-base var --random-state 42
