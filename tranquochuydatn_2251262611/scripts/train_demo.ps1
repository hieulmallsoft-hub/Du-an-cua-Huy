$env:PYTHONPATH = "src"
python -m groundwater.train --data data/sample/groundwater.csv --date-col date --target-col groundwater_level --out-dir artifacts --test-size 0.2 --horizon 1
