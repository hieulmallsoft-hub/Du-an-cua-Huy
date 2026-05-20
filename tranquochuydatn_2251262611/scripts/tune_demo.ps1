$env:PYTHONPATH = "src"
python -m groundwater.tune --data data/sample/groundwater.csv --date-col date --target-col groundwater_level --out-dir artifacts --test-size 0.2 --horizon 1 --cv-splits 5
