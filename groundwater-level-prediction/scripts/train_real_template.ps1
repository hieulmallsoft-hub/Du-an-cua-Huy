$env:PYTHONPATH = "src"
python -m groundwater.tune `
  --data data/real/groundwater_real.csv `
  --date-col date `
  --target-col groundwater_level `
  --feature-cols rainfall,temperature,pumping_rate `
  --out-dir artifacts `
  --test-size 0.2 `
  --horizon 1 `
  --cv-splits 5
