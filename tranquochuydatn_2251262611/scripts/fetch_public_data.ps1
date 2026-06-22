$env:PYTHONPATH = "src"
$ErrorActionPreference = "Stop"
$usgsJson = Join-Path $env:TEMP "usgs_groundwater_323527117050002.json"
$weatherJson = Join-Path $env:TEMP "nasa_power_323527117050002.json"
$usgsUrl = "https://waterservices.usgs.gov/nwis/dv/?format=json&sites=323527117050002&parameterCd=72019&startDT=2018-01-01&endDT=2025-12-31&siteStatus=all"
$weatherUrl = "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,T2M&community=AG&longitude=-117.083475&latitude=32.59100556&start=20180101&end=20251231&format=JSON"

Invoke-WebRequest -UseBasicParsing -Uri $usgsUrl -OutFile $usgsJson
Invoke-WebRequest -UseBasicParsing -Uri $weatherUrl -OutFile $weatherJson

python -m groundwater.fetch_public_data --input-json $usgsJson --state-cd CA --site-no 323527117050002 --station-name 018S002W22E004S --latitude 32.59100556 --longitude -117.083475 --parameter-cd 72019 --start-dt 2018-01-01 --end-dt 2025-12-31 --min-rows 365 --out-csv data/real/groundwater_real.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m groundwater.fetch_weather_data --input-json $weatherJson --groundwater-csv data/real/groundwater_real.csv --groundwater-meta data/real/groundwater_real.meta.json --out-csv data/real/groundwater_weather_real.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
