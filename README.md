# SolarWatch

Solar X-ray monitoring + 24h M/X flare probability forecasting + GNSS TEC status.

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    mkdir -p data

## Pipeline (run in order)

    # 1. smoke-test live NOAA data
    python -m src.solar_data

    # 2. build training set (7d flux + 1y flare events -> hourly features + labels)
    python -m src.build_dataset

    # 3. train + evaluate
    python -m src.train_model

    # 4. dashboard
    streamlit run app.py

## GNSS TEC (optional)

Register free at https://urs.earthdata.nasa.gov/users/new, then:

    export EARTHDATA_USER=your_username
    export EARTHDATA_PASS=your_password
    python -m src.gnss_data

Without credentials the dashboard shows "TEC: awaiting credentials" — it does
not fabricate data.

## What the model predicts

P(M-class or X-class flare in next 24 hours), given hourly GOES X-ray features
up to the current hour. Trained chronologically, evaluated with Brier score,
calibration, and TSS. Persistence baseline included for comparison.

## Data sources (all free, no API key)

- NOAA GOES X-ray flux:    https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json
- NOAA flare event list:   https://services.swpc.noaa.gov/json/goes/primary/xray-flares-1-year.json
- NASA CDDIS IONEX TEC:    https://cddis.nasa.gov/archive/gnss/products/ionex/ (free Earthdata login)