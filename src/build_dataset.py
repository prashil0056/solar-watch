"""
Builds the training dataset: hourly GOES X-ray features + next-24h M/X flare label.

Label definition:
    1 = an M-class or X-class flare occurred during the 24 hours after this hour
    0 = otherwise

Data leakage rules:
    - features use only past data up to and including the hour
    - the label looks forward 24h
    - splitting is chronological (handled in train_model.py)

Run:
    python -m src.build_dataset [lookback_days]

Default lookback is 90 days. Falls back to 7-day SWPC JSON if NCEI fails.
Output:
    data/training.parquet
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

from .ncei_flux import fetch_range as fetch_ncei
from .solar_data import get_solar_data_7d
from .flare_labels import fetch_1y, mx_events


def _hourly_flux(flux_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min X-ray flux to hourly aggregates."""
    df = flux_df.set_index("time_tag").sort_index()
    df["log_flux"] = np.log10(df["flux"].clip(lower=1e-12))
    hourly = df["log_flux"].resample("1h").agg(
        flux_log_mean="mean",
        flux_log_max="max",
        flux_log_min="min",
        flux_log_std="std",
    ).dropna(subset=["flux_log_mean"])
    return hourly.reset_index().rename(columns={"time_tag": "hour"})


def _rolling_features(hourly: pd.DataFrame) -> pd.DataFrame:
    out = hourly.copy().sort_values("hour").reset_index(drop=True)
    for w in (1, 3, 6, 12, 24):
        out[f"flux_mean_{w}h"] = out["flux_log_mean"].rolling(w, min_periods=1).mean()
        out[f"flux_max_{w}h"]  = out["flux_log_max"].rolling(w, min_periods=1).max()
    out["flux_delta_1h"] = out["flux_log_mean"].diff(1).fillna(0)
    out["flux_delta_3h"] = out["flux_log_mean"].diff(3).fillna(0)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"].dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"].dt.hour / 24)
    return out


def _label_hours(hourly: pd.DataFrame, flares: pd.DataFrame) -> pd.Series:
    """For each hour h, label = 1 if any M/X flare began within (h, h+24h]."""
    if flares.empty:
        return pd.Series(0, index=hourly.index, name="flare_next_24h")
    starts = np.sort(flares["begin"].dropna().values)
    labels = []
    for h in hourly["hour"]:
        h_ns = h.to_datetime64()
        end_ns = h_ns + np.timedelta64(24, "h")
        # vectorized: any start in (h, h+24h]
        hit = np.any((starts > h_ns) & (starts <= end_ns))
        labels.append(1 if hit else 0)
    return pd.Series(labels, index=hourly.index, name="flare_next_24h")


def build(lookback_days: int = 90) -> pd.DataFrame:
    print(f"[1/3] fetching {lookback_days} days of X-ray flux from NCEI...")
    end = date.today()
    start = end - timedelta(days=lookback_days)
    try:
        flux = fetch_ncei(start, end)
        print(f"      {len(flux)} rows from NCEI")
    except Exception as e:
        print(f"      NCEI failed ({e}), falling back to 7-day SWPC JSON")
        flux = get_solar_data_7d()
        print(f"      {len(flux)} rows from SWPC")

    print("[2/3] fetching flare events...")
    flares = mx_events(fetch_1y())
    print(f"      {len(flares)} M/X events")

    print("[3/3] resampling + labeling...")
    hourly = _hourly_flux(flux)
    hourly = _rolling_features(hourly)
    hourly["flare_next_24h"] = _label_hours(hourly, flares)

    cutoff = hourly["hour"].max() - pd.Timedelta(hours=24)
    hourly = hourly[hourly["hour"] <= cutoff].reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    hourly.to_parquet("data/training.parquet", index=False)
    print(f"      wrote data/training.parquet  ({len(hourly)} rows, "
          f"{hourly['flare_next_24h'].mean()*100:.2f}% positive)")
    return hourly


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    build(lookback_days=days)