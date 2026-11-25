
"""
Download science-quality GOES XRS-B 1-minute data from NOAA NCEI.

Public archive:
https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/

Public interface:
    fetch_range(start, end) -> pandas.DataFrame

Returned columns:
    time_tag : UTC timestamps
    flux     : XRS-B flux in W/m^2

The downloader caches yearly NetCDF files locally so they are not
downloaded again on every run.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


BASE_URL = (
    "https://data.ngdc.noaa.gov/platforms/"
    "solar-space-observing-satellites/goes"
)

CACHE_DIR = Path("data") / "goes_cache"

# Prefer the newest operationally relevant science-quality satellite.
SATELLITES = (19, 18, 16)

FLUX_VARIABLE = "xrsb_flux"
TIME_VARIABLE = "time"
QUALITY_VARIABLE = "xrsb_flag"

HEADERS = {
    "User-Agent": "SolarWatch/1.0 research project"
}


def _get_yearly_file(satellite: int, year: int) -> str:
    """Find the actual yearly NetCDF filename in NOAA's archive."""

    archive = (
        f"{BASE_URL}/goes{satellite}/l2/data/"
        "xrsf-l2-avg1m_science/"
    )

    request = urllib.request.Request(archive, headers=HEADERS)

    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")

    # NOAA filenames look like:
    # sci_xrsf-l2-avg1m_g19_y2026_v2-2-2.nc
    pattern = (
        rf'href=["\']?([^"\'>\s]*'
        rf'xrsf-l2-avg1m_g{satellite}_y{year}_v[^/"\'>\s]*\.nc)'
    )

    matches = re.findall(pattern, html)

    if not matches:
        raise FileNotFoundError(
            f"No yearly science-quality file found for "
            f"GOES-{satellite}, {year}"
        )

    return archive + matches[0]


def _download_file(url: str, destination: Path) -> Path:
    """Download a file once, using a temporary file to avoid partial cache."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers=HEADERS)

    print(f"      Downloading {url.split('/')[-1]}")

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with open(temporary, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if temporary.stat().st_size == 0:
            raise IOError("Downloaded file is empty")

        temporary.replace(destination)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination


def _read_netcdf(path: Path) -> pd.DataFrame:
    """Read XRS-B flux and convert NOAA's time coordinate to UTC."""

    with xr.open_dataset(
        path,
        engine="netcdf4",
        decode_times=False,
    ) as dataset:

        if TIME_VARIABLE not in dataset.variables:
            raise KeyError(
                f"Missing '{TIME_VARIABLE}' in {path.name}. "
                f"Available variables: {list(dataset.variables)}"
            )

        if FLUX_VARIABLE not in dataset.variables:
            raise KeyError(
                f"Missing '{FLUX_VARIABLE}' in {path.name}. "
                f"Available variables: {list(dataset.variables)}"
            )

        raw_time = np.asarray(dataset[TIME_VARIABLE].values).reshape(-1)
        raw_flux = np.asarray(dataset[FLUX_VARIABLE].values).reshape(-1)

        time_units = dataset[TIME_VARIABLE].attrs.get("units", "")

        # NOAA documents this coordinate as seconds since
        # 2000-01-01 12:00:00 UTC, neglecting leap seconds.
        if "since" in time_units:
            unit = time_units.split("since", 1)[0].strip().lower()
            origin = time_units.split("since", 1)[1].strip()

            origin = origin.replace(" UTC", "").replace("Z", "")
            origin_time = pd.Timestamp(origin, tz="UTC")

            if unit.startswith("second"):
                timestamps = origin_time + pd.to_timedelta(
                    raw_time, unit="s"
                )
            elif unit.startswith("millisecond"):
                timestamps = origin_time + pd.to_timedelta(
                    raw_time, unit="ms"
                )
            else:
                raise ValueError(f"Unsupported NOAA time unit: {time_units}")
        else:
            raise ValueError(
                f"Unexpected time units in {path.name}: {time_units}"
            )

        flux = pd.to_numeric(
            pd.Series(raw_flux), errors="coerce"
        ).to_numpy(dtype=float)

        # NOAA's documented fill value is -9999.
        valid = np.isfinite(flux) & (flux > 0) & (flux != -9999)

        # A zero XRS-B quality flag indicates good data.
        if QUALITY_VARIABLE in dataset.variables:
            flags = np.asarray(
                dataset[QUALITY_VARIABLE].values
            ).reshape(-1)

            valid &= flags == 0

        frame = pd.DataFrame({
            "time_tag": pd.to_datetime(timestamps, utc=True),
            "flux": flux,
        })

        return frame.loc[valid].copy()


def fetch_range(start, end) -> pd.DataFrame:
    """
    Fetch XRS-B data between start and end.

    start/end can be strings or datetime objects.
    Returns a DataFrame with time_tag and flux columns.
    """

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    if end <= start:
        raise ValueError("end must be later than start")

    years = range(start.year, end.year + 1)
    errors = []

    # Find one satellite that has files for every requested year.
    selected_satellite = None
    yearly_urls = {}

    for satellite in SATELLITES:
        try:
            yearly_urls = {
                year: _get_yearly_file(satellite, year)
                for year in years
            }
            selected_satellite = satellite
            break
        except Exception as exc:
            errors.append(f"GOES-{satellite}: {exc}")

    if selected_satellite is None:
        raise RuntimeError(
            "Could not find a complete yearly archive for the requested "
            "date range.\n" + "\n".join(errors)
        )

    print(f"      Using GOES-{selected_satellite} science-quality data")

    frames = []

    for year, url in yearly_urls.items():
        filename = url.split("/")[-1]
        local_path = CACHE_DIR / filename

        print(f"      Reading {year} data...")

        downloaded_path = _download_file(url, local_path)
        frame = _read_netcdf(downloaded_path)

        # Keep only the requested date range.
        frame = frame[
            (frame["time_tag"] >= start)
            & (frame["time_tag"] <= end)
        ]

        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No valid XRS-B data found between {start} and {end}"
        )

    result = pd.concat(frames, ignore_index=True)

    result = (
        result.drop_duplicates(subset="time_tag")
        .sort_values("time_tag")
        .reset_index(drop=True)
    )

    print(
        f"      Retrieved {len(result):,} valid one-minute observations "
        f"from GOES-{selected_satellite}"
    )

    return result


if __name__ == "__main__":
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=7)

    data = fetch_range(start, end)

    print(data.head())
    print(data.tail())
    print(f"\nRows: {len(data):,}")
    print(f"Date range: {data['time_tag'].min()} to {data['time_tag'].max()}")
