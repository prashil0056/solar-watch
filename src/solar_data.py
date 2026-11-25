"""
NOAA SWPC GOES X-ray flux fetcher.

Live endpoint:  https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json
History (7d):   https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json

No API key required. Public JSON, updated roughly every minute.

Network notes:
  - Some ISPs / regions have flaky or blocked routes to services.swpc.noaa.gov.
  - If you see repeated ConnectionError / Timeout, try a VPN or set HTTPS_PROXY.
  - We use a short connect timeout (5s) so a dead route fails fast instead of hanging.
"""
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

LIVE_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
HISTORY_7D_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
LONG_CHANNEL = "0.1-0.8nm"

# (connect, read) — keep connect short so bad routes fail fast
TIMEOUT = (5, 20)
MAX_RETRIES = 2
RETRY_BACKOFF = 2.0  # seconds


class NOAANetworkError(RuntimeError):
    """Raised when NOAA is unreachable from this network."""


def _get_with_retry(url: str) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"[noaa] attempt {attempt} failed ({type(e).__name__}), "
                      f"retrying in {wait:.0f}s...")
                time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            # 4xx/5xx — retrying won't help
            raise NOAANetworkError(f"NOAA returned HTTP {e.response.status_code} for {url}") from e

    proxy_hint = ""
    if not os.environ.get("HTTPS_PROXY") and not os.environ.get("https_proxy"):
        proxy_hint = ("  (No HTTPS_PROXY set. If you're behind a VPN/proxy, export it: "
                      "export HTTPS_PROXY=http://host:port)")

    raise NOAANetworkError(
        f"NOAA unreachable after {MAX_RETRIES} attempts: {type(last_exc).__name__}.\n"
        f"  URL: {url}\n"
        f"  Test manually: curl -v --max-time 10 {url}\n"
        f"  If it hangs or times out, your network can't reach NOAA right now — "
        f"try a VPN (US exit) or a proxy.{proxy_hint}"
    ) from last_exc


def _parse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    df = df[df["energy"] == LONG_CHANNEL]
    df = df.dropna(subset=["time_tag", "flux"]).sort_values("time_tag").reset_index(drop=True)
    cols = ["time_tag", "flux"]
    if "observed_flux" in df.columns:
        cols.append("observed_flux")
    return df[cols]


def fetch(url: str) -> pd.DataFrame:
    r = _get_with_retry(url)
    return _parse(pd.DataFrame(r.json()))


def get_solar_data_live() -> pd.DataFrame:
    return fetch(LIVE_URL)


def get_solar_data_7d() -> pd.DataFrame:
    return fetch(HISTORY_7D_URL)


def latest_flux() -> dict:
    """Return latest observed flux + timestamp. Raises NOAANetworkError on failure."""
    df = get_solar_data_live()
    if df.empty:
        raise RuntimeError("NOAA returned no usable X-ray rows for the 0.1-0.8nm channel.")
    row = df.iloc[-1]
    return {
        "time": row["time_tag"].to_pydatetime(),
        "flux": float(row["flux"]),
        "fetched_at": datetime.now(timezone.utc),
    }


def classify_flare(flux: float) -> str:
    """NOAA flare class from long-channel X-ray flux (W/m^2)."""
    if flux >= 1e-4:   return "X"
    if flux >= 1e-5:   return "M"
    if flux >= 1e-6:   return "C"
    if flux >= 1e-7:   return "B"
    return "A"


if __name__ == "__main__":
    try:
        df = get_solar_data_live()
        print(f"rows: {len(df)}")
        print(df.tail(10).to_string(index=False))
        info = latest_flux()
        print(f"\nlatest: {info['flux']:.3e} W/m^2  ({classify_flare(info['flux'])}-class)  @ {info['time']}")
    except NOAANetworkError as e:
        print(f"\n[NOAA UNREACHABLE]\n{e}")
        raise SystemExit(1)