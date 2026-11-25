"""
Flare event labels for training.

Primary source: Lumara open data (NOAA SWPC GOES flares, C1.0+, JSON, no auth)
    https://lumara-space.app/data/flares.json

Fallback: NOAA SWPC 7-day flare list (live, small window)
    https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json
"""
import requests
import pandas as pd

LUMARA_FLARES = "https://lumara-space.app/data/flares.json"
SWPC_7D = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"

M_X_CLASSES = {"M", "X"}
TIMEOUT = (5, 20)


def _lumara() -> pd.DataFrame:
    r = requests.get(LUMARA_FLARES, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Lumara schema: begin, peak, end, class, peak_flux, satellite
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "begin" in lc: rename[c] = "begin"
        elif "peak" in lc and "flux" not in lc: rename[c] = "max"
        elif "end" in lc: rename[c] = "end"
        elif lc == "class" or "flare_class" in lc: rename[c] = "class"
    df = df.rename(columns=rename)

    for col in ("begin", "max", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if "class" not in df.columns:
        df["class"] = ""
    df["class"] = df["class"].astype(str).str.strip().str[:1].str.upper()
    df = df.dropna(subset=["begin"]).sort_values("begin").reset_index(drop=True)
    return df[["begin", "max", "end", "class"]]


def _swpc_7d() -> pd.DataFrame:
    r = requests.get(SWPC_7D, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df

    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "begin" in lc:  rename[c] = "begin"
        elif "max" in lc:  rename[c] = "max"
        elif "end" in lc:  rename[c] = "end"
        elif "class" in lc: rename[c] = "class"
    df = df.rename(columns=rename)

    for col in ("begin", "max", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if "class" not in df.columns:
        df["class"] = ""
    df["class"] = df["class"].astype(str).str.strip().str[:1].str.upper()
    df = df.dropna(subset=["begin"]).sort_values("begin").reset_index(drop=True)
    return df[["begin", "max", "end", "class"]]


def fetch_1y() -> pd.DataFrame:
    """Historical flare list. Tries Lumara first, falls back to SWPC 7-day."""
    try:
        df = _lumara()
        if not df.empty:
            print(f"[flares] lumara: {len(df)} events")
            return df
    except Exception as e:
        print(f"[flares] lumara failed ({type(e).__name__}), falling back to SWPC 7-day")
    df = _swpc_7d()
    print(f"[flares] swpc 7-day: {len(df)} events")
    return df


def fetch_7d() -> pd.DataFrame:
    return _swpc_7d()


def mx_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "class" not in df.columns:
        return df.iloc[0:0]
    return df[df["class"].isin(M_X_CLASSES)].reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_1y()
    mx = mx_events(df)
    print(f"total: {len(df)}  M/X: {len(mx)}")
    print(mx.tail(10).to_string(index=False))