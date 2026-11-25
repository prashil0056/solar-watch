"""
NASA CDDIS GNSS TEC fetcher (optional — needs free Earthdata credentials).

Register (free): https://urs.earthdata.nasa.gov/users/new
Then set env vars:
    export EARTHDATA_USER=your_username
    export EARTHDATA_PASS=your_password

CDDIS requires auth for archive downloads. If credentials are missing, every
function here raises GNSSAuthError, and the dashboard shows "TEC: awaiting
credentials" instead of fake data.

Data product used: daily IONEX (ionosphere map exchange) TEC grids,
  https://cddis.nasa.gov/archive/gnss/products/ionex/
These are the standard GNSS-derived TEC products, not raw receiver data.
"""
import os
import requests
from datetime import datetime, timedelta, timezone

CDDIS_IONEX_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"


class GNSSAuthError(RuntimeError):
    pass


def _creds():
    u = os.environ.get("EARTHDATA_USER")
    p = os.environ.get("EARTHDATA_PASS")
    if not u or not p:
        raise GNSSAuthError(
            "Set EARTHDATA_USER and EARTHDATA_PASS (free at urs.earthdata.nasa.gov)."
        )
    return u, p


def ionex_url_for(date: datetime) -> str:
    """Build the CDDIS URL for a given day's IONEX product (COD0OPSRAP)."""
    yyyy = date.strftime("%Y")
    ddd = date.strftime("%j")
    yy = date.strftime("%y")
    # COD0OPSRAP = CODE rapid ionosphere product
    fname = f"COD0OPSRAP_{yyyy}{ddd}0000_01D_01H_GIM.INX.gz"
    return f"{CDDIS_IONEX_BASE}/{yyyy}/{ddd}/{fname}"


def download_ionex(date: datetime, out_dir: str = "data") -> str:
    """Download one day's IONEX product. Returns local path."""
    user, pw = _creds()
    url = ionex_url_for(date)
    os.makedirs(out_dir, exist_ok=True)
    local = os.path.join(out_dir, os.path.basename(url))
    with requests.get(url, auth=(user, pw), stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return local


def credentials_present() -> bool:
    return bool(os.environ.get("EARTHDATA_USER") and os.environ.get("EARTHDATA_PASS"))


if __name__ == "__main__":
    if not credentials_present():
        print("No Earthdata credentials set — TEC fetch will be skipped.")
        print("Register free: https://urs.earthdata.nasa.gov/users/new")
    else:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        path = download_ionex(yesterday)
        print(f"downloaded: {path}")