"""
live_features_temporal_v2.py — REAL CMEMS ocean physics for the temporal
branch (u10, v10, uo, vo, zos, thetao, so, rho), replacing the
placeholder zos/so values in live_features_temporal.py.

Confirmed against the training notebook (Sea_Level_Prediction_System_
Data_Chunked_Version.ipynb, cells 26/28/70/81): uo, vo, zos, thetao, so
all come from the SAME CMEMS dataset (cmems_mod_glo_phy_my_0.083deg_P1M-m)
at a fixed near-surface depth (~0.494m); u10/v10 come from ERA5 wind
separately; rho is derived, not fetched.

CREDENTIALS -- read from environment variables, never hardcoded here:
    COPERNICUS_USER, COPERNICUS_PASS
Set these via a .env file (git-ignored) or Streamlit secrets.toml, e.g.:
    export COPERNICUS_USER="your_email"
    export COPERNICUS_PASS="your_password"
Your notebook (Sea_Level_Prediction_System_Data_Chunked_Version.ipynb,
cells 2 and 33) has TWO real credentials hardcoded in plaintext -- if
that notebook is ever committed to git, both leak permanently into repo
history. Rotate both and scrub the notebook before pushing anywhere
public.

IMPORTANT -- "_my_" vs "_anfc_" datasets:
The training pipeline used `cmems_mod_glo_phy_my_0.083deg_P1M-m` -- the
"_my_" ("multi-year") reanalysis product, one combined dataset covering
zos/uo/vo/thetao/so together. That product is NOT near-real-time; it
lags the present by months to a couple of years.

For genuinely live data, the near-real-time counterpart is product
GLOBAL_ANALYSISFORECAST_PHY_001_024 -- but as of a November 2022
Copernicus Marine catalog migration, this product's datasets were
"atomized" into separate per-variable-group files (confirmed against
the official Product User Manual, CMEMS-GLO-PUM-001-024):
    zos              -> cmems_mod_glo_phy_anfc_0.083deg_P1M-m   (unsuffixed -- surface fields group)
    so               -> cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m
    thetao           -> cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m
    uo, vo           -> cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m
So fetch_cmems_ocean_physics() makes 4 separate subset() calls for the
anfc path, not 1 -- there is no single combined near-real-time dataset
the way there is for the training-time reanalysis. Falls back to the
single _my_ dataset if any anfc call fails.
"""
from __future__ import annotations
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

try:
    import copernicusmarine
    HAS_CMEMS = True
except ImportError:
    HAS_CMEMS = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

DATASET_ID_MY = "cmems_mod_glo_phy_my_0.083deg_P1M-m"       # training-time, delayed reanalysis -- ONE combined dataset, not atomized

# The near-real-time "anfc" catalog was atomized in a Nov 2022 Copernicus
# Marine migration -- there is NO single combined anfc dataset like the
# training-time _my_ one. Confirmed against the official Product User
# Manual (CMEMS-GLO-PUM-001-024): zos/mlotst/etc. live in the UNSUFFIXED
# "surface fields" dataset, while so/thetao/cur each get their own
# variable-suffixed dataset. Four separate subset() calls are required.
ANFC_DATASETS = {
    "zos": ("cmems_mod_glo_phy_anfc_0.083deg_P1M-m", ["zos"]),
    "so": ("cmems_mod_glo_phy-so_anfc_0.083deg_P1M-m", ["so"]),
    "thetao": ("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1M-m", ["thetao"]),
    "cur": ("cmems_mod_glo_phy-cur_anfc_0.083deg_P1M-m", ["uo", "vo"]),
}
SURFACE_DEPTH = 0.49402499198913574  # exact depth used in training, matches its normalization

CMEMS_VARS = ["zos", "uo", "vo", "thetao", "so"]


def _get_credentials():
    user = os.environ.get("COPERNICUS_USER")
    pw = os.environ.get("COPERNICUS_PASS")
    if not user or not pw:
        raise RuntimeError(
            "COPERNICUS_USER / COPERNICUS_PASS not set. Set them as "
            "environment variables (or Streamlit secrets) -- never "
            "hardcode credentials in a committed script.")
    return user, pw


def _subset_one(dataset_id, variables, lat_min, lat_max, lon_min, lon_max,
                 start, end, user, pw, target):
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_latitude=lat_min, maximum_latitude=lat_max,
        minimum_longitude=lon_min, maximum_longitude=lon_max,
        start_datetime=start.isoformat() + "T00:00:00",
        end_datetime=end.isoformat() + "T00:00:00",
        minimum_depth=SURFACE_DEPTH, maximum_depth=SURFACE_DEPTH,
        coordinates_selection_method="strict-inside",
        output_filename=str(target),
        username=user, password=pw,
        disable_progress_bar=True,
    )


def _read_var_series(nc_path, var, months):
    with xr.open_dataset(nc_path) as ds:
        if var not in ds.data_vars:
            return np.zeros(months, dtype=np.float32)
        da = ds[var]
        t_dim = next((d for d in da.dims if 'time' in d.lower()), None)
        other_dims = [d for d in da.dims if d != t_dim]
        series = da.astype('float32').mean(dim=other_dims).values if other_dims else da.values
        series = np.asarray(series, dtype=np.float32).ravel()
        out = np.zeros(months, dtype=np.float32)
        n = min(months, len(series))
        out[-n:] = series[-n:]
        return out


def fetch_cmems_ocean_physics(lat: float, lon: float, months: int = 12,
                               half_deg: float = 0.5,
                               end: date | None = None) -> dict:
    """Real zos/uo/vo/thetao/so, most recent `months` available. Returns a
    dict of {var: (months,) spatial-mean arrays}.

    Tries the near-real-time anfc catalog first (4 separate dataset IDs,
    see ANFC_DATASETS -- confirmed against the official CMEMS Product
    User Manual). Falls back to the training-time _my_ reanalysis dataset
    (1 combined dataset ID) if anfc fails or doesn't cover the requested
    window -- that fallback works but isn't truly near-real-time.
    """
    if not HAS_CMEMS:
        raise RuntimeError("copernicusmarine not installed: pip install copernicusmarine")
    if not HAS_XARRAY:
        raise RuntimeError("xarray not installed: pip install xarray")

    user, pw = _get_credentials()
    end = end or date.today()
    start = end - timedelta(days=31 * months)
    lat_min, lat_max = lat - half_deg, lat + half_deg
    lon_min, lon_max = lon - half_deg, lon + half_deg

    out = {v: np.zeros(months, dtype=np.float32) for v in CMEMS_VARS}

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- try anfc (near-real-time), one call per atomized dataset ---
        anfc_ok = True
        for key, (dataset_id, variables) in ANFC_DATASETS.items():
            target = Path(tmpdir) / f"{key}.nc"
            try:
                _subset_one(dataset_id, variables, lat_min, lat_max, lon_min, lon_max,
                            start, end, user, pw, target)
                for v in variables:
                    out[v] = _read_var_series(target, v, months)
            except Exception as e:
                anfc_ok = False
                print(f'  CMEMS anfc fetch failed for {key} ({dataset_id}): {type(e).__name__}: {e}')

        if anfc_ok:
            print('  CMEMS fetch OK via anfc (near-real-time)')
            return out

        # --- fallback: training-time _my_ reanalysis, one combined dataset ---
        print('  Falling back to _my_ reanalysis dataset (NOT near-real-time, '
              'lags present by months-to-years -- see module docstring)')
        target = Path(tmpdir) / "ocean_physics_my.nc"
        try:
            _subset_one(DATASET_ID_MY, CMEMS_VARS, lat_min, lat_max, lon_min, lon_max,
                        start, end, user, pw, target)
            for v in CMEMS_VARS:
                out[v] = _read_var_series(target, v, months)
            print('  CMEMS fetch OK via _my_ fallback')
        except Exception as e:
            print(f'  CMEMS fetch failed on _my_ fallback too: {type(e).__name__}: {e}')
            print('  Returning zeros -- caller should treat this as "unavailable", not "measured zero".')

    return out


def _seawater_density(temp_c: np.ndarray, salinity_psu: np.ndarray) -> np.ndarray:
    """Same UNESCO 1980 linearized equation of state as before -- real
    physics, applied to real thetao/so this time instead of a placeholder."""
    t, s = temp_c, salinity_psu
    return 1000.0 + 0.7 * s - 0.2 * t - 0.005 * (t ** 2) + 0.0008 * s * t


def fetch_temporal_branch_real(lat, lon, days=365, months=12, end: date | None = None,
                                fetch_wind_fn=None) -> np.ndarray:
    """Full (T=days, F=8) temporal branch using REAL CMEMS ocean physics
    for uo/vo/zos/thetao/so + real ERA5 wind for u10/v10 (via
    fetch_wind_fn, e.g. adapt fetch_spatial_branch's ERA5 calls -- pass
    that in rather than importing Earth Engine here, to keep this module
    usable even in environments without EE configured but with CMEMS
    configured, or vice versa).
    """
    ocean = fetch_cmems_ocean_physics(lat, lon, months=months, end=end)

    if fetch_wind_fn is not None:
        u10_m, v10_m = fetch_wind_fn(lat, lon, months=months, end=end)
    else:
        print('  No wind fetch function provided -- u10/v10 will be zero. '
              'Pass fetch_wind_fn= to fetch_temporal_branch_real().')
        u10_m = np.zeros(months, dtype=np.float32)
        v10_m = np.zeros(months, dtype=np.float32)

    rho_m = _seawater_density(ocean['thetao'], ocean['so'])

    monthly = np.stack([u10_m, v10_m, ocean['uo'], ocean['vo'],
                         ocean['zos'], ocean['thetao'], ocean['so'], rho_m], axis=1)  # (months, 8)

    # Expand monthly values to a daily (days, 8) series, same "repeat
    # across the days in that period" approach as the ERA5-substitute
    # version, since these fields don't meaningfully change day to day
    # at this resolution anyway.
    out = np.zeros((days, 8), dtype=np.float32)
    days_per_month = days / months
    for d in range(days):
        m = min(int(d / days_per_month), months - 1)
        out[d] = monthly[m]
    return out
