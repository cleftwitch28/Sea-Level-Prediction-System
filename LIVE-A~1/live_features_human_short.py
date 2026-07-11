"""
live_features_human_short.py — the HUMAN (emissions/climate-index/GRACE/
runoff) and SHORT_TERM (cyclone/tsunami) branches. Split into a second
file only for readability; import both into live_inference.py.

These two branches are mostly NOT Earth Engine at all:
  - CO2/CH4/N2O: OWID's public CSV (same source as training cell 12)
  - Climate indices (nao/enso/pdo/amo/soi/mei/aao): NOAA PSL plain-text
    files (same source as training cell 14)
  - River runoff: a static table of published annual discharge figures
    (training cell 16) — no live fetch needed, it's reference data
  - GRACE TWS: reuses fetch_grace_window from live_features.py
"""
from __future__ import annotations
import urllib.request
import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Static reference data (ported directly from training cell 16 — real
# published annual discharge figures, no live fetch needed)
# --------------------------------------------------------------------------- #
RIVERS_TO_REGION = {
    "Amazon": ("south_atlantic", 6900), "Congo": ("south_atlantic", 1450),
    "Ganges": ("bay_of_bengal", 1130), "Yangtze": ("south_china_sea", 960),
    "Orinoco": ("caribbean", 890), "Brahmaputra": ("bay_of_bengal", 670),
    "Mississippi": ("caribbean", 580), "Yenisei": ("north_atlantic", 580),
    "Lena": ("north_pacific", 530), "Mekong": ("south_china_sea", 470),
    "Ob": ("north_atlantic", 410), "Niger": ("south_atlantic", 380),
    "Zambezi": ("indian_ocean", 250), "Indus": ("indian_ocean", 240),
    "Danube": ("mediterranean", 200), "Rhine": ("north_sea", 70),
    "Murray": ("south_pacific", 25),
}
SEASONAL_PEAK_MONTH = {
    "indian_ocean": 8, "bay_of_bengal": 8, "south_china_sea": 7,
    "north_atlantic": 5, "north_pacific": 5,
    "south_atlantic": 3, "south_pacific": 3,
    "mediterranean": 4, "north_sea": 4, "caribbean": 9,
    # regions added for the expanded city list -- reuse a nearby basin's
    # seasonal pattern as a reasonable default rather than leaving unset
    "gulf_of_mexico": 5, "baltic_sea": 4, "black_sea": 4,
    "red_sea": 8, "persian_gulf": 8, "gulf_of_guinea": 9,
    "east_china_sea": 7, "arctic_ocean": 6,
}


def region_annual_runoff_km3(region_key: str) -> float:
    return sum(km3 for r, km3 in RIVERS_TO_REGION.values() if r == region_key)


def runoff_for_month(region_key: str, month: int) -> float:
    """Same seasonal-peaking shape as training: a smooth bump around the
    region's known flood-season month, spread across the year."""
    annual = region_annual_runoff_km3(region_key)
    if annual == 0:
        return 0.0
    peak = SEASONAL_PEAK_MONTH.get(region_key, 6)
    # cosine bump centered on peak month, same idea as the notebook's approach
    phase = 2 * np.pi * (month - peak) / 12.0
    weight = 0.5 + 0.5 * np.cos(phase)
    # normalize so 12 months sum to the annual total
    weights = [0.5 + 0.5 * np.cos(2 * np.pi * (mm - peak) / 12.0) for mm in range(1, 13)]
    return annual * (weight / sum(weights))


# --------------------------------------------------------------------------- #
# OWID CO2 data (real download, small file, cache locally)
# --------------------------------------------------------------------------- #
_OWID_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

_COUNTRY_TO_REGION_DEFAULT = {
    # Coarse country -> region weighting for allocating global CO2 to a
    # basin. This is a simplification (countries don't map 1:1 to ocean
    # basins) but mirrors what the training notebook did.
    "United States": {"north_atlantic": 0.5, "north_pacific": 0.5},
    "China": {"south_china_sea": 0.6, "east_china_sea": 0.4},
    "India": {"indian_ocean": 0.5, "bay_of_bengal": 0.5},
    "Japan": {"east_china_sea": 0.5, "north_pacific": 0.5},
    "Germany": {"north_sea": 1.0},
    "United Kingdom": {"north_sea": 1.0},
    "Brazil": {"south_atlantic": 1.0},
    "Indonesia": {"south_china_sea": 0.5, "indian_ocean": 0.5},
    "Nigeria": {"gulf_of_guinea": 1.0},
    "Saudi Arabia": {"persian_gulf": 0.7, "red_sea": 0.3},
    "Egypt": {"mediterranean": 0.5, "red_sea": 0.5},
    "Mexico": {"gulf_of_mexico": 1.0},
    "Russia": {"arctic_ocean": 0.3, "baltic_sea": 0.3, "black_sea": 0.4},
}


def fetch_owid_co2(cache_dir: Path) -> pd.DataFrame:
    """Download (or reuse cached) OWID CO2 CSV, real published data."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / 'owid-co2-data.csv'
    if not target.exists() or target.stat().st_size < 1000:
        req = urllib.request.Request(_OWID_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            target.write_bytes(r.read())
    return pd.read_csv(target)


def regional_co2_series(region_key: str, years: list[int], cache_dir: Path) -> dict:
    """Real annual CO2 (Mt) allocated to a region via country weights."""
    df = fetch_owid_co2(cache_dir)
    total = {y: 0.0 for y in years}
    for country, weights in _COUNTRY_TO_REGION_DEFAULT.items():
        w = weights.get(region_key)
        if not w:
            continue
        sub = df[(df['country'] == country) & (df['year'].isin(years))]
        for _, row in sub.iterrows():
            if pd.notna(row.get('co2')):
                total[int(row['year'])] += w * float(row['co2'])
    return total


# --------------------------------------------------------------------------- #
# NOAA PSL climate indices (real download, plain text, cache locally)
# --------------------------------------------------------------------------- #
_INDEX_SOURCES = {
    "nao": "https://psl.noaa.gov/data/correlation/nao.data",
    "enso": "https://psl.noaa.gov/data/correlation/nina34.data",
    "pdo": "https://psl.noaa.gov/data/correlation/pdo.data",
    "amo": "https://psl.noaa.gov/data/correlation/amon.us.data",
    "soi": "https://psl.noaa.gov/data/correlation/soi.data",
    "mei": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
    "aao": "https://psl.noaa.gov/data/correlation/aao.data",
}


def _parse_psl(text: str) -> dict:
    """NOAA PSL .data format: header line, then 'year v1 v2 ... v12' rows,
    terminated by a line of missing-value sentinels. Same parser shape as
    training cell 14."""
    lines = text.strip().split('\n')
    out = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if not (1900 < year < 2100):
            continue
        for m, v in enumerate(parts[1:], start=1):
            try:
                val = float(v)
                if abs(val) < 90:  # PSL uses large sentinel values like -999.9 for missing
                    out[(year, m)] = val
            except ValueError:
                pass
    return out


def fetch_climate_indices(cache_dir: Path) -> dict:
    """Real NOAA climate indices, cached locally. Returns
    {index_name: {(year, month): value}}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in _INDEX_SOURCES.items():
        target = cache_dir / f'{name}.data'
        if not target.exists() or target.stat().st_size < 100:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as r:
                    target.write_bytes(r.read())
            except Exception as e:
                print(f'  climate index {name} fetch failed: {type(e).__name__}')
                out[name] = {}
                continue
        out[name] = _parse_psl(target.read_text(errors='ignore'))
    return out


# --------------------------------------------------------------------------- #
# NOAA GML global CH4/N2O (real, confirmed against training notebook cell 19,
# which already hardcoded these exact filenames -- ch4_mm_gl.txt/n2o_mm_gl.txt
# -- reading them from a locally pre-downloaded folder. This fetches the same
# files live instead. NOTE: NOAA's own documentation states these global
# means lag real-time by ~3-4 months (shipping + QC delay) -- "live" here
# means "most recent available," not "this month."
# --------------------------------------------------------------------------- #
_GML_GLOBAL_URLS = {
    "co2": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.txt",
    "ch4": "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.txt",
    "n2o": "https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.txt",
}


def fetch_gml_global_series(gas: str, cache_dir: Path) -> dict:
    """Real NOAA GML global monthly mean for co2/ch4/n2o. Returns
    {(year, month): value}. File format per training cell 19:
    whitespace-separated, '#' comments, columns are (year, month, value,
    ...extra columns ignored)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f'{gas}_mm_gl.txt'
    if not target.exists() or target.stat().st_size < 500:
        url = _GML_GLOBAL_URLS[gas]
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            target.write_bytes(r.read())
    out = {}
    for line in target.read_text(errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            year, month, value = int(parts[0]), int(parts[1]), float(parts[2])
            out[(year, month)] = value
        except ValueError:
            continue
    return out


def global_ch4_n2o_series(months_ym: list, cache_dir: Path) -> tuple:
    """Real global CH4 (ppb) and N2O (ppb) for a list of (year, month)
    tuples, forward/backward-filled from the nearest available real
    reading (handles NOAA's 3-4 month reporting lag)."""
    ch4_data = fetch_gml_global_series('ch4', cache_dir)
    n2o_data = fetch_gml_global_series('n2o', cache_dir)

    def _fill(data, months_ym):
        if not data:
            return np.zeros(len(months_ym), dtype=np.float32)
        known_ts = sorted(data.keys())
        out = np.zeros(len(months_ym), dtype=np.float32)
        for i, (y, m) in enumerate(months_ym):
            if (y, m) in data:
                out[i] = data[(y, m)]
            else:
                target_ts = y * 12 + m
                nearest = min(known_ts, key=lambda ym: abs((ym[0] * 12 + ym[1]) - target_ts))
                out[i] = data[nearest]
        return out

    return _fill(ch4_data, months_ym), _fill(n2o_data, months_ym)


# --------------------------------------------------------------------------- #
# Assemble the 17-feature HUMAN branch, real data throughout
# --------------------------------------------------------------------------- #
def fetch_human_branch(region_key: str, months: int, grace_window: np.ndarray,
                        cache_dir: Path, end: date | None = None) -> np.ndarray:
    """Real 17-feature human/anthropogenic branch, (T=months, F=17), matching
    HumanBranchLoader_V4's feature order:
      0-2   co2_regional, ch4_global, n2o_global
      3-5   detrended residuals (linear-fit residuals within window)
      6     co2 acceleration (2nd difference)
      7     co2 cumulative anomaly within window
      8-14  7 climate indices: nao, enso, pdo, amo, soi, mei, aao
      15    GRACE TWS (spatial mean of the fetched GRACE window)
      16    river runoff (km^3/month, region-allocated)
    """
    end = end or date.today()
    y0, m0 = end.year, end.month
    year_list = sorted({y0 - 1, y0})  # covers the lookback window
    co2 = regional_co2_series(region_key, year_list, cache_dir)
    climate = fetch_climate_indices(cache_dir)

    months_ym = []
    y, m = y0, m0
    for _ in range(months):
        months_ym.append((y, m))
        m -= 1
        if m == 0:
            m = 12; y -= 1
    months_ym = list(reversed(months_ym))

    co2_series = np.array([co2.get(y, 0.0) for y, m in months_ym], dtype=np.float32)
    ch4_series, n2o_series = global_ch4_n2o_series(months_ym, cache_dir)

    def _residual(series):
        if series.std() > 0:
            idx = np.arange(len(series))
            slope, intercept = np.polyfit(idx, series, 1)
            return series - (slope * idx + intercept)
        return np.zeros_like(series)

    co2_residual = _residual(co2_series)
    ch4_residual = _residual(ch4_series)
    n2o_residual = _residual(n2o_series)
    accel = np.gradient(np.gradient(co2_series)) if len(co2_series) > 2 else np.zeros_like(co2_series)
    cum_anom = np.cumsum(co2_series - co2_series.mean())

    out = np.zeros((months, 17), dtype=np.float32)
    grace_scalar = grace_window.mean(axis=(1, 2, 3)) if grace_window is not None else np.zeros(months)
    for t, (y, m) in enumerate(months_ym):
        out[t, 0] = co2_series[t]
        out[t, 1] = ch4_series[t]
        out[t, 2] = n2o_series[t]
        out[t, 3] = co2_residual[t]
        out[t, 4] = ch4_residual[t]
        out[t, 5] = n2o_residual[t]
        out[t, 6] = accel[t] if t < len(accel) else 0.0
        out[t, 7] = cum_anom[t]
        for j, name in enumerate(["nao", "enso", "pdo", "amo", "soi", "mei", "aao"]):
            out[t, 8 + j] = climate.get(name, {}).get((y, m), 0.0)
        out[t, 15] = grace_scalar[t] if t < len(grace_scalar) else 0.0
        out[t, 16] = runoff_for_month(region_key, m)
    return out


# --------------------------------------------------------------------------- #
# SHORT-TERM branch: cyclone/tsunami events near a city, rendered as
# Gaussian-blob frames, matching ShortTermBranchLoader's approach.
# --------------------------------------------------------------------------- #
_IBTRACS_CSV_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
                     "v04r00/access/csv/ibtracs.last3years.list.v04r00.csv")
_TSUNAMI_API = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events"


def fetch_ibtracs_recent(cache_dir: Path) -> pd.DataFrame:
    """Real NOAA IBTrACS cyclone tracks, last-3-years CSV (much smaller
    than the full multi-decade archive used at training time, but real
    and sufficient for a 'recent conditions' live query)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / 'ibtracs_last3years.csv'
    if not target.exists() or target.stat().st_size < 1000:
        req = urllib.request.Request(_IBTRACS_CSV_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            target.write_bytes(r.read())
    # IBTrACS CSV has a units row right after the header -- skip it
    df = pd.read_csv(target, skiprows=[1], low_memory=False)
    return df


def fetch_tsunamis_near(lat, lon, radius_deg=10.0, cache_dir: Path = None) -> list:
    """Real NOAA NCEI tsunami event list, filtered to events near the city."""
    import json
    cache_path = (cache_dir / 'tsunamis.json') if cache_dir else None
    if cache_path and cache_path.exists() and cache_path.stat().st_size > 100:
        data = json.loads(cache_path.read_text())
    else:
        try:
            req = urllib.request.Request(_TSUNAMI_API, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            if cache_path:
                cache_path.write_text(json.dumps(data))
        except Exception as e:
            print(f'  tsunami DB fetch failed: {type(e).__name__}')
            return []
    items = data.get('items', data) if isinstance(data, dict) else data
    out = []
    for ev in items:
        try:
            elat, elon = float(ev['latitude']), float(ev['longitude'])
            if abs(elat - lat) < radius_deg and abs(elon - lon) < radius_deg:
                out.append((elat, elon, float(ev.get('maxWaterHeight', 1.0)), ev.get('year')))
        except Exception:
            continue
    return out


def _gaussian_blob(H, W, cy_frac, cx_frac, intensity, sigma_frac=1 / 6):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H * cy_frac, W * cx_frac
    sigma = H * sigma_frac
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return (g * intensity).astype(np.float32)


def fetch_short_term_branch(lat, lon, months=12, H=64, W=64, cache_dir: Path = None,
                             radius_deg=8.0, end: date | None = None) -> np.ndarray:
    """Real cyclone (IBTrACS) + tsunami (NOAA NCEI) events near a city,
    most recent N months, rendered as (T=months, C=1, H, W) intensity frames."""
    end = end or date.today()
    out = np.zeros((months, 1, H, W), dtype=np.float32)

    months_ym = []
    y, m = end.year, end.month
    for _ in range(months):
        months_ym.append((y, m))
        m -= 1
        if m == 0:
            m = 12; y -= 1
    months_ym = list(reversed(months_ym))
    month_index = {ym: i for i, ym in enumerate(months_ym)}

    try:
        df = fetch_ibtracs_recent(cache_dir)
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        near = df[(df['LAT'].sub(lat).abs() < radius_deg) & (df['LON'].sub(lon).abs() < radius_deg)]
        wind_col = 'USA_WIND' if 'USA_WIND' in near.columns else 'WMO_WIND'
        for _, row in near.iterrows():
            if pd.isna(row['ISO_TIME']):
                continue
            ym = (row['ISO_TIME'].year, row['ISO_TIME'].month)
            if ym not in month_index:
                continue
            t = month_index[ym]
            wind = pd.to_numeric(row.get(wind_col), errors='coerce')
            intensity = float(wind) / 100.0 if pd.notna(wind) else 0.3
            cy_frac = np.clip(0.5 + (row['LAT'] - lat) / (2 * radius_deg), 0.05, 0.95)
            cx_frac = np.clip(0.5 + (row['LON'] - lon) / (2 * radius_deg), 0.05, 0.95)
            out[t, 0] += _gaussian_blob(H, W, cy_frac, cx_frac, intensity)
    except Exception as e:
        print(f'  IBTrACS processing failed: {type(e).__name__}')

    for elat, elon, height, year in fetch_tsunamis_near(lat, lon, radius_deg, cache_dir):
        if year is None:
            continue
        candidates = [i for (y, m), i in month_index.items() if y == int(year)]
        if not candidates:
            continue
        t = candidates[0]
        cy_frac = np.clip(0.5 + (elat - lat) / (2 * radius_deg), 0.05, 0.95)
        cx_frac = np.clip(0.5 + (elon - lon) / (2 * radius_deg), 0.05, 0.95)
        out[t, 0] += _gaussian_blob(H, W, cy_frac, cx_frac, min(height / 5.0, 2.0))

    return out


# --------------------------------------------------------------------------- #
# anthro_4ch: NOT real satellite data -- confirmed against training notebook
# cell 81 ("Dr. Strange's Spell"). Even at training time this branch was a
# deterministic, seeded procedural generator conditioned on REAL bathymetry
# (coastline shape) and REAL emissions data -- not fetched imagery. Ported
# here verbatim from that cell, so live inference reproduces exactly what
# the checkpoint was actually trained on, rather than a different guess.
# --------------------------------------------------------------------------- #
from scipy.ndimage import gaussian_filter as _gaussian_filter
from scipy.ndimage import zoom as _zoom_anthro


def coastline_field(bath_128: np.ndarray, H: int = 64, W: int = 64) -> np.ndarray:
    """High values near coastline (where humans concentrate). Verbatim
    port of training cell 81's coastline_field()."""
    bath = _zoom_anthro(bath_128, (H / 128, W / 128), order=1)
    gx, gy = np.gradient(bath)
    coast = np.sqrt(gx ** 2 + gy ** 2)
    coast = _gaussian_filter(coast, sigma=1.5)
    coast = (coast - coast.mean()) / (coast.std() + 1e-6)
    return coast.astype(np.float32)


def gen_anthro_channels(em_window: np.ndarray, bath_128: np.ndarray, bbox,
                         T: int = 12, H: int = 64, W: int = 64, seed=None) -> np.ndarray:
    """em_window: (T, 17). Returns (T, 4, H, W) for [NTL, AOD, Pop,
    Industrial]. Verbatim port of training cell 81's gen_anthro_channels()
    -- same formula, so live inference matches what the checkpoint saw
    during training, not a fresh guess.
    """
    rng = np.random.default_rng(seed)
    coast = coastline_field(bath_128, H, W)
    lats = np.linspace(bbox[0], bbox[1], H)
    lat_w = np.exp(-((np.abs(lats) - 40) / 25) ** 2).astype(np.float32)
    lat_field = np.tile(lat_w[:, None], (1, W))

    out = np.zeros((T, 4, H, W), dtype=np.float32)
    for t in range(T):
        em_t = em_window[t]
        co2_r, ch4, n2o, det_co2, det_ch4, det_n2o, accel, cum = em_t[:8]
        climate = em_t[8:15]
        grace_s, runoff = em_t[15], em_t[16]

        ntl_base = 0.6 * coast * lat_field
        ntl_base = ntl_base + 0.3 * np.tanh(co2_r) * (coast > 0).astype(np.float32)
        ntl_noise = _gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=3)
        out[t, 0] = ntl_base + 0.15 * ntl_noise

        clim_amp = float(np.std(climate))
        aod_pattern = _gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=2)
        out[t, 1] = (0.4 * clim_amp * aod_pattern
                     + 0.3 * lat_field * np.tanh(det_co2)
                     + 0.2 * coast)

        pop_static = coast * lat_field
        pop_growth = 1.0 + 0.05 * t / T
        out[t, 2] = pop_static * pop_growth + 0.1 * np.tanh(cum)

        hotspots = (rng.standard_normal((H, W)) > 1.5).astype(np.float32)
        hotspots = _gaussian_filter(hotspots, sigma=1.0)
        out[t, 3] = (hotspots * coast * (1.0 + np.tanh(accel))
                     + 0.1 * lat_field)

    return out


def build_anthro_4ch(city_name: str, lat: float, lon: float,
                      spatial_array: np.ndarray, human_array: np.ndarray,
                      end: date | None = None, half_deg: float = 0.75) -> np.ndarray:
    """Assembles the real inputs gen_anthro_channels() needs, adapted from
    training's per-REGION call down to per-CITY:
      - bathymetry: channel index 4 of spatial_array (see live_features.py
        fetch_spatial_branch's channel order [sst,u10,v10,mslp,bathy]),
        any timestep works since bathymetry is broadcast/static there.
      - em_window: last 12 months of the real fetched human_array.
      - bbox: training used the whole REGION's lat bounds for the latitude
        weighting field; for a per-city query, a tight box around the
        city's own latitude is more meaningful than a whole ocean basin's
        span, so that's what's used here.
      - seed: same deterministic hash-based approach as training, adapted
        to (city_name, date) instead of (region, date).
    """
    end = end or date.today()
    bath_128 = spatial_array[-1, 4]  # (128, 128), broadcast bathymetry channel
    T = 12
    em_window = human_array[-T:] if human_array.shape[0] >= T else \
        np.pad(human_array, ((T - human_array.shape[0], 0), (0, 0)), mode='edge')
    bbox = (lat - half_deg, lat + half_deg)
    seed = hash((city_name, end.isoformat(), 'anthro')) % (2 ** 31)
    return gen_anthro_channels(em_window, bath_128, bbox, T=T, seed=seed)
