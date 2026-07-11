"""
live_inference.py — orchestrates live_features*.py into the exact input
dict SeaLevelModel_V18.forward() expects, with disk caching (fetching
real satellite/climate data is slow -- minutes per city, not something
to redo on every Streamlit rerun) and consistent per-sample z-score
normalization (matching the training notebook's own zscore() utility).

Usage:
    from live_inference import LiveFeaturePipeline
    pipeline = LiveFeaturePipeline(cache_dir=CITY_FEATURE_CACHE_DIR)
    inputs = pipeline.get_features('Mumbai', lat=18.975, lon=72.826,
                                    region_key='indian_ocean')
    # inputs is a dict of numpy arrays, batch dim NOT yet added
"""
from __future__ import annotations
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np

from live_features import (
    fetch_grace_window, fetch_coastal_timelapse, fetch_ocean_chl,
    fetch_s1_backscatter, fetch_spatial_branch, fetch_era5_wind_monthly,
)
from live_features_human_short import fetch_human_branch, fetch_short_term_branch, build_anthro_4ch
from live_features_temporal_v2 import fetch_temporal_branch_real


def zscore(arr: np.ndarray) -> np.ndarray:
    """Same normalization as the training notebook's utility (cell 5):
    per-sample z-score, not a fixed global constant."""
    a = arr.astype(np.float32)
    mu, sd = np.nanmean(a), np.nanstd(a)
    return (a - mu) / sd if sd > 0 else a - mu


class LiveFeaturePipeline:
    def __init__(self, cache_dir: Path, cache_max_age_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ref_cache_dir = self.cache_dir / '_reference_data'  # OWID/NOAA/IBTrACS downloads
        self.max_age = cache_max_age_days * 86400

    def _cache_path(self, city_name: str) -> Path:
        safe = city_name.replace(' ', '_').replace('/', '-')
        return self.cache_dir / f'{safe}_features.pkl'

    def get_features(self, city_name: str, lat: float, lon: float, region_key: str,
                      force_refresh: bool = False) -> dict:
        cache_path = self._cache_path(city_name)
        if not force_refresh and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self.max_age:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

        print(f'Fetching live features for {city_name} (this is slow, cached after)...')
        t0 = time.time()

        grace = fetch_grace_window(lat, lon, months=12)
        coastal = fetch_coastal_timelapse(lat, lon, months=12)
        chl = fetch_ocean_chl(lat, lon, months=12)
        s1 = fetch_s1_backscatter(lat, lon, months=12)
        spatial = fetch_spatial_branch(lat, lon, months=12)
        human = fetch_human_branch(region_key, months=24, grace_window=grace,
                                    cache_dir=self.ref_cache_dir)
        short_term = fetch_short_term_branch(lat, lon, months=12, cache_dir=self.ref_cache_dir)
        try:
            temporal = fetch_temporal_branch_real(
                lat, lon, days=365, months=12, fetch_wind_fn=fetch_era5_wind_monthly)
        except Exception as e:
            print(f'  Real CMEMS temporal branch failed ({type(e).__name__}: {e}); '
                  f'falling back to zeros. Check COPERNICUS_USER/COPERNICUS_PASS '
                  f'and that copernicusmarine + xarray are installed.')
            temporal = np.zeros((365, 8), dtype=np.float32)

        # emissions/coastal(scalar) shapes per v_doom7_arch.py's forward()
        emissions = human  # (T=24, HUMAN_F=17) -- matches HUMAN_T, HUMAN_F directly
        coastal_scalar = human[:, 15:16]  # GRACE-derived scalar per month, as a (T,1) signal
        # anthro_4ch is NOT satellite data even at training time -- it's a
        # deterministic procedural generator conditioned on real bathymetry
        # + real emissions (confirmed against training notebook cell 81).
        # See live_features_human_short.gen_anthro_channels() for the port.
        anthro_4ch = build_anthro_4ch(city_name, lat, lon, spatial, human)

        features = {
            'spatial': zscore(spatial),
            'emissions': zscore(emissions),
            'coastal': zscore(coastal_scalar),
            'temporal': zscore(temporal),
            'short_term': zscore(short_term),
            'anthro_4ch': zscore(anthro_4ch),
            'coastal_timelapse': zscore(coastal),
            'ocean_chl': zscore(chl),
            's1_sar': zscore(s1),
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        print(f'  done in {time.time()-t0:.0f}s, cached to {cache_path}')
        return features
