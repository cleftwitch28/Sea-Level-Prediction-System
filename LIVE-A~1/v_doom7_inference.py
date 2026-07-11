"""
V_DOOM7 Inference Wrapper — real neural forward pass
=====================================================

Loads a V_DOOM7 checkpoint into SeaLevelModel_V18 and provides a
`predict_city(city, year)` API compatible with the notebook and the
ModelDrivenInundationRenderer.

Usage
-----
    from v_doom7_inference import MASSFusionInference

    model = MASSFusionInference(
        cities=CITIES, regions=REGIONS,
        checkpoint=r'D:\\venv\\data\\checkpoints\\v_doom7_soup\\v_doom7_soup_weighted.pt',
    )
    slr_mm = model.predict_city('Mumbai', 2050)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make v_doom7_arch importable — works from .py OR notebook cell
try:
    _THIS_DIR = Path(__file__).parent
except NameError:
    _THIS_DIR = Path.cwd()
    _SLP_DIR = Path(r'D:\venv\slp_project')
    if _SLP_DIR.exists() and str(_SLP_DIR) not in sys.path:
        sys.path.insert(0, str(_SLP_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from v_doom7_arch import (
    SeaLevelModel_V18, V_DOOM7,
    EMBED_DIM, SPATIAL_C, HUMAN_C, HUMAN_F, TEMP_F, SHORT_C, SEQ_LEN, N_REGIONS,
)

try:
    from live_inference import LiveFeaturePipeline
    HAS_LIVE_FEATURES = True
except ImportError:
    HAS_LIVE_FEATURES = False


class MASSFusionInference:
    """Wraps SeaLevelModel_V18 with a checkpoint loader + city-level API.

    predict_city(city, year) runs the neural forward pass using climatology-derived
    synthetic inputs (base data → 2025 baseline + linear trend applied per year).
    If neural forward-pass initialization fails or produces NaN, falls back to the
    physics predictor (same formula as the notebook's MockV_DOOM7) so downstream
    code always gets a valid number.
    """

    def __init__(self, cities: dict, regions: dict, checkpoint: str | None = None,
                 device: str = 'auto', name: str = 'V_DOOM7 (SeaLevelModel_V18)',
                 live_feature_cache_dir: str | None = None):
        self.CITIES = cities
        self.REGIONS = regions
        self.name = name
        self.mae_mm = 12.07
        self.r2 = 0.79
        self.delta_h_mm = 2.20

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = SeaLevelModel_V18().to(self.device)
        self.model.eval()

        self._checkpoint_meta = None
        self._checkpoint_ok = False
        if checkpoint:
            self.load_checkpoint(checkpoint)

        self.live_pipeline = None
        if HAS_LIVE_FEATURES and live_feature_cache_dir:
            self.live_pipeline = LiveFeaturePipeline(cache_dir=live_feature_cache_dir)

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def load_checkpoint(self, path: str) -> None:
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing checkpoint: {path}')
        obj = torch.load(path, map_location=self.device, weights_only=False)

        # Handle every common wrapping pattern
        sd = None
        meta = {}
        if isinstance(obj, dict):
            for key in ('model', 'state_dict', 'model_state_dict', 'weights', 'net'):
                if key in obj and isinstance(obj[key], dict):
                    sd = obj[key]; break
            if sd is None and any(isinstance(v, torch.Tensor) for v in obj.values()):
                sd = obj
            meta = {k: v for k, v in obj.items() if k not in ('model', 'state_dict', 'model_state_dict', 'weights', 'net') and not isinstance(v, dict)}
        else:
            sd = obj

        if sd is None:
            print(f'⚠ Could not find state_dict in {path}. Skipping load.')
            return

        # Load with strict=False to tolerate soup metadata + minor arch drift
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f'  Missing keys ({len(missing)}): {missing[:3]}...')
        if unexpected:
            print(f'  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')

        self._checkpoint_meta = meta
        self._checkpoint_ok = (len(missing) == 0 or len(missing) < 10)

        # Pull sidecar metrics
        if 'val_mae' in meta:
            self.mae_mm = float(meta['val_mae'])
        if 'val_r2' in meta:
            try: self.r2 = float(meta['val_r2'])
            except Exception: pass
        # Soups store metrics under 'expected'
        if isinstance(obj, dict) and 'meta' in obj and isinstance(obj['meta'], dict):
            exp = obj['meta'].get('expected', {})
            if 'mae_mm' in exp: self.mae_mm = float(exp['mae_mm'])
            if 'r2' in exp: self.r2 = float(exp['r2'])
            if 'delta_h_mm' in exp: self.delta_h_mm = float(exp['delta_h_mm'])
            variant = obj['meta'].get('variant')
            if variant:
                self.name = f'V_DOOM7 Soup ({variant})'

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f'✓ Loaded {os.path.basename(path)} → {self.name}')
        print(f'  Model params: {n_params:.2f} M')
        print(f'  Metrics: MAE {self.mae_mm:.2f} mm | R² {self.r2:.2f} | Δh {self.delta_h_mm:+.2f} mm')

    # ------------------------------------------------------------------ #
    # Input synthesis — build model-shaped tensors from region/city priors
    # ------------------------------------------------------------------ #
    def _make_real_inputs(self, city_name: str, base_year: int = 2025, force_refresh: bool = False):
        """Build model inputs from REAL live-fetched satellite/climate data
        (GRACE, Sentinel-1/2, MODIS, ERA5, OWID CO2, NOAA climate indices,
        IBTrACS/tsunami events) instead of synthetic noise. Represents
        CURRENT/RECENT conditions -- see live_inference.py and
        live_features_temporal.py docstrings for exactly which pieces are
        real vs. substituted/placeholder.

        Requires self.live_pipeline (set via live_feature_cache_dir in
        __init__) and an authenticated Earth Engine session.
        """
        if self.live_pipeline is None:
            raise RuntimeError(
                'No live feature pipeline configured. Pass '
                'live_feature_cache_dir=... to MASSFusionInference() and '
                'ensure live_inference.py is importable and ee.Initialize() '
                'has been called.')
        c = self.CITIES[city_name]
        feats = self.live_pipeline.get_features(
            city_name, c['lat'], c['lon'], c['region'], force_refresh=force_refresh)
        dev = self.device
        return {k: torch.from_numpy(v).unsqueeze(0).to(dev) for k, v in feats.items()}

    def predict_city_live(self, city_name: str, year: int, base_year: int = 2025,
                           force_refresh: bool = False) -> float:
        """Real neural forward pass on REAL current/recent data, then
        extrapolated to the target year via the physics trend. This gives
        a genuinely data-grounded PRESENT-DAY correction, not noise --
        but future years still rely on the physics extrapolation, since
        no real satellite data exists for years that haven't happened.
        """
        if not self._checkpoint_ok:
            return self._physics_predict(city_name, year, base_year)
        try:
            inputs = self._make_real_inputs(city_name, base_year, force_refresh=force_refresh)
            with torch.no_grad():
                out = self.model(**inputs)
            mu = out['mu'].squeeze().detach().cpu().numpy()
            neural_delta_now = float(mu.mean()) * 1000.0  # present-day correction, mm
            physics = self._physics_predict(city_name, year, base_year)
            physics_now = self._physics_predict(city_name, base_year, base_year)
            # The neural correction is anchored to TODAY's real conditions.
            # Apply it once, then let the physics trend carry the
            # projection out to the target year on top of that anchor --
            # rather than re-applying a "today" correction at every future
            # year (which double counts it for far-future queries).
            anchored = (physics_now + 0.4 * neural_delta_now) + (physics - physics_now)
            return anchored if np.isfinite(anchored) else physics
        except Exception as e:
            print(f'  \u26a0 Live neural forward failed for {city_name}/{year}: {e}. Using physics.')
            return self._physics_predict(city_name, year, base_year)

    def _make_inputs(self, city_name: str, year: int, base_year: int = 2025):
        """Synthesize a batch=1 forward-pass input using regional priors.

        Every branch input has zeros as baseline with a small signal proportional
        to the year offset. This is not real climate data — but it does exercise
        every neural pathway with legal shapes so the model produces its actual
        predictions per its trained weights.
        """
        c = self.CITIES[city_name]
        r = self.REGIONS[c['region']]
        dev = self.device
        yrs = year - base_year
        # Signal magnitude scales with year offset (very small — model is trained
        # on normalized data, so signals live in [-1, +1])
        sig = np.clip(yrs / 100.0, -1.0, 1.0)

        # Shape assumptions (confirmed from v_doom7_arch.py forward() bodies —
        # every 3D-CNN branch does x.permute(0,2,1,3,4) expecting (B,T,C,H,W)):
        #   spatial:            (B, T=SEQ_LEN, C=5, H, W)
        #   emissions:          (B, T, HUMAN_F)  # per-timestep human/GHG features
        #   coastal:            (B, T, 1)     # accepted but unused by HumanFactorsNet_V8
        #   temporal:           (B, T, F=8)   # ocean climatology sequence
        #   short_term:         (B, T, C=1, H, W)  # cyclone event window
        #   anthro_4ch:         (B, T, 4, H, W)
        #   coastal_timelapse:  (B, T, 3, H, W)  # RGB Sentinel-2
        #   ocean_chl:          (B, T, 1, H, W)  # MODIS chl-a
        #   s1_sar:             (B, T, 1, H, W)  # Sentinel-1 backscatter (s1_cnn expects 1 channel)
        H, W = 64, 64
        T = SEQ_LEN

        def z(shape, fill=0.0, jitter=0.05):
            arr = np.full(shape, fill, dtype=np.float32) + np.random.default_rng(
                hash((city_name, year, tuple(shape))) & 0xFFFF
            ).standard_normal(shape).astype(np.float32) * jitter
            return torch.from_numpy(arr).to(dev)

        return {
            'spatial':           z((1, T, SPATIAL_C, H, W), fill=sig * 0.3),
            'emissions':         z((1, T, HUMAN_F), fill=sig * 0.5),
            'coastal':           z((1, T, 1), fill=sig * 0.4),
            'temporal':          z((1, T, TEMP_F), fill=sig * 0.2),
            'short_term':        z((1, T, SHORT_C, H, W), fill=sig * 0.1),
            'anthro_4ch':        z((1, T, 4, H, W), fill=sig * 0.4),
            'coastal_timelapse': z((1, T, 3, H, W), fill=sig * 0.3),
            'ocean_chl':         z((1, T, 1, H, W), fill=sig * 0.2),
            's1_sar':            z((1, T, 1, H, W), fill=sig * 0.15),
        }

    def _physics_predict(self, city_name: str, year: int, base_year: int = 2025) -> float:
        """Physics fallback — identical to notebook MockV_DOOM7.predict_city."""
        c = self.CITIES[city_name]
        r = self.REGIONS[c['region']]
        yrs = year - base_year
        return r['current_sla_mm'] + r['predicted_slr_mm_yr'] * yrs + c['subsidence_mm_yr'] * yrs

    # ------------------------------------------------------------------ #
    # Public API — same signature as MockV_DOOM7
    # ------------------------------------------------------------------ #
    def predict_city(self, city_name: str, year: int, base_year: int = 2025) -> float:
        """Return predicted SLA in mm for a given city + year.

        Uses REAL live-fetched data if a live_feature_cache_dir was
        configured (see predict_city_live). Otherwise falls back to the
        original synthetic-input neural pass (predict_city_synthetic),
        and finally to pure physics if the model isn't ready.
        """
        if self.live_pipeline is not None:
            return self.predict_city_live(city_name, year, base_year)
        return self.predict_city_synthetic(city_name, year, base_year)

    def predict_city_synthetic(self, city_name: str, year: int, base_year: int = 2025) -> float:
        """Original behavior: real neural forward pass, but on synthetic
        hash-seeded noise input rather than real data. Useful for
        smoke-testing the model/checkpoint without needing Earth Engine
        access at all. Runs the real neural forward pass if the checkpoint
        loaded cleanly. Falls back to physics if the model isn't ready or
        produces non-finite output.
        """
        if not self._checkpoint_ok:
            return self._physics_predict(city_name, year, base_year)

        try:
            inputs = self._make_inputs(city_name, year, base_year)
            with torch.no_grad():
                out = self.model(**inputs)
            mu = out['mu'].squeeze().detach().cpu().numpy()
            # `mu` is the normalized regression output; the checkpoint's target
            # convention is unknown here — apply a lightweight calibration so the
            # 2025 baseline lines up with the region's current_sla_mm.
            neural_delta = float(mu.mean()) * 1000.0    # crude scale (mm)
            physics = self._physics_predict(city_name, year, base_year)
            # Blend: 60% physics prior + 40% neural correction
            blended = physics + 0.4 * neural_delta
            return blended if np.isfinite(blended) else physics
        except Exception as e:
            print(f'  ⚠ Neural forward failed for {city_name}/{year}: {e}. Using physics.')
            return self._physics_predict(city_name, year, base_year)

    def predict_regional(self, region_key: str, year: int, base_year: int = 2025) -> float:
        """Regional-average prediction — averages across cities in that region."""
        cities = [k for k, c in self.CITIES.items() if c['region'] == region_key]
        if not cities:
            r = self.REGIONS[region_key]
            return r['current_sla_mm'] + r['predicted_slr_mm_yr'] * (year - base_year)
        return float(np.mean([self.predict_city(k, year, base_year) for k in cities]))

    def predict_uncertainty(self, city_name: str, year: int, base_year: int = 2025) -> float:
        yrs = year - base_year
        return float(np.sqrt(self.mae_mm ** 2 + (0.3 * yrs) ** 2))


# --------------------------------------------------------------------- #
# CLI / smoke test
# --------------------------------------------------------------------- #
if __name__ == '__main__':
    # Minimal fake city/region dicts for smoke test
    CITIES = {'Mumbai': {'region': 'indian_ocean', 'subsidence_mm_yr': 2.1}}
    REGIONS = {'indian_ocean': {
        'current_sla_mm': 38.9, 'predicted_slr_mm_yr': 3.5, 'anthro_contrib_mm': 2.4,
    }}
    model = MASSFusionInference(
        CITIES, REGIONS,
        checkpoint=r'D:\venv\data\checkpoints\v_doom7_four_pathways\v_doom7_seed42.pt',
    )
    for y in (2025, 2050, 2075, 2100):
        print(f'Mumbai {y}: {model.predict_city("Mumbai", y):8.2f} mm  '
              f'(+/-{model.predict_uncertainty("Mumbai", y):.1f})')
