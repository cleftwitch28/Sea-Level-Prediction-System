"""
basin_imagery.py — Satellite mosaic fetch + smooth, texture-aware
inundation overlay for MASSFusion.

Exports:
  fetch_basin_mosaic(bbox, zoom)  → PIL.Image (ESRI World Imagery mosaic)
  fetch_ee_composite(bbox, year)  → PIL.Image (optional Sentinel-2 via GEE)
  apply_inundation(img, slr_mm, storm_risk, scale) → PIL.Image
"""
from __future__ import annotations
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter

# --------------------------------------------------------------------------- #
# 1) MOSAIC FETCH — ESRI World Imagery tiles (public, no auth)
# --------------------------------------------------------------------------- #
ESRI_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
ESRI_LABELS = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")
HEADERS = {'User-Agent': 'MASSFusion/1.0'}


def _deg2num(lat, lon, zoom):
    lat_rad = math.radians(max(-85.0, min(85.0, lat)))
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _fetch_tile(url, timeout=8, retries=3, backoff=0.6, mode='RGB'):
    """Fetch a single tile, retrying on failure. ESRI's public tile
    endpoint has no auth and will drop/reject requests under load —
    a single scattered tile failure here is what leaves a blank
    256x256 hole in the mosaic that no downstream masking logic can
    truly repair (a real gap in a real coastline just looks blocky).
    Retrying with a short backoff is the actual fix for that; the
    no-data masking is a second line of defense for whatever still
    fails after retries.

    `mode` matters: label/reference tiles are transparent PNGs, and
    forcing them to 'RGB' here would silently throw away that alpha
    channel at the source. Converting an already-RGB image to 'RGBA'
    later does NOT recover transparency — it just bolts on a uniform,
    fully-opaque alpha channel, turning what should be a transparent
    overlay into a solid opaque layer that blanks out whatever it's
    composited on top of. Always fetch label tiles with mode='RGBA'.
    """
    import time
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)
            if r.status_code == 200 and len(r.content) > 300:
                return Image.open(BytesIO(r.content)).convert(mode)
            # 429/5xx are worth retrying; 404 (genuinely no tile at this
            # z/x/y) is not, so don't burn retries on it.
            if r.status_code == 404:
                return None
        except Exception as e:
            last_exc = e
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    return None


def fetch_basin_mosaic(bbox, zoom=5, max_tiles=60, with_labels=True):
    # bbox is [S, N, W, E] -- this matches every region in
    # massfusion_meta.json (verified: e.g. bay_of_bengal = [5, 25, 78, 100]
    # means lat 5-25N, lon 78-100E). A previous version of this function
    # unpacked it as [S, W, N, E], which silently swapped latitude and
    # longitude bounds and made every fetch request a nonsensical,
    # mostly out-of-coverage tile range -- that was the actual source of
    # the giant blank/placeholder regions, not the water/overlay logic.
    S, N, W_, E = bbox
    for z in range(zoom, 1, -1):
        x0, y1 = _deg2num(N, W_, z)
        x1, y0 = _deg2num(S, E, z)
        nx, ny = x1 - x0 + 1, y0 - y1 + 1
        if nx * ny <= max_tiles and nx > 0 and ny > 0:
            zoom = z; break
    else:
        zoom, nx, ny = 2, 4, 3
        x0, y1 = _deg2num(N, W_, zoom)

    TILE = 256
    canvas = Image.new('RGB', (nx * TILE, ny * TILE), (10, 15, 22))
    lbl_canvas = Image.new('RGBA', (nx * TILE, ny * TILE), (0, 0, 0, 0))
    # Track which tiles actually had imagery so the overlay step can
    # skip painting flood colors over blank/no-data canvas fill. (Kept
    # as a fast-path hint only — apply_inundation does NOT rely on this
    # surviving a save/reload, see _detect_flat_nodata below.)
    no_data = np.ones((ny * TILE, nx * TILE), dtype=bool)
    import time
    for i in range(nx):
        for j in range(ny):
            tile = _fetch_tile(ESRI_URL.format(z=zoom, x=x0 + i, y=y1 + j))
            if tile:
                canvas.paste(tile, (i * TILE, j * TILE))
                no_data[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = False
            else:
                # One extra delayed attempt for a tile that failed even
                # after _fetch_tile's own retries — a longer cooldown
                # sometimes succeeds where rapid retries don't.
                time.sleep(1.0)
                tile = _fetch_tile(ESRI_URL.format(z=zoom, x=x0 + i, y=y1 + j), retries=2, backoff=1.2)
                if tile:
                    canvas.paste(tile, (i * TILE, j * TILE))
                    no_data[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = False
            if with_labels:
                lbl = _fetch_tile(ESRI_LABELS.format(z=zoom, x=x0 + i, y=y1 + j), mode='RGBA')
                if lbl:
                    lbl_canvas.paste(lbl, (i * TILE, j * TILE))
            # Small delay between requests so we don't burst the public,
            # unauthenticated ESRI endpoint faster than it tolerates.
            time.sleep(0.05)
    if with_labels:
        canvas = Image.alpha_composite(canvas.convert('RGBA'), lbl_canvas).convert('RGB')
    canvas = ImageEnhance.Contrast(canvas).enhance(1.05)
    canvas = ImageEnhance.Color(canvas).enhance(1.10)

    canvas.info['no_data_mask'] = no_data
    canvas.info['zoom'] = zoom
    canvas.info['center_lat'] = (S + N) / 2.0
    return canvas


# --------------------------------------------------------------------------- #
# 2) WATER DETECTION — texture-adaptive
# --------------------------------------------------------------------------- #
def _local_std(brightness, K=9):
    try:
        from scipy.ndimage import uniform_filter
        m1 = uniform_filter(brightness, size=K)
        m2 = uniform_filter(brightness * brightness, size=K)
        return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    except ImportError:
        gx = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, :1]))
        gy = np.abs(np.diff(brightness, axis=0, prepend=brightness[:1, :]))
        return (gx + gy) * 0.5


def _otsu_threshold(values, n_bins=64):
    v = values.ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.05
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return (lo + hi) * 0.5
    hist, edges = np.histogram(v, bins=n_bins, range=(lo, hi))
    p = hist.astype(np.float64) / hist.sum()
    cum = np.cumsum(p)
    cum_mean = np.cumsum(p * (edges[:-1] + edges[1:]) * 0.5)
    total_mean = cum_mean[-1]
    denom = cum * (1 - cum)
    denom[denom < 1e-9] = 1e-9
    between_var = (total_mean * cum - cum_mean) ** 2 / denom
    idx = int(np.argmax(between_var))
    return float(edges[idx])


def _water_mask(arr):
    """arr = HxWx3 float32 in [0,1]. Returns bool water mask.

    NOTE: this function does not know about no-data/blank tiles at all
    — by design. It's a pure "does this look like water" classifier.
    apply_inundation is responsible for subtracting no-data pixels
    afterward via _detect_flat_nodata, which is a *stricter* flatness
    test than anything in here. Do not try to make this function smart
    about no-data too (e.g. by excluding dark pixels): that just makes
    two different heuristics fight each other. Keep no-data exclusion
    in exactly one place.
    """
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.float32)
    max_c = np.maximum(np.maximum(R, G), B)
    min_c = np.minimum(np.minimum(R, G), B)
    chroma = max_c - min_c

    ls = _local_std(brightness, K=9)
    thresh = _otsu_threshold(ls)
    thresh = float(np.clip(thresh, 0.020, 0.060))
    smooth = ls < thresh

    plausible_color = ~((brightness > 0.60) & (chroma > 0.30))
    bluish = (B >= R - 0.015) & (B >= G - 0.05)
    dark_water = brightness < 0.22
    water_colored = bluish | dark_water

    candidate = smooth & plausible_color & water_colored
    candidate = _keep_large_components(candidate, min_frac=0.003)

    water = candidate

    frac = float(water.mean())
    if frac < 0.01 or frac > 0.85:
        return np.zeros_like(water)

    if frac > 0.55:
        land = ~water
        b_water = float(brightness[water].mean()) if water.any() else 0.0
        b_land = float(brightness[land].mean()) if land.any() else 0.0
        if b_land < b_water:
            water = land

    return water


def _keep_large_components(mask, min_frac=0.003):
    if not mask.any():
        return mask
    try:
        from scipy.ndimage import label
    except ImportError:
        return mask
    lbl, n = label(mask)
    if n == 0:
        return mask
    total = mask.size
    counts = np.bincount(lbl.ravel())
    keep_ids = [i for i in range(1, n + 1) if counts[i] >= total * min_frac]
    if not keep_ids:
        return np.zeros_like(mask)
    return np.isin(lbl, keep_ids)


# --------------------------------------------------------------------------- #
# 3) OVERLAY
# --------------------------------------------------------------------------- #
def _distance_transform(mask):
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(mask)
    except ImportError:
        H, W = mask.shape
        D = np.where(mask, np.inf, 0.0).astype(np.float32)
        ORTHO, DIAG = 1.0, 1.4142135
        for y in range(H):
            for x in range(W):
                if D[y, x] == 0: continue
                v = D[y, x]
                if y > 0:
                    v = min(v, D[y-1, x] + ORTHO)
                    if x > 0: v = min(v, D[y-1, x-1] + DIAG)
                    if x < W-1: v = min(v, D[y-1, x+1] + DIAG)
                if x > 0: v = min(v, D[y, x-1] + ORTHO)
                D[y, x] = v
        for y in range(H-1, -1, -1):
            for x in range(W-1, -1, -1):
                if D[y, x] == 0: continue
                v = D[y, x]
                if y < H-1:
                    v = min(v, D[y+1, x] + ORTHO)
                    if x > 0: v = min(v, D[y+1, x-1] + DIAG)
                    if x < W-1: v = min(v, D[y+1, x+1] + DIAG)
                if x < W-1: v = min(v, D[y, x+1] + ORTHO)
                D[y, x] = v
        return D


def _gaussian_blur(a, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma)
    except ImportError:
        img = Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.asarray(img, dtype=np.float32) / 255.0


def _detect_flat_nodata(arr, min_frac=0.01, std_thresh=0.012):
    """Detect blank/placeholder regions directly from pixel content —
    no metadata required, no color-range guessing.

    Real satellite imagery, even calm open water or a shadowed inlet,
    still carries per-pixel sensor/compression noise. A synthetic
    fallback-fill block (used when tile fetches fail) is perfectly
    flat, regardless of *what color* that fill happens to be — black
    canvas background, an olive "imagery unavailable" placeholder tile,
    whatever. So this test intentionally does NOT look at brightness or
    hue at all: flatness alone is the signal. Mixing in a brightness
    condition here (as a previous version of this function did, to
    "let the black canvas remain water") is what caused the overlay to
    trace the shape of missing tiles — don't reintroduce that.
    """
    brightness = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    ls = _local_std(brightness, K=15)
    flat = ls < std_thresh
    return _keep_large_components(flat, min_frac=min_frac)


def apply_inundation(img, slr_mm, storm_risk='high', scale='city'):
    """Dual-layer inundation overlay:
      🔵 Cyan  = permanent SLR extent
      🟠 Amber = storm-surge vulnerable halo
    No-data pixels (failed tile fetches, blank canvas fill, placeholder
    tiles of any color) are excluded from BOTH the water mask and the
    colorable land mask, so they're never painted and never used as a
    fake "water" anchor for the distance transform.
    """
    arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    H, W = arr.shape[:2]
    water = _water_mask(arr)

    if water.sum() < arr.size * 0.005 / 3:
        return img

    # No-data detection: metadata mask (fast path, set at fetch time) OR
    # content-based flatness (works even if metadata was dropped by a
    # save/reload/resize/JPEG-cache round trip in between).
    meta_no_data = img.info.get('no_data_mask')
    if meta_no_data is not None and meta_no_data.shape != water.shape:
        meta_no_data = None
    content_no_data = _detect_flat_nodata(arr)
    no_data = content_no_data if meta_no_data is None else (meta_no_data | content_no_data)

    # Small dilation (a few px) so blur doesn't leak flood color across
    # the no-data boundary. This does NOT need to be aggressive — the
    # flatness detector already finds the true extent of the blank
    # region; we're just softening its edge, not compensating for a
    # mask that's wrong to begin with.
    if no_data.any():
        dist_to_nodata = _distance_transform(~no_data)
        no_data = no_data | (dist_to_nodata <= 4)

    water = water & ~no_data
    land_all = ~water
    dist = _distance_transform(land_all)
    land = land_all & ~no_data

    reach_factor = 6.0 if scale == 'basin' else 1.0
    # /5.5 (was /12) — deltas are now computed against the model's own 2025
    # baseline, which made them smaller; without this rescale the flood zone
    # became nearly invisible and "before" and "after" looked identical.
    raw_perm_reach = slr_mm / 5.5 * reach_factor

    # Cap reach as a fraction of image size rather than a fixed pixel
    # count, so a large stitched mosaic can't get its city blanketed.
    min_dim = min(H, W)
    perm_reach = float(np.clip(raw_perm_reach, 5.0, 0.09 * min_dim))
    surge_mult = {'low': 1.8, 'medium': 2.6, 'high': 3.6, 'extreme': 4.8}.get(storm_risk, 3.0)
    surge_reach = float(np.clip(perm_reach * surge_mult, perm_reach, 0.22 * min_dim))

    perm_alpha = np.clip(1.0 - dist / perm_reach, 0.0, 1.0) * land
    surge_alpha = np.clip(1.0 - dist / surge_reach, 0.0, 1.0) * land
    surge_only = np.clip(surge_alpha - perm_alpha, 0.0, 1.0)

    # Exponent 0.8 (was 0.9) keeps the gradient but makes the interior of
    # the flooded zone noticeably more solid.
    perm_alpha = _gaussian_blur(perm_alpha, 1.6) ** 0.8
    surge_only = _gaussian_blur(surge_only, 2.6) ** 0.85

    perm_alpha[no_data] = 0.0
    surge_only[no_data] = 0.0

    CYAN = np.array([0.10, 0.58, 0.92], dtype=np.float32)
    AMBER = np.array([0.95, 0.60, 0.20], dtype=np.float32)
    A_S = surge_only * 0.62
    A_P = perm_alpha * 0.82

    out = arr.copy()
    for i in range(3):
        out[..., i] = out[..., i] * (1 - A_S) + AMBER[i] * A_S
        out[..., i] = out[..., i] * (1 - A_P) + CYAN[i] * A_P

    # Bright "new shoreline" contour along the outer edge of the permanent
    # flood zone. This single line is what makes the before/after slider
    # readable at a glance — a soft translucent tint alone is easy to miss.
    zone = (perm_alpha > 0.06).astype(np.float32)
    zone_soft = _gaussian_blur(zone, 1.4)
    edge = (zone_soft > 0.25) & (zone_soft < 0.75) & ~water & ~no_data
    if edge.any():
        EDGE = np.array([0.70, 0.95, 1.00], dtype=np.float32)
        A_E = _gaussian_blur(edge.astype(np.float32), 0.8) * 0.9
        for i in range(3):
            out[..., i] = out[..., i] * (1 - A_E) + EDGE[i] * A_E

    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out).filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=3))


# --------------------------------------------------------------------------- #
# 4) Optional Google Earth Engine
# --------------------------------------------------------------------------- #
def fetch_ee_composite(bbox, year=2024, size=1024):
    import os
    key_file = os.environ.get('EE_SERVICE_ACCOUNT_JSON')
    if not key_file or not Path(key_file).exists():
        return None
    try:
        import ee, json as _json
        with open(key_file) as f:
            creds = _json.load(f)
        credentials = ee.ServiceAccountCredentials(creds['client_email'], key_file)
        ee.Initialize(credentials)
        S, W_, N, E = bbox
        region = ee.Geometry.Rectangle([W_, S, E, N])
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filterDate(f'{year}-01-01', f'{year}-12-31')
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        img = col.median().select(['B4', 'B3', 'B2']).divide(3000)
        url = img.getThumbURL({'region': region, 'dimensions': size,
                                'format': 'jpg', 'min': 0, 'max': 1})
        r = requests.get(url, timeout=30, headers=HEADERS)
        if r.status_code == 200 and len(r.content) > 1000:
            return Image.open(BytesIO(r.content)).convert('RGB')
    except Exception as e:
        print(f'EE fetch failed: {e}')
    return None


if __name__ == '__main__':
    print('Building synthetic scene...')
    H = W = 512
    rng = np.random.default_rng(42)
    city = 0.22 + rng.uniform(-0.15, 0.15, (H, W, 3))
    tex = rng.normal(0, 0.08, (H, W))
    for c in range(3): city[..., c] += tex
    rm = np.zeros((H, W), dtype=bool)
    for i in range(H):
        j0 = int(i * 0.6) + 40; j1 = j0 + 90
        rm[i, max(0, j0):min(W, j1)] = True
    img_arr = city.copy(); img_arr[rm] = np.array([0.42, 0.38, 0.28])
    img_arr = np.clip(img_arr, 0, 1).astype(np.float32)
    pil = Image.fromarray((img_arr * 255).astype(np.uint8))
    w = _water_mask(img_arr)
    print(f'  city classified as water: {w[~rm].mean()*100:.1f}%   (want ~0%)')
    print(f'  river classified as water: {w[rm].mean()*100:.1f}%   (want ~100%)')
    out = apply_inundation(pil, slr_mm=400, storm_risk='extreme', scale='basin')
    out.save('synth_smoke.png')
    print('  saved synth_smoke.png')
