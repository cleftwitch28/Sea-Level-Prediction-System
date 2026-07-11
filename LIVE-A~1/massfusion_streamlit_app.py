"""
MASSFusion — Streamlit Sea-Level Rise Dashboard
================================================

Reads massfusion_meta.json + timelapse_cache.json from D:\\

Usage:
    pip install streamlit plotly pillow requests scipy
    cd D:\\
    streamlit run massfusion_streamlit_app.py --server.runOnSave=true

Then open http://localhost:8501.
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Load basin_imagery from the same folder as this script (D:\ or D:\venv\slp_project)
_HERE = Path(__file__).parent
for p in [_HERE, Path(r'D:\venv\slp_project'), Path(r'D:\\')]:
    if (p / 'basin_imagery.py').exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
        break

try:
    from basin_imagery import (fetch_basin_mosaic, apply_inundation,
                                 fetch_ee_composite)
    HAS_BASIN = True
except ImportError:
    HAS_BASIN = False

try:
    from v_doom7_inference import MASSFusionInference
    HAS_LIVE_MODEL = True
except ImportError:
    HAS_LIVE_MODEL = False

try:
    import ee
    HAS_EE_LIB = True
except ImportError:
    HAS_EE_LIB = False
# --------------------------------------------------------------------------- #
# 1) CONFIG
# --------------------------------------------------------------------------- #
def _find_data_dir():
    candidates = []
    if os.environ.get('MASSFUSION_DATA_DIR'):
        candidates.append(Path(os.environ['MASSFUSION_DATA_DIR']))
    # Normal case for a git-deployed app: data files sit next to the script
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path('.').resolve())
    # Windows local-dev convenience paths
    candidates += [Path('D:/'), Path('D:/venv'), Path('D:/venv/slp_project')]
    candidates.append(Path.home())

    for d in candidates:
        if (d / 'massfusion_meta.json').exists():
            return d
    return Path(__file__).resolve().parent


DATA_DIR = _find_data_dir()
TIMELAPSE_PATH = DATA_DIR / 'timelapse_cache.json'
META_PATH = DATA_DIR / 'massfusion_meta.json'
BASIN_CACHE_DIR = DATA_DIR / 'basin_imagery_cache'
BASIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CITY_CACHE_DIR = DATA_DIR / 'city_imagery_cache'
CITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CITY_FEATURE_CACHE_DIR = DATA_DIR / 'city_feature_cache'
CITY_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint location
CHECKPOINT_PATH = Path(os.environ.get('MASSFUSION_CHECKPOINT',
                                       DATA_DIR / 'v_doom7_soup_weighted.pt'))


def _init_earth_engine() -> bool:
    """Initializes Earth Engine using Streamlit Secrets or local auth.

    IMPORTANT: fails SILENTLY (console log only). Earth Engine is an
    optional extra — most visitors use cached predictions and ESRI
    imagery, and they should never see a scary red auth error for a
    feature they didn't ask for. The '⚙️ Advanced' expander is the one
    place that reports live-mode availability.
    """
    if not HAS_EE_LIB:
        print("DEBUG: Earth Engine library not installed")
        return False

    try:
        from google.oauth2.service_account import Credentials
        
        # 1. Look explicitly in Streamlit secrets first
        if "EE_SERVICE_ACCOUNT_JSON" in st.secrets:
            ee_json_str = st.secrets["EE_SERVICE_ACCOUNT_JSON"]
            
            # Parse the string into a dictionary
            creds_dict = json.loads(ee_json_str)
            
            # CRITICAL FIX: Define the explicit permissions (scopes) required
            SCOPES = [
                'https://www.googleapis.com/auth/earthengine',
                'https://www.googleapis.com/auth/cloud-platform'
            ]
            
            # Generate credentials AND attach the scopes
            credentials = Credentials.from_service_account_info(creds_dict).with_scopes(SCOPES)
            
            # Pass the project ID explicitly
            project_id = creds_dict.get("project_id")
            ee.Initialize(credentials=credentials, project=project_id)
            return True
            
        else:
            # Fallback for local dev if previously authenticated via CLI
            ee.Initialize()
            return True
            
    except Exception as e:
        # Console only — never an error box in the visitor-facing UI.
        print(f"DEBUG: Earth Engine auth failed: {e}")
        return False


@st.cache_resource(show_spinner=False)
def _ee_ready() -> bool:
    """Cache the auth attempt so we don't retry (slowly) on every rerun."""
    return _init_earth_engine()


HAS_CMEMS_CREDS = bool(os.environ.get('COPERNICUS_USER')) and bool(os.environ.get('COPERNICUS_PASS'))

# DEBUG PRINTS
print("DEBUG: DATA_DIR =", DATA_DIR)
print("DEBUG: HAS_BASIN =", HAS_BASIN)
print("DEBUG: HAS_LIVE_MODEL =", HAS_LIVE_MODEL)
print("DEBUG: HAS_CMEMS_CREDS =", HAS_CMEMS_CREDS)
print("DEBUG: CHECKPOINT_PATH exists =", CHECKPOINT_PATH.exists())
print("DEBUG: Basin cache dir exists:", BASIN_CACHE_DIR.exists())
print("DEBUG: Meta file exists:", META_PATH.exists())

st.set_page_config(page_title='MASSFusion — SLR Dashboard',
                   page_icon='🌊', layout='wide')


# --------------------------------------------------------------------------- #
# 2) DATA LOADERS
# --------------------------------------------------------------------------- #
@st.cache_data
def load_timelapse(_mtime: float):
    if not TIMELAPSE_PATH.exists():
        st.error(f'Timelapse file not found: {TIMELAPSE_PATH}')
        st.stop()
    return json.loads(TIMELAPSE_PATH.read_text())


@st.cache_data
def load_meta(_mtime: float):
    if not META_PATH.exists():
        st.error(f'Meta file not found: {META_PATH}')
        st.stop()
    return json.loads(META_PATH.read_text())


def _mt(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


TIMELAPSE = load_timelapse(_mt(TIMELAPSE_PATH))
META = load_meta(_mt(META_PATH))
CITIES = META['cities']
REGIONS = META['regions']
MODEL_INFO = META['model']


@st.cache_resource(show_spinner='🧠 Loading V_DOOM7 checkpoint…')
def load_live_model():
    """Loads the real model ONCE per app process."""
    if not HAS_LIVE_MODEL:
        print("DEBUG: v_doom7_inference not importable -- live model disabled")
        return None
    if not CHECKPOINT_PATH.exists():
        print(f"DEBUG: checkpoint not found at {CHECKPOINT_PATH} -- live model disabled")
        return None

    ee_ok = _ee_ready()
    if not ee_ok:
        print("DEBUG: Earth Engine not available -- live model disabled")
        return None
    if not HAS_CMEMS_CREDS:
        print("DEBUG: COPERNICUS_USER/COPERNICUS_PASS not set -- live model disabled")
        return None

    try:
        model = MASSFusionInference(
            CITIES, REGIONS,
            checkpoint=str(CHECKPOINT_PATH),
            live_feature_cache_dir=str(CITY_FEATURE_CACHE_DIR),
        )
        return model
    except Exception as e:
        print(f"DEBUG: MASSFusionInference failed to initialize: {e}")
        return None


LIVE_MODEL = load_live_model()
LIVE_INFERENCE_AVAILABLE = LIVE_MODEL is not None
print("DEBUG: LIVE_INFERENCE_AVAILABLE =", LIVE_INFERENCE_AVAILABLE)


# --------------------------------------------------------------------------- #
# 3) IMAGE HELPERS — city cache + live basin fetch + smooth flood overlay
# --------------------------------------------------------------------------- #
def decode_b64_image(data_uri: str) -> Image.Image:
    _, b64 = data_uri.split(',', 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')


def city_bbox(lat: float, lon: float, half_deg: float = 0.16):
    """Small bbox around a city, in [S, N, W, E] order."""
    import math
    lon_half = half_deg / max(0.15, math.cos(math.radians(lat)))
    return [lat - half_deg, lat + half_deg, lon - lon_half, lon + lon_half]


@st.cache_data(show_spinner='🛰️ Fetching city imagery…')
def city_base_image(entity_key: str) -> bytes | None:
    cache_file = CITY_CACHE_DIR / f'{entity_key.replace(" ", "_").replace("/", "-")}_base.jpg'
    if cache_file.exists() and cache_file.stat().st_size > 5000:
        return cache_file.read_bytes()

    if not HAS_BASIN or entity_key not in CITIES:
        return None

    c = CITIES[entity_key]
    bbox = city_bbox(c['lat'], c['lon'])
    img = fetch_basin_mosaic(bbox, zoom=13, max_tiles=64, with_labels=True)
    if img is None:
        return None

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=88)
    buf.seek(0)
    cache_file.write_bytes(buf.getvalue())
    return buf.getvalue()


def city_image(entity_key: str, year: int):
    raw = city_base_image(entity_key)
    if raw is None:
        return None
    return Image.open(io.BytesIO(raw)).convert('RGB')


@st.cache_data(show_spinner='🛰️ Fetching satellite mosaic…')
def basin_base_image(region_key: str, use_ee: bool = False) -> bytes | None:
    r = REGIONS[region_key]
    bbox = r['bbox']
    cache_file = BASIN_CACHE_DIR / f'{region_key}_base.jpg'
    
    if cache_file.exists() and cache_file.stat().st_size > 5000:
        print(f"DEBUG: Using cached {region_key} image")
        return cache_file.read_bytes()
    
    print(f"DEBUG: Fetching new mosaic for {region_key}, bbox={bbox}")
    if not HAS_BASIN:
        print("DEBUG: HAS_BASIN is False")
        return None
    
    img = None
    if use_ee:
        img = fetch_ee_composite(bbox, year=2024, size=1024)
        print("DEBUG: Tried GEE")
    
    if img is None:
        img = fetch_basin_mosaic(bbox, zoom=5, with_labels=True)
        print(f"DEBUG: Used ESRI, image size = {img.size if img else None}")
    
    if img is None:
        print("DEBUG: Both fetch methods failed!")
        return None
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=88)
    buf.seek(0)
    cache_file.write_bytes(buf.getvalue())
    print(f"DEBUG: Saved new cache for {region_key}")
    return buf.getvalue()

def basin_image(region_key: str, year: int, use_ee: bool = False):
    raw = basin_base_image(region_key, use_ee=use_ee)
    if raw is None:
        st.error("❌ Failed to fetch base image for basin")
        return None
    
    base = Image.open(io.BytesIO(raw)).convert('RGB')

    r = REGIONS[region_key]
    predicted_sla = get_prediction(region_key, year)
    delta = predicted_sla - get_baseline_prediction(region_key)
    
    flooded = apply_inundation(base, slr_mm=max(0, delta),
                               storm_risk=r.get('risk', 'high'), scale='basin')
    return flooded


def city_image_flooded(entity_key: str, year: int):
    base = city_image(entity_key, year)
    if base is None or not HAS_BASIN:
        return base
    c = CITIES[entity_key]
    r = REGIONS[c['region']]

    predicted_sla = get_prediction(entity_key, year)
    # Rise since the 2025 baseline (same prediction source), plus local
    # land subsidence — this is what the flood overlay should visualise.
    delta = (predicted_sla - get_baseline_prediction(entity_key)
             + c['subsidence_mm_yr'] * (year - 2025))

    return apply_inundation(base, slr_mm=max(0, delta),
                             storm_risk=c['storm_risk'], scale='city')


def get_prediction(entity_key: str, year: int) -> float:
    use_live = LIVE_INFERENCE_AVAILABLE and st.session_state.get('use_live_inference', False)
    if use_live:
        try:
            if entity_key in CITIES:
                return LIVE_MODEL.predict_city_live(entity_key, year)
            elif entity_key in REGIONS:
                return LIVE_MODEL.predict_regional(entity_key, year)
        except Exception as e:
            st.warning(f'⚠️ Live inference failed ({e}); showing cached prediction instead.')

    data = TIMELAPSE.get(entity_key) or TIMELAPSE.get('__region_' + entity_key)
    if not data or 'predictions' not in data:
        r = REGIONS.get(entity_key)
        if r:
            return float(r['current_sla_mm'] + r['predicted_slr_mm_yr'] * (year - 2025))
        return 0.0
    return float(data['predictions'].get(str(year), data['meta']['baseline_sla']))


def get_baseline_prediction(entity_key: str) -> float:
    """The model's own 2025 value, from the SAME source as the current
    prediction. All deltas must be computed against this — never against
    the region's `current_sla_mm`, which lives on a different reference
    and previously produced nonsense like 2015 being "+62 mm above" 2025."""
    return get_prediction(entity_key, 2025)


def year_mode(y: int):
    if y < 2025:  return 'HISTORICAL', '#8b949e'
    if y == 2025: return 'BASELINE', '#3fb950'
    return 'MODEL PROJECTION', '#f0883e'


def year_mode_friendly(y: int) -> str:
    if y < 2025:  return f'Viewing history ({y})'
    if y == 2025: return "Viewing today's baseline (2025)"
    return f'Viewing a future projection ({y})'


# --------------------------------------------------------------------------- #
# 4) SIDEBAR
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"# 🌊 MASSFusion")
    st.caption('Explore how sea levels around coastal cities are projected '
               'to change between 2015 and 2100.')
    if not HAS_BASIN:
        st.warning('⚠️ Imagery module not found — the flood overlay is disabled.')
    st.divider()

    col1, col2 = st.columns(2)
    if col1.button('🔄 Reload data', use_container_width=True,
                   help='Re-read the prediction data from disk'):
        st.cache_data.clear()
        st.rerun()
    if col2.button('🧹 Start over', use_container_width=True,
                   help='Reset all controls to their defaults'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.divider()

    entity_type = st.radio('What would you like to explore?',
                           ['City', 'Ocean basin'], horizontal=True)

    if entity_type == 'City':
        entity_key = st.selectbox('🏙️ Coastal city', sorted(CITIES.keys()),
                                    index=sorted(CITIES.keys()).index('Mumbai')
                                    if 'Mumbai' in CITIES else 0)
        c = CITIES[entity_key]
        r = REGIONS[c['region']]
        display_name = entity_key
    else:
        entity_key = st.selectbox('🌍 Ocean basin', sorted(REGIONS.keys()),
                                  format_func=lambda k: REGIONS[k]['name'])
        r = REGIONS[entity_key]
        c = {'lat': r['center'][0], 'lon': r['center'][1],
             'population_M': 0, 'storm_risk': r['risk'],
             'subsidence_mm_yr': 0}
        display_name = r['name']

    year = st.slider('🕓 Year', 2015, 2100, 2050, 1,
                     help='Years before 2025 show history. 2025 is "today" '
                          '(the baseline). Later years are the model\'s '
                          'projection of the future.')
    mode, mode_color = year_mode(year)
    st.markdown(f"<div style='padding:8px;background:#0d1117;color:{mode_color};"
                f"border-radius:4px;font-size:11px;letter-spacing:0.8px;text-align:center;'>"
                f"{year_mode_friendly(year)}</div>", unsafe_allow_html=True)

    st.divider()
    with st.expander('🧠 About the model', expanded=False):
        st.markdown(f"**Model:** {MODEL_INFO['name']}")
        st.caption(f"On average, its estimates land within about "
                   f"**{MODEL_INFO['mae_mm']:.0f} mm** of observed values "
                   f"(mean absolute error), and it explains "
                   f"**{MODEL_INFO['r2']*100:.0f}%** of the variation in the "
                   f"historical record (R² {MODEL_INFO['r2']:.2f}).")
        if MODEL_INFO.get('delta_h_mm') is not None:
            st.caption(f"Estimated human-driven component: "
                       f"**{MODEL_INFO['delta_h_mm']:+.2f} mm/yr**.")
        st.caption('Data sources: ERA5, AVISO, GRACE Mascon RL06.1 v04, '
                   'Sentinel-2, Sentinel-1, MODIS-Aqua, IBTrACS.')

    with st.expander('⚙️ Advanced', expanded=False):
        st.markdown('**Prediction source**')
        if LIVE_INFERENCE_AVAILABLE:
            st.checkbox(
                'Live model run (slower, uses fresh satellite data)',
                value=False,
                key='use_live_inference',
                help='Runs the real model on freshly-fetched satellite and '
                     'ocean data for CURRENT conditions, then extrapolates '
                     'to your selected year. The first request per city can '
                     'take several minutes. When off (default), you get '
                     'instant pre-computed projections. The two can differ: '
                     'the pre-computed curve is a fixed snapshot, while a '
                     'live run reflects conditions right now.'
            )
            if st.button('🗑️ Clear live-data cache'):
                import shutil
                shutil.rmtree(CITY_FEATURE_CACHE_DIR, ignore_errors=True)
                CITY_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                st.success('Cleared. The next live prediction will re-fetch everything.')
        else:
            st.checkbox('Live model run', value=False, disabled=True,
                        help='Unavailable — see reason below.')
            reasons = []
            if not HAS_LIVE_MODEL:
                reasons.append('model inference module not importable')
            elif not CHECKPOINT_PATH.exists():
                reasons.append('model checkpoint file not found')
            else:
                if not HAS_EE_LIB or not _ee_ready():
                    reasons.append('Earth Engine not connected (optional — '
                                   'needed only for live model runs)')
                if not HAS_CMEMS_CREDS:
                    reasons.append('Copernicus credentials not set')
            for reason in reasons:
                st.caption(f'⚠️ {reason}')

        st.markdown('**Imagery source**')
        use_ee = st.checkbox(
            'Use Google Earth Engine imagery',
            value=bool(os.environ.get('EE_SERVICE_ACCOUNT_JSON')),
            help='Requires an Earth Engine service account. Falls back to '
                 'ESRI World Imagery if unavailable.'
        )
        if st.button('🗑️ Clear basin image cache'):
            for f in BASIN_CACHE_DIR.glob('*.jpg'):
                f.unlink()
            st.cache_data.clear()
            st.success('Cleared. Reload the page.')
        st.caption(f'Data folder: `{DATA_DIR}`')


# --------------------------------------------------------------------------- #
# 5) MAIN
# --------------------------------------------------------------------------- #
predicted_sla = get_prediction(entity_key, year)
baseline_sla = get_baseline_prediction(entity_key)
delta = predicted_sla - baseline_sla
anthro_delta = r['anthro_contrib_mm'] * (year - 2025)

st.markdown(f"## {display_name} — {year}")
header_bits = []
if entity_type == 'City':
    header_bits.append(f"{r['name']} basin")
    header_bits.append(f"Population {c['population_M']} M")
header_bits.append(f"Storm risk **{c['storm_risk'].upper()}**")
header_bits.append(f"Risk score {r['risk_score']}/100")
st.caption(' • '.join(header_bits))

if LIVE_INFERENCE_AVAILABLE and st.session_state.get('use_live_inference', False):
    st.caption("🛰️ **Live estimate** — the model just ran on fresh satellite and "
               "ocean data, extrapolated to the selected year.")
else:
    st.caption("⚡ **Instant estimate** from pre-computed projections. "
               "Want a fresh model run? Open **⚙️ Advanced** in the sidebar.")


# --------------------------------------------------------------------------- #
# 6) IMAGE + STATS (BEFORE / AFTER SLIDER)
# --------------------------------------------------------------------------- #
try:
    from streamlit_image_comparison import image_comparison
    HAS_COMPARE = True
except ImportError:
    HAS_COMPARE = False

col_img, col_stats = st.columns([3, 2])

with col_img:
    # 1. Prepare the "Before" (Baseline) and "After" (Dynamic) images
    if entity_type == 'City':
        img_before = city_image(entity_key, 2025) # Fixed to baseline year
        img_after = city_image_flooded(entity_key, year)
    else:
        raw_bytes = basin_base_image(entity_key, use_ee=use_ee)
        img_before = Image.open(io.BytesIO(raw_bytes)).convert('RGB') if raw_bytes else None
        img_after = basin_image(entity_key, year, use_ee=use_ee)

    if img_after is None or img_before is None:
        st.info("Couldn't load the satellite imagery for this location. "
                "Check your internet connection, then press 🔄 Reload data "
                "in the sidebar.")
    elif not HAS_COMPARE:
        st.error("Missing library. Please stop the server and run: pip install streamlit-image-comparison")
        st.image(img_after, use_container_width=True, caption=f"Projected ({year})")
    else:
        # --- Dynamic top-right label based on the year ---
        if year < 2025:
            right_label = f"Historical ({year})"
        elif year == 2025:
            right_label = "Baseline (2025)"
        else:
            right_label = f"Model Projection ({year})"

        st.caption('↔️ **Drag the white divider** on the image to compare '
                   f'today (left) with {right_label.lower()} (right).')

        # 2. Render the interactive slider
        image_comparison(
            img1=img_before,
            img2=img_after,
            label1="Baseline (2025)",
            label2=right_label,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )

        # 3. Add the caption back underneath
        cap = (f'Sea level in {year}: {predicted_sla:+.0f} mm vs the long-term average  |  '
               f'🔵 Water at high tide  •  🟠 At risk during storms')
        if entity_type == 'Ocean basin':
            cap += f'  •  Imagery: {"Google Earth Engine (Sentinel-2)" if use_ee else "ESRI World Imagery"}'
        st.caption(cap)

with col_stats:
    # Sign-aware arrow + colour: red only when the sea is HIGHER than the
    # 2025 baseline, green when lower. Never a red ▲ on a negative number.
    d_arrow = '▲' if delta >= 0 else '▼'
    d_class = 'metric-red' if delta >= 0 else 'metric-green'
    horizon = year - 2025
    if horizon > 0:
        horizon_lbl = f'{horizon} years after 2025'
    elif horizon < 0:
        horizon_lbl = f'{-horizon} years before 2025'
    else:
        horizon_lbl = 'this is the baseline year'

    if year < 2025:
        sla_lbl = f'Sea level in {year} (historical)'
    elif year == 2025:
        sla_lbl = 'Sea level today (2025 baseline)'
    else:
        sla_lbl = f'Predicted sea level in {year}'

    st.markdown(f"""
<style>
.metric-card {{ background:#0d1117; padding:12px 16px; border-radius:6px;
                border-left:3px solid #58a6ff; margin-bottom:8px; }}
.metric-val {{ font-size:26px; font-weight:700; color:#58a6ff; }}
.metric-lbl {{ font-size:10px; color:#8b949e; letter-spacing:0.8px;
              text-transform:uppercase; }}
.metric-sub {{ font-size:11px; color:#8b949e; }}
.metric-red    {{ border-left-color:#f85149; }} .metric-red .metric-val {{ color:#f85149; }}
.metric-orange {{ border-left-color:#f0883e; }} .metric-orange .metric-val {{ color:#f0883e; }}
.metric-green  {{ border-left-color:#3fb950; }} .metric-green .metric-val {{ color:#3fb950; }}
</style>

<div class="metric-card">
  <div class="metric-val">{predicted_sla:.1f} mm</div>
  <div class="metric-lbl">{sla_lbl}</div>
  <div class="metric-sub">≈ {abs(predicted_sla)/10:.1f} cm {'above' if predicted_sla >= 0 else 'below'} the long-term average</div>
</div>
<div class="metric-card {d_class}">
  <div class="metric-val">{d_arrow} {delta:+.1f} mm</div>
  <div class="metric-lbl">Change vs 2025 baseline</div>
  <div class="metric-sub">{horizon_lbl} • ≈ {abs(delta)/10:.1f} cm {'higher' if delta >= 0 else 'lower'} than today</div>
</div>
<div class="metric-card metric-orange">
  <div class="metric-val">{anthro_delta:+.1f} mm</div>
  <div class="metric-lbl">Human-caused portion since 2025</div>
  <div class="metric-sub">from emissions, groundwater use, etc.</div>
</div>
<div class="metric-card metric-green">
  <div class="metric-val">{r['risk_score']} / 100</div>
  <div class="metric-lbl">Adaptation urgency</div>
  <div class="metric-sub">overall risk score for this region (does not change with the year)</div>
</div>
""", unsafe_allow_html=True)
    st.caption('mm = millimetres. 100 mm = 10 cm ≈ 4 inches.')
# --------------------------------------------------------------------------- #
# 7) TIMELINE + MITIGATIONS
# --------------------------------------------------------------------------- #
col_chart, col_mitig = st.columns([3, 2])

with col_chart:
    if HAS_PLOTLY:
        data = TIMELAPSE.get(entity_key) or TIMELAPSE.get('__region_' + entity_key)
        if data and 'predictions' in data:
            years = sorted(int(y) for y in data['predictions'].keys())
            preds = [data['predictions'][str(y)] for y in years]
        else:
            years = list(range(2015, 2101))
            preds = [r['current_sla_mm'] + r['predicted_slr_mm_yr'] * (y - 2025)
                     for y in years]
        anthros = [r['anthro_contrib_mm'] * (y - 2025) for y in years]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=preds, mode='lines',
                                  name='Total sea level (mm)',
                                  line=dict(color='#58a6ff', width=3),
                                  fill='tozeroy', fillcolor='rgba(88,166,255,0.1)'))
        fig.add_trace(go.Scatter(x=years, y=anthros, mode='lines',
                                  name='Human-caused (mm)',
                                  line=dict(color='#f0883e', width=2, dash='dot')))
        fig.add_vline(x=year, line_dash='dash', line_color='#f85149', line_width=2,
                      annotation_text=f'{year}', annotation_position='top right')
        fig.add_vline(x=2025, line_dash='dot', line_color='#3fb950', line_width=1)
        # In live mode the curve above is still the pre-computed snapshot;
        # mark the live model's estimate explicitly so the chart and the
        # metric cards never silently contradict each other.
        live_on = (LIVE_INFERENCE_AVAILABLE
                   and st.session_state.get('use_live_inference', False))
        if live_on:
            fig.add_trace(go.Scatter(
                x=[year], y=[predicted_sla], mode='markers',
                name=f'Live estimate ({year})',
                marker=dict(size=13, color='#ff7b72', symbol='diamond',
                            line=dict(width=1.5, color='#0d1117'))))
        fig.update_layout(
            title='Sea-level projection: 2015 → 2100',
            height=340, margin=dict(l=40, r=20, t=40, b=30),
            plot_bgcolor='#0d1117', paper_bgcolor='#161b22',
            font=dict(color='#c9d1d9', size=11),
            xaxis=dict(gridcolor='rgba(139,148,158,0.15)', title=''),
            yaxis=dict(gridcolor='rgba(139,148,158,0.15)', title='mm'),
            legend=dict(orientation='h', yanchor='top', y=1.12, x=0.5,
                        xanchor='center', bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig, use_container_width=True)
        if live_on:
            st.caption('The blue curve is the pre-computed projection; '
                       f'the ◆ diamond is the live model\'s estimate for {year}. '
                       'They can differ because the live run uses current '
                       'ocean conditions.')
    else:
        st.info('`pip install plotly` for the timeline chart')

with col_mitig:
    st.markdown('### 🛡️ Regional Mitigations')
    for m in r.get('mitigations', []):
        st.markdown(f'- {m}')
    st.divider()
    st.markdown('### 📊 Details')
    if entity_type == 'City':
        st.markdown(f"""
- **Land sinking** (subsidence, measured by satellite radar): {c['subsidence_mm_yr']:.1f} mm/yr
- **Regional sea-level trend**: rising {r['predicted_slr_mm_yr']:.1f} mm/yr
- **Coordinates**: {c['lat']:.3f}°, {c['lon']:.3f}°
- **Ocean region**: {r['name']}
""")
    else:
        st.markdown(f"""
- **Regional sea-level trend**: rising {r['predicted_slr_mm_yr']:.1f} mm/yr
- **Center of region**: {r['center'][0]:.1f}°, {r['center'][1]:.1f}°
""")


# --------------------------------------------------------------------------- #
# 8) WORLD MAP
# --------------------------------------------------------------------------- #
with st.expander('🗺️ World map — all cities', expanded=False):
    if HAS_PLOTLY:
        risk_col_map = {'low': '#3fb950', 'medium': '#f0883e',
                        'high': '#f85149', 'extreme': '#8957e5'}

        fig_map = go.Figure()
        # One trace per risk level so the map gets a real, clickable legend
        # instead of unexplained colored dots.
        for risk_level in ['low', 'medium', 'high', 'extreme']:
            ks = [k for k in CITIES if CITIES[k]['storm_risk'] == risk_level]
            if not ks:
                continue
            fig_map.add_trace(go.Scattergeo(
                lon=[CITIES[k]['lon'] for k in ks],
                lat=[CITIES[k]['lat'] for k in ks],
                text=ks,
                hovertext=[f"<b>{k}</b><br>{CITIES[k]['population_M']}M people • "
                           f"Storm risk: {risk_level.upper()}"
                           for k in ks],
                hoverinfo='text', mode='markers',
                name=f'{risk_level.title()} storm risk',
                marker=dict(
                    size=[min(6 + CITIES[k]['population_M']**0.5 * 3, 22) for k in ks],
                    color=risk_col_map[risk_level], opacity=0.85,
                    line=dict(width=1, color='#0d1117')),
            ))
        fig_map.update_geos(
            projection_type='natural earth',
            showcoastlines=True, coastlinecolor='#30363d',
            showland=True, landcolor='#161b22',
            showocean=True, oceancolor='#0d1117',
            showcountries=True, countrycolor='#30363d', bgcolor='#0d1117',
        )
        fig_map.update_layout(
            height=500, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='#0d1117',
            legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0.5,
                        xanchor='center', bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#c9d1d9')),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption('Dot size = population • Dot color = storm risk • '
                   'Hover over a dot for details. To view a city, pick it '
                   'from the sidebar.')


st.markdown('---')
st.caption(f"MASSFusion • model: {MODEL_INFO['name']} • "
           f"{len(CITIES)} cities • {len(REGIONS)} ocean basins")
