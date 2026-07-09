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
    """Initializes Earth Engine using Streamlit Secrets or local auth."""
    if not HAS_EE_LIB:
        st.error("Earth Engine library not installed.")
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
        # Show the actual error in the Streamlit UI
        st.error(f"🌍 Earth Engine Authentication Error: {str(e)}")
        return False


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

    ee_ok = _init_earth_engine()
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
    st.success(f"✅ Base image loaded: {base.size[0]}×{base.size[1]} px")
    
    r = REGIONS[region_key]
    predicted_sla = get_prediction(region_key, year)
    delta = predicted_sla - r['current_sla_mm']
    
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
    slr_mm = predicted_sla + c['subsidence_mm_yr'] * (year - 2025)
    delta = slr_mm - r['current_sla_mm']
    
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


def year_mode(y: int):
    if y < 2025:  return 'HISTORICAL', '#8b949e'
    if y == 2025: return 'BASELINE', '#3fb950'
    return 'MODEL PROJECTION', '#f0883e'


# --------------------------------------------------------------------------- #
# 4) SIDEBAR
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"# 🌊 MASSFusion")
    st.caption(f"**Model:** {MODEL_INFO['name']}")
    st.caption(f"MAE **{MODEL_INFO['mae_mm']:.2f} mm** • R² **{MODEL_INFO['r2']:.2f}**")
    if MODEL_INFO.get('delta_h_mm') is not None:
        st.caption(f"Δh **+{MODEL_INFO['delta_h_mm']:.2f} mm** (anthropogenic recovery)")
    st.caption(f"📂 Data: `{DATA_DIR}`")
    if not HAS_BASIN:
        st.warning('⚠️ basin_imagery.py not found — inundation overlay disabled.')
    st.divider()

    col1, col2 = st.columns(2)
    if col1.button('🔄 Reload data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    if col2.button('🧹 Reset UI', use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.divider()

    entity_type = st.radio('View', ['City', 'Ocean basin'], horizontal=True)

    if entity_type == 'City':
        entity_key = st.selectbox('🏙️ Coastal city', sorted(CITIES.keys()),
                                    index=sorted(CITIES.keys()).index('Mumbai')
                                    if 'Mumbai' in CITIES else 0)
        c = CITIES[entity_key]
        r = REGIONS[c['region']]
    else:
        entity_key = st.selectbox('🌍 Ocean basin', sorted(REGIONS.keys()))
        r = REGIONS[entity_key]
        c = {'lat': r['center'][0], 'lon': r['center'][1],
             'population_M': 0, 'storm_risk': r['risk'],
             'subsidence_mm_yr': 0}

    year = st.slider('🕓 Projection year', 2015, 2100, 2050, 1,
                     help="2015-2024: historical • 2025: baseline • 2026-2100: V_DOOM7 projection")
    mode, mode_color = year_mode(year)
    st.markdown(f"<div style='padding:8px;background:#0d1117;color:{mode_color};"
                f"border-radius:4px;font-size:11px;letter-spacing:0.8px;text-align:center;'>"
                f"● {mode}</div>", unsafe_allow_html=True)

    st.divider()
    with st.expander('🧠 Prediction source', expanded=False):
        if LIVE_INFERENCE_AVAILABLE:
            st.checkbox(
                'Use live model inference',
                value=False,
                key='use_live_inference',
                help='Runs the real V_DOOM7 checkpoint on freshly-fetched '
                     'GRACE/Sentinel/MODIS/CMEMS/ERA5 data for the CURRENT '
                     'conditions, then extrapolates to your selected year. '
                     'First request per city takes several minutes (real '
                     'satellite/ocean data fetches); cached to disk after that.'
            )
            if st.button('🗑️ Clear feature cache (force live re-fetch)'):
                import shutil
                shutil.rmtree(CITY_FEATURE_CACHE_DIR, ignore_errors=True)
                CITY_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                st.success('Cleared. Next live prediction will re-fetch everything.')
        else:
            st.checkbox('Use live model inference', value=False, disabled=True,
                        help='Unavailable — see reason below.')
            reasons = []
            if not HAS_LIVE_MODEL:
                reasons.append('`v_doom7_inference.py` not importable')
            elif not CHECKPOINT_PATH.exists():
                reasons.append(f'checkpoint not found at `{CHECKPOINT_PATH}`')
            else:
                if not HAS_EE_LIB or not _init_earth_engine():
                    reasons.append('Earth Engine not authenticated')
                if not HAS_CMEMS_CREDS:
                    reasons.append('COPERNICUS_USER/COPERNICUS_PASS not set')
            for reason in reasons:
                st.caption(f'⚠️ {reason}')
        st.caption('When off (default), predictions come from '
                   '`timelapse_cache.json` — instant, but a fixed snapshot.')

    st.divider()
    with st.expander('🛰️ Imagery source'):
        use_ee = st.checkbox(
            'Use Google Earth Engine',
            value=bool(os.environ.get('EE_SERVICE_ACCOUNT_JSON')),
            help='Requires `pip install earthengine-api` and '
                 'EE_SERVICE_ACCOUNT_JSON env-var pointing to your service key. '
                 'Falls back to ESRI World Imagery if unavailable.'
        )
        if st.button('🗑️ Clear basin image cache'):
            for f in BASIN_CACHE_DIR.glob('*.jpg'):
                f.unlink()
            st.cache_data.clear()
            st.success('Cleared. Reload the page.')

    st.divider()
    st.caption('Data: ERA5, AVISO, GRACE Mascon RL06.1 v04, Sentinel-2, '
               'Sentinel-1, MODIS-Aqua, IBTrACS.')


# --------------------------------------------------------------------------- #
# 5) MAIN
# --------------------------------------------------------------------------- #
predicted_sla = get_prediction(entity_key, year)
baseline_sla = r['current_sla_mm']
delta = predicted_sla - baseline_sla
anthro_delta = r['anthro_contrib_mm'] * (year - 2025)

st.markdown(f"## {entity_key} — {year}")
st.caption(f"{r['name']} basin • Population {c.get('population_M', '?')} M • "
           f"Storm risk **{c['storm_risk'].upper()}** • Risk score {r['risk_score']}/100")

if LIVE_INFERENCE_AVAILABLE and st.session_state.get('use_live_inference', False):
    st.caption("🧠 **Live model inference** — real V_DOOM7 forward pass on freshly-fetched "
               "current satellite/ocean data, extrapolated to the selected year.")
else:
    st.caption("📦 Cached prediction from `timelapse_cache.json` — "
               "toggle live inference in the sidebar for a real-time model run.")


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
        st.info('Could not load imagery for this entity.')
    elif not HAS_COMPARE:
        st.error("Missing library. Please stop the server and run: pip install streamlit-image-comparison")
        st.image(img_after, use_container_width=True, caption=f"Projected ({year})")
    else:
        # --- NEW: Dynamic top-right label based on the year ---
        if year < 2025:
            right_label = f"Historical ({year})"
        elif year == 2025:
            right_label = "Baseline (2025)"
        else:
            right_label = f"Model Projections ({year})"

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
        cap = (f'{mode} • {year} • +{predicted_sla:.0f} mm SLA  |  '
               f'🔵 Permanent SLR  •  🟠 Storm-surge vulnerable zone')
        if entity_type == 'Ocean basin':
            cap += f'  •  Source: {"Google Earth Engine (Sentinel-2)" if use_ee else "ESRI World Imagery"}'
        st.caption(cap)

with col_stats:
    st.markdown(f"""
<style>
.metric-card {{ background:#0d1117; padding:12px 16px; border-radius:6px;
                border-left:3px solid #58a6ff; margin-bottom:8px; }}
.metric-val {{ font-size:26px; font-weight:700; color:#58a6ff; }}
.metric-lbl {{ font-size:10px; color:#8b949e; letter-spacing:0.8px;
              text-transform:uppercase; }}
.metric-red    {{ border-left-color:#f85149; }} .metric-red .metric-val {{ color:#f85149; }}
.metric-orange {{ border-left-color:#f0883e; }} .metric-orange .metric-val {{ color:#f0883e; }}
.metric-green  {{ border-left-color:#3fb950; }} .metric-green .metric-val {{ color:#3fb950; }}
</style>

<div class="metric-card">
  <div class="metric-val">{predicted_sla:.1f} mm</div>
  <div class="metric-lbl">Predicted SLA ({year})</div>
</div>
<div class="metric-card metric-red">
  <div class="metric-val">▲ {delta:+.1f} mm</div>
  <div class="metric-lbl">Δ vs 2025 baseline ({year - 2025} yr horizon)</div>
</div>
<div class="metric-card metric-orange">
  <div class="metric-val">{anthro_delta:.1f} mm</div>
  <div class="metric-lbl">Anthropogenic ΔMAE</div>
</div>
<div class="metric-card metric-green">
  <div class="metric-val">{r['risk_score']} / 100</div>
  <div class="metric-lbl">Adaptation Urgency</div>
</div>
""", unsafe_allow_html=True)
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
                                  name='Total SLA (mm)',
                                  line=dict(color='#58a6ff', width=3),
                                  fill='tozeroy', fillcolor='rgba(88,166,255,0.1)'))
        fig.add_trace(go.Scatter(x=years, y=anthros, mode='lines',
                                  name='Anthropogenic (mm)',
                                  line=dict(color='#f0883e', width=2, dash='dot')))
        fig.add_vline(x=year, line_dash='dash', line_color='#f85149', line_width=2,
                      annotation_text=f'{year}', annotation_position='top right')
        fig.add_vline(x=2025, line_dash='dot', line_color='#3fb950', line_width=1)
        fig.update_layout(
            title='SLA projection: 2015 → 2100',
            height=340, margin=dict(l=40, r=20, t=40, b=30),
            plot_bgcolor='#0d1117', paper_bgcolor='#161b22',
            font=dict(color='#c9d1d9', size=11),
            xaxis=dict(gridcolor='rgba(139,148,158,0.15)', title=''),
            yaxis=dict(gridcolor='rgba(139,148,158,0.15)', title='mm'),
            legend=dict(orientation='h', yanchor='top', y=1.12, x=0.5,
                        xanchor='center', bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig, use_container_width=True)
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
- **Subsidence** (InSAR): {c['subsidence_mm_yr']:.1f} mm/yr
- **Regional trend**: {r['predicted_slr_mm_yr']:.1f} mm/yr
- **Coordinates**: {c['lat']:.3f}°, {c['lon']:.3f}°
- **Region**: {r['name']}
""")
    else:
        st.markdown(f"""
- **Regional trend**: {r['predicted_slr_mm_yr']:.1f} mm/yr
- **BBox**: {r['bbox']}
- **Center**: {r['center'][0]:.1f}°, {r['center'][1]:.1f}°
""")


# --------------------------------------------------------------------------- #
# 8) WORLD MAP
# --------------------------------------------------------------------------- #
with st.expander('🗺️ World map — all cities', expanded=False):
    if HAS_PLOTLY:
        lats = [CITIES[k]['lat'] for k in CITIES]
        lons = [CITIES[k]['lon'] for k in CITIES]
        names = list(CITIES.keys())
        risks = [CITIES[k]['storm_risk'] for k in CITIES]
        pops = [CITIES[k]['population_M'] for k in CITIES]
        risk_col_map = {'low': '#3fb950', 'medium': '#f0883e',
                        'high': '#f85149', 'extreme': '#8957e5'}
        colors = [risk_col_map.get(rk, '#8b949e') for rk in risks]
        sizes = [min(6 + p**0.5 * 3, 22) for p in pops]

        fig_map = go.Figure(go.Scattergeo(
            lon=lons, lat=lats, text=names,
            hovertext=[f'<b>{n}</b><br>{p}M pop • Storm risk: {rk.upper()}'
                       for n, p, rk in zip(names, pops, risks)],
            hoverinfo='text', mode='markers',
            marker=dict(size=sizes, color=colors, opacity=0.85,
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
            height=500, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#0d1117',
        )
        st.plotly_chart(fig_map, use_container_width=True)


st.markdown('---')
st.caption(f"MASSFusion V_DOOM7 Soup • MAE {MODEL_INFO['mae_mm']:.2f} mm • "
           f"R² {MODEL_INFO['r2']:.2f} • {len(CITIES)} cities × {len(REGIONS)} basins • "
           f"{len(TIMELAPSE)} timelapse entries")
