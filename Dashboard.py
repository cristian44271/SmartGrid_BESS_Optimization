import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linprog

st.set_page_config(page_title="Smart Grid Project Dashboard", layout="wide")

import requests
import gridstatus
import datetime
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import os
import glob

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'data_valid' not in st.session_state:
    st.session_state.data_valid = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'lp_run' not in st.session_state:
    st.session_state.lp_run = False
if 'active_train_end' not in st.session_state:
    st.session_state.active_train_end = None
if 'active_test_start' not in st.session_state:
    st.session_state.active_test_start = None
if 'active_test_end' not in st.session_state:
    st.session_state.active_test_end = None
if 'active_demand_charge' not in st.session_state:
    st.session_state.active_demand_charge = 50.0
if 'map_lat' not in st.session_state:
    st.session_state.map_lat = 44.56
if 'map_lon' not in st.session_state:
    st.session_state.map_lon = -123.26

def on_load_click():
    st.session_state.data_loaded = True
    st.session_state.model_trained = False
    st.session_state.lp_run = False

# ============================================================
# 1. DATA LOADING
# ============================================================
@st.cache_data
def load_and_align_data(uploaded_bytes=None, dataset_path=None, lat=44.56, lon=-123.26):
    # Load user-uploaded load or sample dataset
    if uploaded_bytes is not None:
        raw_df = pd.read_csv(uploaded_bytes)
    elif dataset_path is not None:
        raw_df = pd.read_csv(dataset_path)
    else:
        # Fallback if somehow neither is provided
        raw_df = pd.read_csv('datasets/Kelley Engineering Center Net Energy Usage (kWh).csv')

    raw_df['clean_time'] = raw_df.iloc[:, 0].astype(str).str.split(' GMT').str[0]
    raw_df['Time'] = pd.to_datetime(raw_df['clean_time'],
                                    format='%a %b %d %Y %H:%M:%S',
                                    errors='coerce')
    mask_nat = raw_df['Time'].isna()
    if mask_nat.any():
        raw_df.loc[mask_nat, 'Time'] = pd.to_datetime(
            raw_df.loc[mask_nat].iloc[:, 0], errors='coerce')
    raw_df['Load_kWh'] = pd.to_numeric(raw_df.iloc[:, 1], errors='coerce').fillna(0)
    raw_df['Load_kW'] = raw_df['Load_kWh'] * 4
    raw_df = raw_df.dropna(subset=['Time']).set_index('Time').sort_index()

    # Solar & price
    d_start = raw_df.index.min().strftime('%Y-%m-%d')
    d_end = raw_df.index.max().strftime('%Y-%m-%d')
    
    # 1. Fetch Solar
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d_start}&end_date={d_end}&hourly=shortwave_radiation&timezone=America%2FLos_Angeles"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        solar_df = pd.DataFrame({
            'Time': pd.to_datetime(data['hourly']['time']),
            'GHI_Wm2': data['hourly']['shortwave_radiation']
        }).set_index('Time').sort_index()
    else:
        url_f = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={d_start}&end_date={d_end}&hourly=shortwave_radiation&timezone=America%2FLos_Angeles"
        resp_f = requests.get(url_f)
        if resp_f.status_code == 200:
            data_f = resp_f.json()
            solar_df = pd.DataFrame({
                'Time': pd.to_datetime(data_f['hourly']['time']),
                'GHI_Wm2': data_f['hourly']['shortwave_radiation']
            }).set_index('Time').sort_index()
        else:
            solar_df = pd.DataFrame()

    # 2. Fetch CAISO
    iso = gridstatus.CAISO()
    end_iso = (raw_df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        price_raw = iso.get_lmp(date=d_start, end=end_iso, market="DAY_AHEAD_HOURLY", locations=["TH_NP15_GEN-APND"])
        price_raw['Time'] = pd.to_datetime(price_raw['Time']).dt.tz_localize(None)
        price_raw['Price_USD_per_kWh'] = price_raw['LMP'] / 1000.0
        price_df = price_raw.set_index('Time').sort_index()
    except Exception:
        price_df = pd.DataFrame()

    sim_start = raw_df.index.min()
    sim_end = raw_df.index.max()

    time_grid = pd.date_range(start=sim_start, end=sim_end, freq='15min')
    df = pd.DataFrame(index=time_grid)

    df['Load_kW']  = raw_df[['Load_kW']].reindex(df.index).interpolate(method='time')['Load_kW'].clip(lower=0).fillna(0)
    if not solar_df.empty:
        df['GHI_Wm2']  = solar_df[['GHI_Wm2']].reindex(df.index).interpolate(method='time')['GHI_Wm2'].clip(lower=0).fillna(0)
    else:
        df['GHI_Wm2']  = 0.0
    
    if not price_df.empty:
        df['Price']    = price_df[['Price_USD_per_kWh']].reindex(df.index).ffill().bfill()['Price_USD_per_kWh']
    else:
        df['Price']    = 0.0

    # Feature Engineering
    df['HourOfDay']    = df.index.hour + df.index.minute / 60.0
    df['DayOfWeek']    = df.index.dayofweek
    df['Load_Lag24h']  = df['Load_kW'].shift(96)
    df['Load_Lag7d']   = df['Load_kW'].shift(96 * 7)
    df_ml = df.dropna().copy()
    
    return df_ml

# ============================================================
# 2. ML TRAINING
# ============================================================
@st.cache_data
def train_ml_model(df_ml, train_end_str):
    train_boundary = pd.Timestamp(train_end_str) + pd.Timedelta(hours=23, minutes=45)
    train_mask = df_ml.index <= train_boundary
    df_train = df_ml[train_mask]
    
    features = ['HourOfDay', 'DayOfWeek', 'GHI_Wm2', 'Load_Lag24h', 'Load_Lag7d']
    X_train = df_train[features]
    y_train = df_train['Load_kW']

    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    if not X_train.empty:
        rf_model.fit(X_train, y_train)
    
    peak_demand_train = y_train.max() if not y_train.empty else 0
    return rf_model, features, peak_demand_train

# ============================================================
# 3. LP OPTIMIZATION
# ============================================================
@st.cache_data
def run_lp_optimization(df_ml, _rf_model, features, test_start_str, test_end_str, demand_charge_per_mw):
    battery_capacity   = 400.0   # kWh
    max_power          = 100.0   # kW
    efficiency         = 0.95
    dt_hours           = 0.25
    lp_demand_coeff    = demand_charge_per_mw / 1000.0  # $/kW

    train_boundary = pd.Timestamp(test_start_str) - pd.Timedelta(minutes=15)
    test_mask = (df_ml.index > train_boundary) & (df_ml.index <= pd.Timestamp(test_end_str) + pd.Timedelta(hours=23, minutes=45))
    df_test = df_ml[test_mask].copy()

    N_test = len(df_test)
    if N_test == 0:
        return df_test, battery_capacity, max_power, efficiency, dt_hours

    loads_actual = df_test['Load_kW'].values
    prices       = df_test['Price'].values

    Load_S1 = loads_actual.copy()
    Load_S3 = np.zeros(N_test)
    SOC_S3  = np.zeros(N_test)
    soc_current_S3 = 0.0

    all_preds = np.maximum(_rf_model.predict(df_test[features]), 0)
    num_days  = N_test // 96

    for d in range(num_days):
        si = d * 96
        ei = (d + 1) * 96

        pred_load   = all_preds[si:ei]
        today_price = prices[si:ei]

        c = np.zeros(289)
        c[0:96]   =  today_price * dt_hours
        c[96:192] = -today_price * dt_hours
        c[288]    =  lp_demand_coeff

        bounds = ([(0, max_power)            for _ in range(96)] +
                  [(0, min(max_power, pl))   for pl in pred_load] +
                  [(0, battery_capacity)     for _ in range(96)] +
                  [(0, None)])

        A_ub = np.zeros((96, 289))
        b_ub = -pred_load.copy()
        for i in range(96):
            A_ub[i, i]      =  1
            A_ub[i, 96+i]   = -1
            A_ub[i, 288]    = -1

        A_eq = np.zeros((96, 289))
        b_eq = np.zeros(96)
        for i in range(96):
            A_eq[i, 192+i] =  1
            A_eq[i, i]     = -efficiency * dt_hours
            A_eq[i, 96+i]  =  dt_hours / efficiency
            if i > 0:
                A_eq[i, 192+i-1] = -1
            else:
                b_eq[i] = soc_current_S3

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

        if res.success:
            P_ch_opt  = res.x[0:96]
            P_dis_opt = res.x[96:192]
            target_pk = res.x[288]
        else:
            P_ch_opt  = np.zeros(96)
            P_dis_opt = np.zeros(96)
            target_pk = np.inf

        for i in range(96):
            gi = si + i
            if gi >= N_test: break
            cl = loads_actual[gi]
            p_ch  = P_ch_opt[i]
            p_dis = P_dis_opt[i]

            p_dis = max(p_dis, max(0, cl - target_pk))
            if cl + p_ch - p_dis > target_pk:
                p_ch = max(0, target_pk - cl + p_dis)
            if p_dis > cl:
                p_dis = cl; p_ch = 0

            p_dis = min(p_dis, max_power, soc_current_S3 * efficiency / dt_hours)
            p_ch  = min(p_ch,  max_power, (battery_capacity - soc_current_S3) / (efficiency * dt_hours))

            soc_current_S3 += p_ch * efficiency * dt_hours - p_dis * dt_hours / efficiency
            SOC_S3[gi]  = soc_current_S3
            Load_S3[gi] = cl + p_ch - p_dis

    df_test['Load_S1']        = Load_S1
    df_test['Load_S3']        = Load_S3
    df_test['SOC_S3']         = SOC_S3
    df_test['Predicted_Load'] = all_preds

    return df_test, battery_capacity, max_power, efficiency, dt_hours

# ============================================================
# FAST: Rule-based S2
# ============================================================
def run_s2_simulation(loads_actual, times, discharge_threshold_kw, charge_threshold_kw,
                      battery_capacity, max_power, efficiency, dt_hours):
    N = len(loads_actual)
    Load_S2 = np.zeros(N)
    SOC_S2  = np.zeros(N)
    soc = 0.0

    for idx in range(N):
        cl = loads_actual[idx]
        hr = times[idx].hour
        p_ch = 0.0; p_dis = 0.0

        if hr < 6 and soc < battery_capacity:
            p_ch = min(max_power, (battery_capacity - soc) / (efficiency * dt_hours))
            if cl < charge_threshold_kw:
                headroom = charge_threshold_kw - cl
                p_ch = min(p_ch, headroom)
            else:
                p_ch = 0.0

        if cl > discharge_threshold_kw and soc > 0:
            req = cl - discharge_threshold_kw
            p_dis = min(max_power, soc * efficiency / dt_hours, req, cl)

        soc += p_ch * efficiency * dt_hours - p_dis * dt_hours / efficiency
        SOC_S2[idx]  = soc
        Load_S2[idx] = cl + p_ch - p_dis

    return Load_S2, SOC_S2

# ============================================================
# COST HELPER
# ============================================================
def calc_cost_breakdown(load_series, price_series, dt_hours, demand_charge_per_mw):
    load = np.asarray(load_series, dtype=float)
    price = np.asarray(price_series, dtype=float)

    energy_cost = np.sum(load * price) * dt_hours          # $
    total_kwh   = np.sum(load) * dt_hours                  # kWh
    peak_kw     = load.max() if len(load)>0 else 0         # kW
    peak_mw     = peak_kw / 1000.0

    if hasattr(load_series, 'index'):
        n_days = max(1, len(set(load_series.index.date)))
    else:
        n_days = max(1, len(load) // 96)

    demand_cost = demand_charge_per_mw * peak_mw * n_days  # $

    return {
        'energy_cost': energy_cost,
        'peak_kw':     peak_kw,
        'demand_cost': demand_cost,
        'total_kwh':   total_kwh,
        'total_cost':  energy_cost + demand_cost,
        'n_days':      n_days,
    }

# ============================================================
# SIDEBAR UI
# ============================================================
st.sidebar.header("📂 0. Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Custom Load CSV (Optional)",
     type=["csv"],
    help="Two columns: Timestamp (col 1) and Load in kWh per 15 min (col 2)."
)

sample_files = glob.glob("datasets/*.csv")
sample_options = [os.path.basename(f) for f in sample_files]
default_idx = 0
for i, f in enumerate(sample_options):
    if "Kelley" in f:
        default_idx = i

if len(sample_options) > 0:
    selected_sample = st.sidebar.selectbox("Or choose a Sample Dataset from GitHub:", options=sample_options, index=default_idx)
else:
    selected_sample = None

st.sidebar.markdown("---")
st.sidebar.header("📍 1. Location (For Solar API)")
st.sidebar.markdown("Search for a city or click on the map to set your location.")

search_col1, search_col2 = st.sidebar.columns([3, 1])
city_query = search_col1.text_input("Search City", value="", placeholder="Seattle, WA", label_visibility="collapsed")
if search_col2.button("🔍", help="Search for coordinates"):
    if city_query:
        geolocator = Nominatim(user_agent="smart_grid_dashboard")
        location = geolocator.geocode(city_query)
        if location:
            st.session_state.map_lat = location.latitude
            st.session_state.map_lon = location.longitude
        else:
            st.sidebar.error("City not found!")

m = folium.Map(location=[st.session_state.map_lat, st.session_state.map_lon], zoom_start=10)
folium.Marker([st.session_state.map_lat, st.session_state.map_lon], tooltip="Selected Location").add_to(m)
map_data = st_folium(m, height=250, use_container_width=True)

if map_data and map_data.get("last_clicked"):
    new_lat = map_data["last_clicked"]["lat"]
    new_lon = map_data["last_clicked"]["lng"]
    if new_lat != st.session_state.map_lat or new_lon != st.session_state.map_lon:
        st.session_state.map_lat = new_lat
        st.session_state.map_lon = new_lon
        st.rerun()

st.sidebar.caption(f"Selected Coordinates: **{st.session_state.map_lat:.4f}, {st.session_state.map_lon:.4f}**")
user_lat = st.session_state.map_lat
user_lon = st.session_state.map_lon

st.sidebar.button("📥 Load Data Source", type="primary", on_click=on_load_click)

# LOAD DATA TRIGGER
if st.session_state.data_loaded:
    with st.spinner("Loading and aligning data..."):
        try:
            if uploaded_file is not None:
                df_ml = load_and_align_data(uploaded_bytes=uploaded_file, dataset_path=None, lat=user_lat, lon=user_lon)
            else:
                d_path = os.path.join("datasets", selected_sample) if selected_sample else None
                df_ml = load_and_align_data(uploaded_bytes=None, dataset_path=d_path, lat=user_lat, lon=user_lon)
            if not df_ml.empty:
                st.session_state.data_valid = True
                data_min = df_ml.index.min().date()
                data_max = df_ml.index.max().date()
            else:
                st.session_state.data_valid = False
        except Exception as e:
            st.session_state.data_valid = False
            st.sidebar.error(f"Error loading data: {e}")

    if not st.session_state.data_valid:
        st.title("ECE 537 Smart Grid — Peak Shaving & ML Optimization")
        st.error("Data could not be loaded. Please ensure the uploaded file has valid timestamps and data.")
        st.stop()
else:
    st.title("ECE 537 Smart Grid — Peak Shaving & ML Optimization")
    st.info("👈 Upload your data (or use defaults) and click **Load Data Source** to begin.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("📅 2. Select the Train / Test Window")
default_train_end = data_min + (data_max - data_min) * 2 // 3

col_t1, col_t2 = st.sidebar.columns(2)
train_start = col_t1.date_input("Train start", value=data_min,
                                min_value=data_min, max_value=data_max)
train_end   = col_t2.date_input("Train end", value=default_train_end,
                                min_value=train_start, max_value=data_max)

if st.sidebar.button("⚙️ Train ML Model", type="primary"):
    st.session_state.model_trained = True
    st.session_state.lp_run = False
    st.session_state.active_train_end = train_end

if st.session_state.model_trained and st.session_state.active_train_end != train_end:
    st.sidebar.warning("⚠️ Train dates changed. Click Train ML Model to update.")

safe_test_min = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).date()
if safe_test_min > data_max: safe_test_min = data_max

col_t3, col_t4 = st.sidebar.columns(2)
test_start = col_t3.date_input("Test start", value=safe_test_min,
                               min_value=safe_test_min, max_value=data_max)
test_end   = col_t4.date_input("Test end", value=data_max,
                               min_value=test_start, max_value=data_max)

st.sidebar.markdown("---")
st.sidebar.header("💸 3. Select the overcharge based on peak demand")
demand_charge_per_mw = st.sidebar.number_input("Demand Charge ($/MW/day)", value=50.0, step=5.0)

if st.sidebar.button("🚀 Run LP Optimization", type="primary"):
    st.session_state.lp_run = True
    st.session_state.active_test_start = test_start
    st.session_state.active_test_end = test_end
    st.session_state.active_demand_charge = demand_charge_per_mw

out_of_date = (
    st.session_state.active_test_start != test_start or 
    st.session_state.active_test_end != test_end or 
    st.session_state.active_demand_charge != demand_charge_per_mw
)
if st.session_state.lp_run and out_of_date:
    st.sidebar.warning("⚠️ Test dates or Demand Charge changed. Click Run LP Optimization to update.")

st.sidebar.markdown("---")
st.sidebar.header("⚡ 4. Select the static threshold for case 2")

# We don't have peak_demand_train yet if model isn't trained, so default to 0
if st.session_state.model_trained:
    rf_model, features, peak_demand_train = train_ml_model(df_ml, str(st.session_state.active_train_end))
else:
    peak_demand_train = 0.0

default_discharge_kw = round((peak_demand_train * 0.85), 2)
default_charge_kw = round((peak_demand_train * 0.50), 2)

col_th1, col_th2 = st.sidebar.columns(2)
discharge_threshold_kw = col_th1.number_input("Discharge threshold (kW)", value=default_discharge_kw, step=10.0)
charge_threshold_kw = col_th2.number_input("Charge threshold (kW)", value=default_charge_kw, step=10.0)

# ============================================================
# MAIN UI
# ============================================================
st.title("ECE 537 Smart Grid — Peak Shaving & ML Optimization")
st.markdown("Comparing three scenarios using real building consumption and "
            "**CAISO** wholesale electricity prices.")
st.markdown("1st Scenario: No BESS.")
st.markdown("2nd Scenario: Rule-Based BESS (Fixed Threshold).")
st.markdown("3rd Scenario: ML + Optimization (Random Forest forecast + linprog Day-Ahead schedule).")

if not st.session_state.model_trained:
    st.info("👈 Please select your Training Window and click **Train ML Model** to begin.")
    st.stop()
    
if not st.session_state.lp_run:
    st.info("👈 ML Model trained! Now select your Test Window and click **Run LP Optimization** to simulate the Day-Ahead schedule.")
    st.stop()

with st.spinner("Running LP optimisation on testing data (cached after first run) …"):
    df_test, battery_capacity, max_power, efficiency, dt_hours = run_lp_optimization(
        df_ml, _rf_model=rf_model, features=features, 
        test_start_str=str(st.session_state.active_test_start), 
        test_end_str=str(st.session_state.active_test_end), 
        demand_charge_per_mw=st.session_state.active_demand_charge
    )

st.sidebar.markdown("---")
st.sidebar.header("🔍5. Select Period of interest")
view_mode = st.sidebar.radio("View mode", ["Entire Period", "Specific Day"])

if view_mode == "Specific Day":
    unique_dates = sorted(set(df_test.index.date))
    selected_date = st.sidebar.date_input("Pick a day", value=unique_dates[0],
                                          min_value=unique_dates[0],
                                          max_value=unique_dates[-1])
    df_plot = df_test[df_test.index.date == selected_date].copy()
    title_tag = f"— {selected_date.strftime('%A, %b %d %Y')}"
else:
    df_plot = df_test.copy()
    title_tag = f"— Full Test Period ({test_start} → {test_end})"

# Run S2 on the visible slice (instant)
Load_S2, SOC_S2 = run_s2_simulation(
    df_plot['Load_S1'].values, df_plot.index,
    discharge_threshold_kw, charge_threshold_kw, battery_capacity, max_power, efficiency, dt_hours)
df_plot['Load_S2'] = Load_S2
df_plot['SOC_S2']  = SOC_S2

# ============================================================
# COST BREAKDOWN TABLE
# ============================================================
bd_s1 = calc_cost_breakdown(df_plot['Load_S1'], df_plot['Price'], dt_hours, demand_charge_per_mw)
bd_s2 = calc_cost_breakdown(df_plot['Load_S2'], df_plot['Price'], dt_hours, demand_charge_per_mw)
bd_s3 = calc_cost_breakdown(df_plot['Load_S3'], df_plot['Price'], dt_hours, demand_charge_per_mw)

st.subheader(f"Cost Breakdown {title_tag}")
st.caption(f"Demand charge: **${demand_charge_per_mw:.0f}/MW/day** × peak MW × "
           f"{bd_s1['n_days']} day(s)  ·  "
           f"Battery: **{battery_capacity:.0f} kWh** / **{max_power:.0f} kW** / η={efficiency}")

breakdown_df = pd.DataFrame({
    "S1 — No BESS": {
        "Energy Cost ($)":           bd_s1['energy_cost'],
        "Peak Demand Cost ($)":      bd_s1['demand_cost'],
        "Peak Power (MW)":           bd_s1['peak_kw'] / 1000,
        "Total Consumption (kWh)":   bd_s1['total_kwh'],
        "Total Cost ($)":            bd_s1['total_cost'],
    },
    "S2 — Rule-Based": {
        "Energy Cost ($)":           bd_s2['energy_cost'],
        "Peak Demand Cost ($)":      bd_s2['demand_cost'],
        "Peak Power (MW)":           bd_s2['peak_kw'] / 1000,
        "Total Consumption (kWh)":   bd_s2['total_kwh'],
        "Total Cost ($)":            bd_s2['total_cost'],
    },
    "S3 — ML + Opt": {
        "Energy Cost ($)":           bd_s3['energy_cost'],
        "Peak Demand Cost ($)":      bd_s3['demand_cost'],
        "Peak Power (MW)":           bd_s3['peak_kw'] / 1000,
        "Total Consumption (kWh)":   bd_s3['total_kwh'],
        "Total Cost ($)":            bd_s3['total_cost'],
    },
})

def highlight_best(row):
    styles = [''] * len(row)
    if row.name in ["Energy Cost ($)", "Peak Demand Cost ($)", "Total Cost ($)"]:
        best = row.min()
        for i, v in enumerate(row):
            if v == best:
                styles[i] = 'background-color: #2d6a4f; color: white'
    return styles

st.dataframe(
    breakdown_df.style
        .format({
            "S1 — No BESS":    lambda v: f"${v:,.2f}" if "Cost" in str(breakdown_df.index[breakdown_df.eq(v).any(axis=1)].tolist()) else f"{v:,.2f}",
            "S2 — Rule-Based": lambda v: f"${v:,.2f}" if "Cost" in str(breakdown_df.index[breakdown_df.eq(v).any(axis=1)].tolist()) else f"{v:,.2f}",
            "S3 — ML + Opt":   lambda v: f"${v:,.2f}" if "Cost" in str(breakdown_df.index[breakdown_df.eq(v).any(axis=1)].tolist()) else f"{v:,.2f}",
        })
        .apply(highlight_best, axis=1),
    use_container_width=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("S1 Total", f"${bd_s1['total_cost']:,.2f}", delta_color="off")
if bd_s1['total_cost'] > 0:
    c2.metric("S2 Total", f"${bd_s2['total_cost']:,.2f}",
              f"{(bd_s2['total_cost']-bd_s1['total_cost'])/bd_s1['total_cost']*100:+.1f}% vs S1",
              delta_color="inverse")
    c3.metric("S3 Total", f"${bd_s3['total_cost']:,.2f}",
              f"{(bd_s3['total_cost']-bd_s1['total_cost'])/bd_s1['total_cost']*100:+.1f}% vs S1",
              delta_color="inverse")
else:
    c2.metric("S2 Total", f"${bd_s2['total_cost']:,.2f}")
    c3.metric("S3 Total", f"${bd_s3['total_cost']:,.2f}")

# ============================================================
# MATPLOTLIB PLOTS
# ============================================================
st.markdown("---")
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

axs[0].fill_between(df_plot.index, df_plot['Load_S1'],
                    color='#d3d3d3', alpha=0.55, label='S1: Actual Load (No BESS)')
axs[0].plot(df_plot.index, df_plot['Predicted_Load'],
            color='orange', ls='--', lw=1.5, label='ML Predicted Load')
axs[0].plot(df_plot.index, df_plot['Load_S2'],
            color='#1f77b4', lw=1.5, label='S2: Rule-Based')
axs[0].plot(df_plot.index, df_plot['Load_S3'],
            color='#d62728', lw=2, label='S3: ML + Optimisation')
axs[0].axhline(y=discharge_threshold_kw, color='#1f77b4', ls=':', lw=1,
               label=f'S2 Discharge ({discharge_threshold_kw:.0f} kW)')
axs[0].axhline(y=charge_threshold_kw, color='#2ca02c', ls=':', lw=1,
               label=f'S2 Charge ({charge_threshold_kw:.0f} kW)')
axs[0].set_ylabel("Power Drawn from Grid (kW)")
axs[0].set_title("Grid Load")
axs[0].legend(loc='upper right', fontsize=8)
axs[0].grid(True, alpha=0.3)

axs[1].fill_between(df_plot.index, df_plot['Load_S1'] - df_plot['Load_S2'],
                    color='#1f77b4', alpha=0.4, label='S2 Battery Power (kW)')
axs[1].plot(df_plot.index, df_plot['Load_S1'] - df_plot['Load_S3'],
            color='#d62728', lw=1.5, label='S3 Battery Power (kW)')
axs[1].axhline(y=0, color='black', lw=1)
axs[1].set_ylabel("Battery Power (kW)\n(+) Discharge (-) Charge")
axs[1].set_title("Battery Output Power")
axs[1].legend(loc='upper right', fontsize=8)
axs[1].grid(True, alpha=0.3)

axs[2].step(df_plot.index, df_plot['Price'] * 1000,
            color='#9467bd', where='post', lw=1.5)
axs[2].set_ylabel("Price (USD / MWh)")
axs[2].set_title("CAISO Day-Ahead Wholesale Electricity Price")
axs[2].grid(True, alpha=0.3)

axs[3].plot(df_plot.index, (df_plot['SOC_S2'] / battery_capacity) * 100,
            color='#1f77b4', lw=1.5, label='S2: Rule-Based SOC')
axs[3].plot(df_plot.index, (df_plot['SOC_S3'] / battery_capacity) * 100,
            color='#d62728', lw=2, label='S3: ML + Opt SOC')
axs[3].set_ylabel("SOC (%)")
axs[3].set_ylim([-5, 105])
axs[3].set_title("Battery State of Charge")
axs[3].legend(loc='upper right', fontsize=8)

plt.tight_layout()
st.pyplot(fig)
