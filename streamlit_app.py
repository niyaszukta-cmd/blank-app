import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sqlite3
import os
import pytz

# Try importing calculator
try:
    from gex_calculator import EnhancedGEXDEXCalculator, calculate_dual_gex_dex_flow, detect_gamma_flip_zones
    CALCULATOR_AVAILABLE = True
except Exception as e:
    CALCULATOR_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Try importing data collector functions
try:
    from data_collector import (
        get_available_dates, get_snapshots_for_date, get_strike_data,
        get_flow_metrics, get_intraday_history, get_database_stats,
        DATABASE_FILE
    )
    DATABASE_AVAILABLE = os.path.exists(DATABASE_FILE)
except:
    DATABASE_AVAILABLE = False
    DATABASE_FILE = "gex_dex_history.db"

# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def check_password():
    """Returns True if user has entered correct password"""
    
    def password_entered():
        username = st.session_state["username"].strip().lower()
        password = st.session_state["password"]
        
        users = {
            "demo": "demo123",
            "premium": "premium123",
            "niyas": "nyztrade123"
        }
        
        if username in users and password == users[username]:
            st.session_state["password_correct"] = True
            st.session_state["authenticated_user"] = username
            del st.session_state["password"]
            return
        
        st.session_state["password_correct"] = False
        st.session_state["authenticated_user"] = None
    
    if "password_correct" not in st.session_state:
        st.markdown("## 🔐 NYZTrade Dashboard Login")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.text_input("Username", key="username", placeholder="Enter username")
            st.text_input("Password", type="password", key="password", placeholder="Enter password")
            st.button("Login", on_click=password_entered, use_container_width=True)
            
            st.markdown("---")
            st.info("""
            **Demo Credentials:**
            - Free: `demo` / `demo123`
            - Premium: `premium` / `premium123`
            
            **Contact**: Subscribe to NYZTrade YouTube
            """)
        
        return False
    
    elif not st.session_state["password_correct"]:
        st.markdown("## 🔐 NYZTrade Dashboard Login")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.error("😕 Incorrect username or password")
            st.text_input("Username", key="username", placeholder="Enter username")
            st.text_input("Password", type="password", key="password", placeholder="Enter password")
            st.button("Login", on_click=password_entered, use_container_width=True)
        
        return False
    
    return True

def get_user_tier():
    if "authenticated_user" not in st.session_state:
        return "guest"
    
    username = st.session_state["authenticated_user"]
    premium_users = ["premium", "niyas"]
    
    return "premium" if username in premium_users else "basic"

def get_ist_time():
    """Get current time in IST"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NYZTrade - GEX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check auth
if not check_password():
    st.stop()

user_tier = get_user_tier()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'view_mode': 'live',  # 'live' or 'historical'
        'selected_date': None,
        'selected_snapshot_id': None,
        'session_snapshots': {},  # For session-based captures
        'session_times': [],
        'auto_capture': True,
        'capture_interval': 3,
        'last_capture_time': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .db-status {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .db-offline {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================

def check_database():
    """Check if database exists and has data"""
    if not os.path.exists(DATABASE_FILE):
        return False, "Database file not found"
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM snapshots")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            return True, f"{count} snapshots available"
        return False, "Database empty"
    except Exception as e:
        return False, str(e)


def load_snapshot_from_db(snapshot_id):
    """Load complete snapshot data from database"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get snapshot info
        cursor.execute("""
            SELECT timestamp, symbol, futures_ltp, spot_price, fetch_method,
                   total_gex, total_dex, gex_bias, dex_bias, combined_bias,
                   atm_strike, atm_straddle_premium, pcr, expiry_date
            FROM snapshots WHERE id = ?
        """, (snapshot_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None, None, None, None, None
        
        timestamp, symbol, futures_ltp, spot_price, fetch_method = row[:5]
        total_gex, total_dex, gex_bias, dex_bias, combined_bias = row[5:10]
        atm_strike, atm_straddle_premium, pcr, expiry_date = row[10:]
        
        # Get strike data
        df = pd.read_sql_query("""
            SELECT strike as Strike, call_oi as Call_OI, put_oi as Put_OI,
                   call_oi_change as Call_OI_Change, put_oi_change as Put_OI_Change,
                   call_volume as Call_Volume, put_volume as Put_Volume,
                   call_iv as Call_IV, put_iv as Put_IV,
                   call_ltp as Call_LTP, put_ltp as Put_LTP,
                   net_gex as Net_GEX_B, net_dex as Net_DEX_B,
                   hedging_pressure as Hedging_Pressure
            FROM strike_data WHERE snapshot_id = ?
            ORDER BY strike ASC
        """, conn, params=(snapshot_id,))
        
        # Add computed columns
        df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']
        df['Call_GEX'] = df['Net_GEX_B'].clip(lower=0)
        df['Put_GEX'] = df['Net_GEX_B'].clip(upper=0)
        
        # Get flow metrics
        cursor.execute("""
            SELECT gex_near_total, gex_near_positive, gex_near_negative,
                   dex_near_total, dex_near_positive, dex_near_negative,
                   combined_signal, max_call_oi_strike, max_put_oi_strike
            FROM flow_metrics WHERE snapshot_id = ?
        """, (snapshot_id,))
        
        flow_row = cursor.fetchone()
        if flow_row:
            flow_metrics = {
                'gex_near_total': flow_row[0],
                'gex_near_positive': flow_row[1],
                'gex_near_negative': flow_row[2],
                'gex_near_bias': gex_bias,
                'dex_near_total': flow_row[3],
                'dex_near_positive': flow_row[4],
                'dex_near_negative': flow_row[5],
                'dex_near_bias': dex_bias,
                'combined_signal': flow_row[6],
                'combined_bias': combined_bias,
                'max_call_oi_strike': flow_row[7],
                'max_put_oi_strike': flow_row[8]
            }
        else:
            flow_metrics = {
                'gex_near_bias': gex_bias,
                'dex_near_bias': dex_bias,
                'combined_bias': combined_bias,
                'gex_near_total': total_gex,
                'dex_near_total': total_dex
            }
        
        atm_info = {
            'atm_strike': atm_strike,
            'atm_straddle_premium': atm_straddle_premium,
            'spot_price': spot_price,
            'expiry_date': expiry_date
        }
        
        conn.close()
        
        return df, futures_ltp, fetch_method, atm_info, flow_metrics, timestamp
        
    except Exception as e:
        return None, None, None, None, None, None


def get_db_snapshots_for_slider(symbol, date):
    """Get snapshots for time slider"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, futures_ltp, total_gex
            FROM snapshots 
            WHERE symbol = ? AND DATE(timestamp) = ?
            ORDER BY timestamp ASC
        """, (symbol, date))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{'id': r[0], 'timestamp': r[1], 'futures_ltp': r[2], 'total_gex': r[3]} for r in rows]
    except:
        return []


# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">📊 NYZTrade - Advanced GEX + DEX Analysis</p>', unsafe_allow_html=True)
st.markdown("**Real-time Gamma & Delta Exposure Analysis | Historical Database Backtest**")

# User badge
if user_tier == "premium":
    st.sidebar.success("👑 **Premium Member**")
else:
    st.sidebar.info(f"🆓 **Free Member** | User: {st.session_state.get('authenticated_user', 'guest')}")

# Logout
if st.sidebar.button("🚪 Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.header("⚙️ Dashboard Settings")

symbol = st.sidebar.selectbox(
    "Select Index",
    ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
    index=0
)

strikes_range = st.sidebar.slider(
    "Strikes Range",
    min_value=5,
    max_value=20,
    value=12
)

expiry_index = st.sidebar.selectbox(
    "Expiry Selection",
    [0, 1, 2],
    format_func=lambda x: ["Current Weekly", "Next Weekly", "Monthly"][x],
    index=0
)

# ============================================================================
# DATABASE STATUS & TIME MACHINE
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Historical Database")

db_available, db_message = check_database()

if db_available:
    st.sidebar.markdown(f'<div class="db-status">✅ {db_message}</div>', unsafe_allow_html=True)
    
    # View Mode Toggle
    view_mode = st.sidebar.radio(
        "View Mode",
        ["🔴 Live Data", "📜 Historical Data"],
        index=0 if st.session_state.view_mode == 'live' else 1,
        key="view_mode_radio"
    )
    
    st.session_state.view_mode = 'live' if "Live" in view_mode else 'historical'
    
    # Historical data selector
    if st.session_state.view_mode == 'historical':
        try:
            available_dates = get_available_dates(symbol)
            
            if available_dates:
                selected_date = st.sidebar.selectbox(
                    "Select Date",
                    available_dates,
                    format_func=lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%d %b %Y'),
                    key="date_selector"
                )
                st.session_state.selected_date = selected_date
            else:
                st.sidebar.warning(f"No data for {symbol}")
                st.session_state.view_mode = 'live'
        except Exception as e:
            st.sidebar.error(f"Error loading dates: {e}")
            st.session_state.view_mode = 'live'
else:
    st.sidebar.markdown(f'<div class="db-offline">❌ {db_message}</div>', unsafe_allow_html=True)
    st.sidebar.info("""
    **To enable historical data:**
    1. Run `data_collector.py` 
    2. Or use `--continuous` mode
    """)
    st.session_state.view_mode = 'live'

# Manual Refresh
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ============================================================================
# TIME MACHINE UI (for historical mode)
# ============================================================================

if st.session_state.view_mode == 'historical' and st.session_state.selected_date:
    st.markdown("---")
    st.markdown("### ⏰ Time Machine - Historical Backtest")
    
    # Get snapshots for selected date
    snapshots = get_db_snapshots_for_slider(symbol, st.session_state.selected_date)
    
    if snapshots:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.caption(f"📊 **{len(snapshots)} snapshots** on {datetime.strptime(st.session_state.selected_date, '%Y-%m-%d').strftime('%d %b %Y')}")
        
        with col2:
            if st.button("🔴 Switch to Live", use_container_width=True):
                st.session_state.view_mode = 'live'
                st.rerun()
        
        # Time Slider
        time_labels = []
        for s in snapshots:
            try:
                t = datetime.strptime(s['timestamp'], '%Y-%m-%d %H:%M:%S')
                time_labels.append(t.strftime('%I:%M %p'))
            except:
                time_labels.append(s['timestamp'])
        
        selected_idx = st.select_slider(
            "🕐 Select Time Point",
            options=list(range(len(snapshots))),
            value=len(snapshots) - 1,
            format_func=lambda x: time_labels[x],
            key="hist_time_slider"
        )
        
        st.session_state.selected_snapshot_id = snapshots[selected_idx]['id']
        
        # Quick Jump Buttons
        st.markdown("**⚡ Quick Jump:**")
        cols = st.columns(6)
        jumps = [("First", 0), ("9:30", None), ("11:00", None), ("13:00", None), ("15:00", None), ("Last", -1)]
        
        for idx, (label, target) in enumerate(jumps):
            with cols[idx]:
                if st.button(label, key=f"jump_{label}", use_container_width=True):
                    if target == 0:
                        st.session_state.selected_snapshot_id = snapshots[0]['id']
                    elif target == -1:
                        st.session_state.selected_snapshot_id = snapshots[-1]['id']
                    else:
                        # Find closest to target time
                        target_hour = int(label.split(':')[0])
                        target_min = int(label.split(':')[1]) if ':' in label else 0
                        
                        for s in snapshots:
                            try:
                                t = datetime.strptime(s['timestamp'], '%Y-%m-%d %H:%M:%S')
                                if t.hour >= target_hour and t.minute >= target_min:
                                    st.session_state.selected_snapshot_id = s['id']
                                    break
                            except:
                                pass
                    st.rerun()
        
        # Intraday History Chart from Database
        try:
            hist_df = get_intraday_history(symbol, st.session_state.selected_date)
            
            if not hist_df.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                                   vertical_spacing=0.08, subplot_titles=('📈 Futures Price', '📊 GEX Flow'))
                
                # Convert timestamp
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                
                # Price line
                fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['futures_ltp'],
                                        mode='lines+markers', line=dict(color='#6c5ce7', width=2),
                                        marker=dict(size=4), name='Price'), row=1, col=1)
                
                # GEX bars
                gex_colors = ['#00d4aa' if x > 0 else '#ff6b6b' for x in hist_df['total_gex']]
                fig.add_trace(go.Bar(x=hist_df['timestamp'], y=hist_df['total_gex'],
                                    marker_color=gex_colors, name='GEX'), row=2, col=1)
                
                # Mark selected point
                selected_snap = next((s for s in snapshots if s['id'] == st.session_state.selected_snapshot_id), None)
                if selected_snap:
                    try:
                        sel_time = datetime.strptime(selected_snap['timestamp'], '%Y-%m-%d %H:%M:%S')
                        fig.add_vline(x=sel_time, line_dash="dash", line_color="orange", line_width=2)
                    except:
                        pass
                
                fig.update_layout(height=300, showlegend=False, template='plotly_dark',
                                 margin=dict(l=50, r=50, t=50, b=30))
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load history chart: {e}")
    else:
        st.warning("No snapshots found for selected date")
        st.session_state.view_mode = 'live'

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_data(symbol, strikes_range, expiry_index):
    if not CALCULATOR_AVAILABLE:
        return None, None, None, None, f"Calculator not available: {IMPORT_ERROR}"
    
    try:
        calculator = EnhancedGEXDEXCalculator()
        df, futures_ltp, fetch_method, atm_info = calculator.fetch_and_calculate_gex_dex(
            symbol=symbol,
            strikes_range=strikes_range,
            expiry_index=expiry_index
        )
        return df, futures_ltp, fetch_method, atm_info, None
    except Exception as e:
        return None, None, None, None, str(e)

# ============================================================================
# MAIN ANALYSIS - LOAD DATA
# ============================================================================

st.markdown("---")

if st.session_state.view_mode == 'historical' and st.session_state.selected_snapshot_id:
    # Load from database
    result = load_snapshot_from_db(st.session_state.selected_snapshot_id)
    
    if result[0] is not None:
        df, futures_ltp, fetch_method, atm_info, flow_metrics, timestamp = result
        is_historical = True
        
        try:
            hist_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            hist_time_str = hist_time.strftime('%I:%M:%S %p IST on %d %b %Y')
        except:
            hist_time_str = timestamp
            hist_time = None
        
        st.warning(f"📜 **HISTORICAL MODE** - Viewing data from {hist_time_str}")
    else:
        st.error("Failed to load historical data")
        st.session_state.view_mode = 'live'
        is_historical = False
        flow_metrics = None
else:
    # Fetch live data
    is_historical = False
    hist_time = None
    
    with st.spinner(f"🔄 Fetching live {symbol} data..."):
        df, futures_ltp, fetch_method, atm_info, error = fetch_live_data(symbol, strikes_range, expiry_index)
    
    if error:
        st.error(f"❌ Error: {error}")
        st.info("""
        **Troubleshooting:**
        1. Make sure gex_calculator.py is uploaded
        2. Check requirements.txt
        3. Wait 1-2 minutes for dependencies
        """)
        st.stop()
    
    if df is None:
        st.error("❌ Failed to fetch data")
        st.stop()
    
    # Calculate flow metrics
    try:
        flow_metrics = calculate_dual_gex_dex_flow(df, futures_ltp)
    except Exception as e:
        flow_metrics = None
    
    st.success(f"🔴 **LIVE MODE** - Real-time data via {fetch_method}")

# ============================================================================
# KEY METRICS
# ============================================================================

st.subheader("📊 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_gex = float(df['Net_GEX_B'].sum())
    st.metric(
        "Total Net GEX",
        f"{total_gex:.4f}B",
        delta="Bullish" if total_gex > 0 else "Volatile"
    )

with col2:
    if 'Call_GEX' in df.columns:
        call_gex = float(df['Call_GEX'].sum())
    else:
        call_gex = float(df[df['Net_GEX_B'] > 0]['Net_GEX_B'].sum())
    st.metric("Call GEX", f"{call_gex:.4f}B")

with col3:
    if 'Put_GEX' in df.columns:
        put_gex = float(df['Put_GEX'].sum())
    else:
        put_gex = float(df[df['Net_GEX_B'] < 0]['Net_GEX_B'].sum())
    st.metric("Put GEX", f"{put_gex:.4f}B")

with col4:
    st.metric("Futures LTP", f"₹{futures_ltp:,.2f}")

with col5:
    if atm_info:
        st.metric("ATM Straddle", f"₹{atm_info['atm_straddle_premium']:.2f}")

# ============================================================================
# FLOW METRICS
# ============================================================================

if flow_metrics:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gex_bias = flow_metrics.get('gex_near_bias', 'N/A')
        if "BULLISH" in str(gex_bias).upper():
            st.markdown(f'<div class="success-box"><b>GEX:</b> {gex_bias}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box"><b>GEX:</b> {gex_bias}</div>', unsafe_allow_html=True)
    
    with col2:
        dex_bias = flow_metrics.get('dex_near_bias', 'N/A')
        st.info(f"**DEX Bias:** {dex_bias}")
    
    with col3:
        combined_bias = flow_metrics.get('combined_bias', 'N/A')
        st.info(f"**Combined:** {combined_bias}")
else:
    st.warning("Flow metrics unavailable")

# ============================================================================
# GAMMA FLIP ZONES
# ============================================================================

gamma_flip_zones = []
if not is_historical:
    try:
        gamma_flip_zones = detect_gamma_flip_zones(df)
        if gamma_flip_zones:
            st.warning(f"⚡ **{len(gamma_flip_zones)} Gamma Flip Zone(s) Detected!**")
    except:
        pass

# ============================================================================
# CHARTS
# ============================================================================

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 GEX Profile", "📈 DEX Profile", "🎯 Hedging Pressure", "📋 Data Table", "💡 Strategies"])

# TAB 1: GEX Profile
with tab1:
    mode_text = f"[HISTORICAL - {hist_time.strftime('%I:%M %p') if hist_time else 'DB'}]" if is_historical else "[LIVE]"
    st.subheader(f"NYZTrade - {symbol} Gamma Exposure Profile {mode_text}")
    
    fig = go.Figure()
    
    colors = ['green' if x > 0 else 'red' for x in df['Net_GEX_B']]
    
    fig.add_trace(go.Bar(
        y=df['Strike'],
        x=df['Net_GEX_B'],
        orientation='h',
        marker_color=colors,
        name='Net GEX',
        hovertemplate='<b>Strike:</b> %{y}<br><b>Net GEX:</b> %{x:.4f}B<extra></extra>'
    ))
    
    if gamma_flip_zones:
        max_gex = df['Net_GEX_B'].abs().max()
        for zone in gamma_flip_zones:
            fig.add_shape(
                type="rect",
                y0=zone['lower_strike'],
                y1=zone['upper_strike'],
                x0=-max_gex * 1.5,
                x1=max_gex * 1.5,
                fillcolor="yellow",
                opacity=0.2,
                layer="below",
                line_width=0
            )
    
    fig.add_hline(
        y=futures_ltp,
        line_dash="dash",
        line_color="blue",
        line_width=3,
        annotation_text=f"Futures: {futures_ltp:,.2f}"
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Net GEX (Billions)",
        yaxis_title="Strike Price",
        template='plotly_white',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if total_gex > 0.5:
        st.success("🟢 **Strong Positive GEX**: Sideways to bullish market expected")
    elif total_gex < -0.5:
        st.error("🔴 **Negative GEX**: High volatility expected")
    else:
        st.warning("⚖️ **Neutral GEX**: Mixed signals")

# TAB 2: DEX Profile
with tab2:
    mode_text = f"[HISTORICAL]" if is_historical else "[LIVE]"
    st.subheader(f"NYZTrade - {symbol} Delta Exposure Profile {mode_text}")
    
    fig2 = go.Figure()
    
    dex_colors = ['green' if x > 0 else 'red' for x in df['Net_DEX_B']]
    
    fig2.add_trace(go.Bar(
        y=df['Strike'],
        x=df['Net_DEX_B'],
        orientation='h',
        marker_color=dex_colors,
        name='Net DEX',
        hovertemplate='<b>Strike:</b> %{y}<br><b>Net DEX:</b> %{x:.4f}B<extra></extra>'
    ))
    
    fig2.add_hline(
        y=futures_ltp,
        line_dash="dash",
        line_color="blue",
        line_width=3
    )
    
    fig2.update_layout(
        height=600,
        xaxis_title="Net DEX (Billions)",
        yaxis_title="Strike Price",
        template='plotly_white'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# TAB 3: Hedging Pressure
with tab3:
    st.subheader(f"NYZTrade - {symbol} Hedging Pressure Index")
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Bar(
        y=df['Strike'],
        x=df['Hedging_Pressure'],
        orientation='h',
        marker=dict(
            color=df['Hedging_Pressure'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Pressure", x=1.15)
        ),
        name='Hedging Pressure',
        hovertemplate='<b>Strike:</b> %{y}<br><b>Pressure:</b> %{x:.2f}%<extra></extra>'
    ))
    
    if 'Total_Volume' in df.columns:
        max_pressure = df['Hedging_Pressure'].abs().max()
        max_vol = df['Total_Volume'].max()
        
        if max_vol > 0:
            vol_scale = (max_pressure * 0.3) / max_vol
            scaled_volume = df['Total_Volume'] * vol_scale
            
            fig3.add_trace(go.Scatter(
                y=df['Strike'],
                x=scaled_volume,
                mode='lines+markers',
                line=dict(color='cyan', width=2),
                marker=dict(size=4),
                name='Volume',
                hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                customdata=df['Total_Volume']
            ))
    
    fig3.add_hline(
        y=futures_ltp,
        line_dash="dash",
        line_color="blue",
        line_width=3
    )
    
    fig3.update_layout(
        height=600,
        xaxis_title="Hedging Pressure (%)",
        yaxis_title="Strike Price",
        template='plotly_white'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# TAB 4: Data Table
with tab4:
    st.subheader("Strike-wise Analysis")
    
    if is_historical:
        st.caption(f"📜 Historical data")
    
    display_cols = [c for c in ['Strike', 'Call_OI', 'Put_OI', 'Net_GEX_B', 'Net_DEX_B', 'Hedging_Pressure', 'Total_Volume'] if c in df.columns]
    display_df = df[display_cols].copy()
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    csv = df.to_csv(index=False)
    timestamp_str = hist_time.strftime('%Y%m%d_%H%M') if hist_time else get_ist_time().strftime('%Y%m%d_%H%M')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"NYZTrade_{symbol}_{timestamp_str}.csv",
        mime="text/csv",
        use_container_width=True
    )

# TAB 5: Strategies
with tab5:
    st.subheader("💡 Trading Strategies")
    
    if is_historical:
        st.info(f"📜 Strategies based on historical data")
    
    if flow_metrics and atm_info:
        gex_bias_val = flow_metrics.get('gex_near_total', total_gex)
        dex_bias_val = flow_metrics.get('dex_near_total', 0)
        
        st.markdown("### 📊 Market Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("GEX Flow", f"{gex_bias_val:.2f}")
            st.metric("DEX Flow", f"{dex_bias_val:.2f}")
        with col2:
            st.metric("ATM Strike", f"{atm_info['atm_strike']}")
            st.metric("Straddle", f"₹{atm_info['atm_straddle_premium']:.2f}")
        
        st.markdown("---")
        
        if gex_bias_val > 50:
            st.success("### 🟢 Strong Positive GEX - Sideways/Bullish")
            st.code(f"""
Strategy: Iron Condor / Short Straddle
ATM: {atm_info['atm_strike']}
Premium: ₹{atm_info['atm_straddle_premium']:.2f}
            """)
        elif gex_bias_val < -50:
            st.error("### 🔴 Negative GEX - High Volatility")
            st.code(f"""
Strategy: Long Straddle
ATM: {atm_info['atm_strike']}
Cost: ₹{atm_info['atm_straddle_premium']:.2f}
            """)
        else:
            st.warning("### ⚖️ Neutral - Wait for Clarity")
    else:
        st.warning("Metrics unavailable")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

ist_time = get_ist_time()

with col1:
    st.info(f"⏰ {ist_time.strftime('%H:%M:%S')} IST")

with col2:
    st.info(f"📅 {ist_time.strftime('%d %b %Y')}")

with col3:
    if is_historical:
        st.warning(f"📜 Historical")
    else:
        st.success(f"🔴 Live: {symbol}")

with col4:
    if db_available:
        st.success("💾 DB Online")
    else:
        st.error("💾 DB Offline")

st.markdown(f"**💡 NYZTrade YouTube | Data: {fetch_method if not is_historical else 'Database'}**")
