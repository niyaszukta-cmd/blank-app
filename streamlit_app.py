import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import hashlib
import hmac
import pytz

# Try importing calculator
try:
    from gex_calculator import EnhancedGEXDEXCalculator, calculate_dual_gex_dex_flow, detect_gamma_flip_zones
    CALCULATOR_AVAILABLE = True
except Exception as e:
    CALCULATOR_AVAILABLE = False
    IMPORT_ERROR = str(e)

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
# SESSION STATE INITIALIZATION FOR TIME MACHINE
# ============================================================================

def init_time_machine_state():
    """Initialize all Time Machine session state variables"""
    defaults = {
        'data_snapshots': {},
        'snapshot_times': [],
        'selected_time_index': None,
        'is_live_mode': True,
        'last_capture_time': None,
        'auto_capture': True,
        'capture_interval': 3,
        'force_capture': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_time_machine_state()

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
    .countdown-timer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .time-machine-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    .historical-indicator {
        color: #ffa500;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
        50% { opacity: 0.7; box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
        100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TIME MACHINE FUNCTIONS
# ============================================================================

def capture_snapshot(df, futures_ltp, fetch_method, atm_info, flow_metrics):
    """Capture current data as a snapshot for Time Machine"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist).replace(microsecond=0)
    
    # Check capture interval
    if st.session_state.last_capture_time:
        elapsed = (now - st.session_state.last_capture_time).total_seconds() / 60
        if elapsed < st.session_state.capture_interval:
            return False
    
    # Store snapshot
    st.session_state.data_snapshots[now] = {
        'df': df.copy(),
        'futures_ltp': futures_ltp,
        'fetch_method': fetch_method,
        'atm_info': atm_info.copy() if atm_info else None,
        'flow_metrics': flow_metrics.copy() if flow_metrics else None
    }
    
    # Add to times list
    if now not in st.session_state.snapshot_times:
        st.session_state.snapshot_times.append(now)
        st.session_state.snapshot_times.sort()
    
    st.session_state.last_capture_time = now
    
    # Limit to 500 snapshots (memory management)
    while len(st.session_state.snapshot_times) > 500:
        oldest = st.session_state.snapshot_times.pop(0)
        st.session_state.data_snapshots.pop(oldest, None)
    
    return True


def render_time_machine():
    """Render the Time Machine UI with slider"""
    st.markdown("---")
    
    # Header row
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("### ⏰ Time Machine - Backtest Mode")
    
    with col2:
        if st.session_state.is_live_mode:
            st.markdown('<span class="live-indicator"></span> **LIVE**', unsafe_allow_html=True)
        else:
            st.markdown('<span class="historical-indicator">📜 HISTORICAL</span>', unsafe_allow_html=True)
    
    with col3:
        if not st.session_state.is_live_mode:
            if st.button("🔴 Go Live", use_container_width=True, key="go_live_btn"):
                st.session_state.is_live_mode = True
                st.session_state.selected_time_index = None
                st.rerun()
    
    # No snapshots yet
    if not st.session_state.snapshot_times:
        st.info("📝 No historical data yet. Snapshots will be captured automatically every few minutes.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.auto_capture = st.checkbox(
                "🔄 Auto-capture enabled",
                value=st.session_state.auto_capture,
                key="auto_cap_empty"
            )
        with col2:
            st.session_state.capture_interval = st.selectbox(
                "Capture interval",
                options=[1, 2, 3, 5, 10],
                index=[1, 2, 3, 5, 10].index(st.session_state.capture_interval) if st.session_state.capture_interval in [1, 2, 3, 5, 10] else 2,
                format_func=lambda x: f"{x} min",
                key="interval_empty"
            )
        return None
    
    # Show time range info
    first_time = st.session_state.snapshot_times[0]
    last_time = st.session_state.snapshot_times[-1]
    
    st.caption(f"📊 **{len(st.session_state.snapshot_times)} snapshots** | {first_time.strftime('%I:%M %p')} → {last_time.strftime('%I:%M %p')}")
    
    # Time Slider
    if len(st.session_state.snapshot_times) > 1:
        time_labels = [t.strftime('%I:%M %p') for t in st.session_state.snapshot_times]
        
        current_idx = st.session_state.selected_time_index
        if current_idx is None:
            current_idx = len(st.session_state.snapshot_times) - 1
        
        selected_idx = st.select_slider(
            "🕐 Select Time Point",
            options=list(range(len(st.session_state.snapshot_times))),
            value=current_idx,
            format_func=lambda x: time_labels[x],
            key="time_slider"
        )
        
        # If not at latest, switch to historical mode
        if selected_idx != len(st.session_state.snapshot_times) - 1:
            st.session_state.is_live_mode = False
            st.session_state.selected_time_index = selected_idx
        
        # Quick Jump Buttons
        st.markdown("**⚡ Quick Jump:**")
        cols = st.columns(8)
        presets = [
            ("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60),
            ("2h", 120), ("3h", 180), ("Start", 9999)
        ]
        
        for idx, (label, minutes) in enumerate(presets):
            with cols[idx]:
                if st.button(label, key=f"preset_{minutes}", use_container_width=True):
                    if minutes == 9999:
                        target_idx = 0
                    else:
                        ist = pytz.timezone('Asia/Kolkata')
                        target_time = datetime.now(ist) - timedelta(minutes=minutes)
                        # Find closest snapshot
                        target_idx = min(
                            range(len(st.session_state.snapshot_times)),
                            key=lambda i: abs((st.session_state.snapshot_times[i] - target_time).total_seconds())
                        )
                    st.session_state.selected_time_index = target_idx
                    st.session_state.is_live_mode = False
                    st.rerun()
    
    # Capture Settings Expander
    with st.expander("⚙️ Capture Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.auto_capture = st.checkbox(
                "🔄 Auto-capture",
                value=st.session_state.auto_capture,
                key="auto_cap_settings"
            )
        
        with col2:
            st.session_state.capture_interval = st.selectbox(
                "Interval",
                options=[1, 2, 3, 5, 10],
                index=[1, 2, 3, 5, 10].index(st.session_state.capture_interval) if st.session_state.capture_interval in [1, 2, 3, 5, 10] else 2,
                format_func=lambda x: f"{x} min",
                key="interval_settings"
            )
        
        with col3:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.data_snapshots = {}
                st.session_state.snapshot_times = []
                st.session_state.selected_time_index = None
                st.session_state.is_live_mode = True
                st.rerun()
    
    # Return historical data if in historical mode
    if not st.session_state.is_live_mode and st.session_state.selected_time_index is not None:
        selected_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
        return st.session_state.data_snapshots.get(selected_time)
    
    return None


def create_history_chart():
    """Create intraday price and GEX flow history chart"""
    if len(st.session_state.snapshot_times) < 2:
        return None
    
    times = []
    prices = []
    gex_values = []
    
    for t in st.session_state.snapshot_times:
        if t in st.session_state.data_snapshots:
            snapshot = st.session_state.data_snapshots[t]
            times.append(t)
            prices.append(snapshot['futures_ltp'])
            
            # Get GEX total from flow_metrics or calculate from df
            if snapshot.get('flow_metrics') and 'gex_near_total' in snapshot['flow_metrics']:
                gex_values.append(snapshot['flow_metrics']['gex_near_total'])
            else:
                gex_values.append(float(snapshot['df']['Net_GEX_B'].sum()))
    
    if len(times) < 2:
        return None
    
    # Create subplot with price on top, GEX below
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=('📈 Futures Price', '📊 GEX Flow')
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=times, y=prices,
            mode='lines+markers',
            line=dict(color='#6c5ce7', width=2),
            marker=dict(size=5),
            name='Futures',
            hovertemplate='%{x|%I:%M %p}<br>₹%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # GEX bars
    gex_colors = ['#00d4aa' if x > 0 else '#ff6b6b' for x in gex_values]
    fig.add_trace(
        go.Bar(
            x=times, y=gex_values,
            marker_color=gex_colors,
            name='GEX Flow',
            hovertemplate='%{x|%I:%M %p}<br>GEX: %{y:.4f}B<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Mark selected time if in historical mode
    if not st.session_state.is_live_mode and st.session_state.selected_time_index is not None:
        selected_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
        if selected_time in st.session_state.data_snapshots:
            selected_price = st.session_state.data_snapshots[selected_time]['futures_ltp']
            fig.add_vline(x=selected_time, line_dash="dash", line_color="orange", line_width=2)
            fig.add_annotation(
                x=selected_time, y=selected_price,
                text="📍 Selected",
                showarrow=True, arrowhead=2,
                row=1, col=1
            )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=30),
        paper_bgcolor='rgba(26, 26, 46, 0.8)',
        plot_bgcolor='rgba(26, 26, 46, 0.8)'
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">📊 NYZTrade - Advanced GEX + DEX Analysis</p>', unsafe_allow_html=True)
st.markdown("**Real-time Gamma & Delta Exposure Analysis for Indian Markets | With Time Machine Backtest**")

# User badge
if user_tier == "premium":
    st.sidebar.success("👑 **Premium Member**")
else:
    st.sidebar.info(f"🆓 **Free Member** | User: {st.session_state.get('authenticated_user', 'guest')}")

# Logout button
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

# Time Machine Stats in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⏰ Time Machine Stats")

if st.session_state.snapshot_times:
    st.sidebar.metric("Snapshots", len(st.session_state.snapshot_times))
    st.sidebar.caption(f"First: {st.session_state.snapshot_times[0].strftime('%I:%M %p')}")
    st.sidebar.caption(f"Last: {st.session_state.snapshot_times[-1].strftime('%I:%M %p')}")
else:
    st.sidebar.info("No snapshots yet")

# Manual Capture Button
if st.sidebar.button("📸 Capture Now", use_container_width=True, type="primary"):
    st.session_state.force_capture = True

# Auto-refresh
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto-Refresh")

if user_tier == "premium":
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Interval (seconds)",
            min_value=30,
            max_value=300,
            value=60,
            step=30
        )
        
        if 'countdown_start' not in st.session_state:
            st.session_state.countdown_start = time.time()
        
        elapsed = time.time() - st.session_state.countdown_start
        remaining = max(0, refresh_interval - int(elapsed))
        
        countdown_placeholder = st.sidebar.empty()
        countdown_placeholder.markdown(f'<div class="countdown-timer">⏱️ Next refresh: {remaining}s</div>', unsafe_allow_html=True)
else:
    st.sidebar.info("🔒 Auto-refresh: Premium only")
    auto_refresh = False
    refresh_interval = 60

# Manual refresh
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()
    if 'countdown_start' in st.session_state:
        st.session_state.countdown_start = time.time()
    st.rerun()

# ============================================================================
# TIME MACHINE UI
# ============================================================================

historical_data = render_time_machine()

# Display History Chart if we have snapshots
if len(st.session_state.snapshot_times) >= 2:
    history_chart = create_history_chart()
    if history_chart:
        st.plotly_chart(history_chart, use_container_width=True)

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol, strikes_range, expiry_index):
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

# Check if we're viewing historical data
if historical_data and not st.session_state.is_live_mode:
    # Use historical data
    df = historical_data['df']
    futures_ltp = historical_data['futures_ltp']
    fetch_method = historical_data['fetch_method']
    atm_info = historical_data['atm_info']
    flow_metrics = historical_data['flow_metrics']
    
    is_historical = True
    hist_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
    
    st.warning(f"📜 **HISTORICAL MODE** - Viewing data from {hist_time.strftime('%I:%M:%S %p IST')}")
else:
    # Fetch live data
    is_historical = False
    hist_time = None
    
    with st.spinner(f"🔄 Fetching live {symbol} data..."):
        df, futures_ltp, fetch_method, atm_info, error = fetch_data(symbol, strikes_range, expiry_index)
    
    if error:
        st.error(f"❌ Error: {error}")
        st.info("""
        **Troubleshooting:**
        1. Make sure gex_calculator.py is uploaded
        2. Check requirements.txt includes: streamlit pandas numpy plotly scipy requests pytz
        3. Wait 1-2 minutes for dependencies
        """)
        st.stop()
    
    if df is None:
        st.error("❌ Failed to fetch data")
        st.stop()
    
    # Calculate flow metrics for live data
    try:
        flow_metrics = calculate_dual_gex_dex_flow(df, futures_ltp)
    except Exception as e:
        flow_metrics = None
    
    # Auto-capture snapshot
    if st.session_state.auto_capture or st.session_state.force_capture:
        if capture_snapshot(df, futures_ltp, fetch_method, atm_info, flow_metrics):
            st.toast("📸 Snapshot captured!", icon="✅")
        st.session_state.force_capture = False
    
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
    call_gex = float(df['Call_GEX'].sum())
    st.metric("Call GEX", f"{call_gex:.4f}B")

with col3:
    put_gex = float(df['Put_GEX'].sum())
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
        gex_bias = flow_metrics['gex_near_bias']
        if "BULLISH" in gex_bias:
            st.markdown(f'<div class="success-box"><b>GEX:</b> {gex_bias}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box"><b>GEX:</b> {gex_bias}</div>', unsafe_allow_html=True)
    
    with col2:
        dex_bias = flow_metrics['dex_near_bias']
        st.info(f"**DEX Bias:** {dex_bias}")
    
    with col3:
        combined_bias = flow_metrics['combined_bias']
        st.info(f"**Combined:** {combined_bias}")
else:
    st.warning("Flow metrics unavailable")

# ============================================================================
# GAMMA FLIP ZONES
# ============================================================================

try:
    gamma_flip_zones = detect_gamma_flip_zones(df)
    if gamma_flip_zones:
        st.warning(f"⚡ **{len(gamma_flip_zones)} Gamma Flip Zone(s) Detected!**")
except:
    gamma_flip_zones = []

# ============================================================================
# CHARTS
# ============================================================================

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 GEX Profile", "📈 DEX Profile", "🎯 Hedging Pressure", "📋 Data Table", "💡 Strategies"])

# TAB 1: GEX Profile
with tab1:
    mode_text = f"[HISTORICAL - {hist_time.strftime('%I:%M %p')}]" if is_historical else "[LIVE]"
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
    mode_text = f"[HISTORICAL - {hist_time.strftime('%I:%M %p')}]" if is_historical else "[LIVE]"
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
    mode_text = f"[HISTORICAL - {hist_time.strftime('%I:%M %p')}]" if is_historical else "[LIVE]"
    st.subheader(f"NYZTrade - {symbol} Hedging Pressure Index {mode_text}")
    
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
    
    st.info("💡 **Hedging Pressure**: +100% = Max support | -100% = High volatility zone")

# TAB 4: Data Table
with tab4:
    st.subheader("Strike-wise Analysis")
    
    if is_historical:
        st.caption(f"📜 Historical data from {hist_time.strftime('%I:%M:%S %p IST')}")
    
    display_cols = ['Strike', 'Call_OI', 'Put_OI', 'Net_GEX_B', 'Net_DEX_B', 'Hedging_Pressure', 'Total_Volume']
    display_df = df[display_cols].copy()
    
    for col in ['Call_OI', 'Put_OI', 'Total_Volume']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
    
    if 'Hedging_Pressure' in display_df.columns:
        display_df['Hedging_Pressure'] = display_df['Hedging_Pressure'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    csv = df.to_csv(index=False)
    timestamp = hist_time.strftime('%Y%m%d_%H%M') if is_historical else get_ist_time().strftime('%Y%m%d_%H%M')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"NYZTrade_{symbol}_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True
    )

# TAB 5: Strategies
with tab5:
    st.subheader("💡 Trading Strategies")
    
    if is_historical:
        st.info(f"📜 Strategies based on historical data from {hist_time.strftime('%I:%M %p IST')}")
    
    if flow_metrics and atm_info:
        gex_bias_val = flow_metrics['gex_near_total']
        dex_bias_val = flow_metrics['dex_near_total']
        
        st.markdown("### 📊 Current Market Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("GEX Flow", f"{gex_bias_val:.2f}")
            st.metric("DEX Flow", f"{dex_bias_val:.2f}")
        with col2:
            st.metric("ATM Strike", f"{atm_info['atm_strike']}")
            st.metric("Straddle Premium", f"₹{atm_info['atm_straddle_premium']:.2f}")
        
        st.markdown("---")
        
        # Strong Positive GEX
        if gex_bias_val > 50:
            st.success("### 🟢 Strong Positive GEX - Sideways/Bullish")
            
            st.markdown("#### Strategy 1: Iron Condor")
            st.code(f"""
Sell {symbol} {int(futures_ltp)} CE
Buy  {symbol} {int(futures_ltp + 200)} CE
Sell {symbol} {int(futures_ltp)} PE
Buy  {symbol} {int(futures_ltp - 200)} PE

Max Profit: Premium collected
Risk: MODERATE
Best: Price stays {int(futures_ltp - 100)} to {int(futures_ltp + 100)}
            """)
            
            st.markdown("#### Strategy 2: Short Straddle")
            st.code(f"""
Sell {symbol} {atm_info['atm_strike']} CE + PE

Premium: ₹{atm_info['atm_straddle_premium']:.2f}
Risk: HIGH - Use stops
Exit if price moves ₹{atm_info['atm_straddle_premium']*0.5:.2f}
            """)
        
        # Negative GEX
        elif gex_bias_val < -50:
            st.error("### 🔴 Negative GEX - High Volatility")
            
            st.markdown("#### Strategy: Long Straddle")
            st.code(f"""
Buy {symbol} {atm_info['atm_strike']} CE + PE

Cost: ₹{atm_info['atm_straddle_premium']:.2f}
Upper BE: {atm_info['atm_strike'] + atm_info['atm_straddle_premium']:.0f}
Lower BE: {atm_info['atm_strike'] - atm_info['atm_straddle_premium']:.0f}
Risk: HIGH - Needs big move
            """)
        
        # Neutral
        else:
            st.warning("### ⚖️ Neutral/Mixed Signals")
            
            if dex_bias_val > 20:
                st.markdown("#### Bull Call Spread")
                st.code(f"""
Buy  {symbol} {int(futures_ltp)} CE
Sell {symbol} {int(futures_ltp + 100)} CE
Risk: MODERATE
                """)
            elif dex_bias_val < -20:
                st.markdown("#### Bear Put Spread")
                st.code(f"""
Buy  {symbol} {int(futures_ltp)} PE
Sell {symbol} {int(futures_ltp - 100)} PE
Risk: MODERATE
                """)
            else:
                st.info("⏸️ **Wait for Clarity** - Mixed signals, stay cautious")
        
        st.markdown("---")
        st.markdown("### ⚠️ Risk Rules")
        st.markdown("""
1. Max 2% capital per trade
2. Always use stops
3. Monitor theta decay
4. Take profit at 50-70% max
5. Avoid tight stops near gamma flip zones
        """)
        
        if user_tier != "premium":
            st.info("🔒 Premium: Backtested parameters coming soon")
    
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
        st.warning(f"📜 Historical: {hist_time.strftime('%I:%M %p')}")
    else:
        st.success(f"🔴 Live: {symbol}")

with col4:
    if gamma_flip_zones:
        st.warning(f"⚡ {len(gamma_flip_zones)} Flip(s)")
    else:
        st.success("✅ No Flips")

st.markdown(f"**💡 NYZTrade YouTube | Data: {fetch_method} | Snapshots: {len(st.session_state.snapshot_times)}**")

# ============================================================================
# AUTO-REFRESH
# ============================================================================

if auto_refresh and user_tier == "premium" and st.session_state.is_live_mode:
    elapsed = time.time() - st.session_state.countdown_start
    if elapsed >= refresh_interval:
        st.session_state.countdown_start = time.time()
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()
