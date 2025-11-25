import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import hashlib
import hmac

# Try importing calculator (we'll handle errors gracefully)
try:
    from gex_calculator import EnhancedGEXDEXCalculator, calculate_dual_gex_dex_flow, detect_gamma_flip_zones
    CALCULATOR_AVAILABLE = True
except Exception as e:
    CALCULATOR_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ============================================================================
# AUTHENTICATION FUNCTIONS (Built-in)
# ============================================================================

def check_password():
    """Returns True if user has entered correct password"""
    
    def password_entered():
        """Checks whether a password entered by user is correct"""
        username = st.session_state["username"].strip().lower()
        password = st.session_state["password"]
        
        # Hardcoded users (you can change these)
        users = {
            "demo": "demo123",
            "premium": "premium123",
            "niyas": "nyztrade123"  # Add your own username/password
        }
        
        if username in users and password == users[username]:
            st.session_state["password_correct"] = True
            st.session_state["authenticated_user"] = username
            del st.session_state["password"]
            return
        
        st.session_state["password_correct"] = False
        st.session_state["authenticated_user"] = None
    
    # First run
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
    
    # Password incorrect
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
    
    # Password correct
    return True

def get_user_tier():
    """Get user tier"""
    if "authenticated_user" not in st.session_state:
        return "guest"
    
    username = st.session_state["authenticated_user"]
    
    # Premium users list
    premium_users = ["premium", "niyas"]
    
    return "premium" if username in premium_users else "basic"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NYZTrade - GEX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CHECK AUTHENTICATION
# ============================================================================

if not check_password():
    st.stop()

user_tier = get_user_tier()

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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">📊 NYZTrade - Advanced GEX + DEX Analysis</p>', unsafe_allow_html=True)
st.markdown("**Real-time Gamma & Delta Exposure Analysis for Indian Markets**")

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

# Auto-refresh (Premium feature)
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
else:
    st.sidebar.info("🔒 Auto-refresh: Premium only")
    auto_refresh = False
    refresh_interval = 60

# Manual refresh
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol, strikes_range, expiry_index):
    """Fetch and calculate GEX/DEX data"""
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
# MAIN ANALYSIS
# ============================================================================

st.markdown("---")

# Fetch data
with st.spinner(f"🔄 Fetching live {symbol} data..."):
    df, futures_ltp, fetch_method, atm_info, error = fetch_data(symbol, strikes_range, expiry_index)

if error:
    st.error(f"❌ Error: {error}")
    st.info("""
    **Troubleshooting:**
    1. Make sure `gex_calculator.py` is uploaded to GitHub
    2. Check that `requirements.txt` includes: `streamlit pandas numpy plotly scipy requests`
    3. Wait 1-2 minutes for Streamlit to install dependencies
    """)
    st.stop()

if df is None:
    st.error("❌ Failed to fetch data")
    st.stop()

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

try:
    flow_metrics = calculate_dual_gex_dex_flow(df, futures_ltp)
    
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
        
except Exception as e:
    flow_metrics = None
    st.warning(f"Flow metrics unavailable: {e}")

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

tab1, tab2, tab3, tab4 = st.tabs(["📊 GEX Profile", "📈 DEX Profile", "📋 Data Table", "💡 Strategies"])

# GEX Chart
with tab1:
    st.subheader(f"NYZTrade - {symbol} Gamma Exposure Profile")
    
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
    
    # Add gamma flip zones
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
    
    # Interpretation
    if total_gex > 0.5:
        st.success("🟢 **Strong Positive GEX**: Market expected to be sideways to bullish. Consider selling premium strategies.")
    elif total_gex < -0.5:
        st.error("🔴 **Negative GEX**: High volatility expected. Consider buying volatility strategies.")
    else:
        st.warning("⚖️ **Neutral GEX**: Mixed signals. Follow DEX for direction.")

# DEX Chart
with tab2:
    st.subheader(f"NYZTrade - {symbol} Delta Exposure Profile")
    
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

# Data Table
with tab3:
    st.subheader("Strike-wise Analysis")
    
    display_cols = ['Strike', 'Call_OI', 'Put_OI', 'Net_GEX_B', 'Net_DEX_B', 'Total_Volume']
    display_df = df[display_cols].copy()
    
    for col in ['Call_OI', 'Put_OI', 'Total_Volume']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"NYZTrade_{symbol}_GEX_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Strategies
with tab4:
    st.subheader("💡 Trading Strategies")
    
    if flow_metrics:
        gex_bias_val = flow_metrics['gex_near_total']
        
        if gex_bias_val > 50:
            st.success("### 🦅 Iron Condor Strategy")
            st.write("**Rationale**: Strong positive GEX indicates sideways market")
            st.code(f"Sell {symbol} {int(futures_ltp)} CE/PE + Buy wings")
            
        elif gex_bias_val < -50:
            st.error("### 🎭 Long Straddle Strategy")
            st.write("**Rationale**: Negative GEX indicates high volatility")
            if atm_info:
                st.code(f"Buy {symbol} {atm_info['atm_strike']} Straddle (₹{atm_info['atm_straddle_premium']:.2f})")
        else:
            st.info("### ⏸️ Wait for Clarity")
            st.write("Mixed signals - stay cautious")
    
    if user_tier != "premium":
        st.info("🔒 Detailed strategies available in Premium")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

with col2:
    st.info(f"📊 {symbol}")

with col3:
    st.info(f"🔧 {fetch_method}")

st.markdown("**💡 Subscribe to NYZTrade on YouTube!**")

# Auto-refresh
if auto_refresh and user_tier == "premium":
    time.sleep(refresh_interval)
    st.rerun()
