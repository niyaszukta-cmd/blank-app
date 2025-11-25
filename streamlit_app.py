import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="NYZTrade Dashboard", layout="wide")

st.title("NYZTrade - GEX Dashboard")
st.write("Real-time Gamma Exposure Analysis for Indian Markets")

st.sidebar.header("Dashboard Settings")
symbol = st.sidebar.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"])

if st.sidebar.button("Refresh Data"):
    st.rerun()

st.subheader(f"{symbol} - Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Total GEX", "Rs 1.2 Cr", "+15%")
col2.metric("Call GEX", "Rs 0.8 Cr", "+10%")
col3.metric("Put GEX", "Rs 0.4 Cr", "-5%")

st.write("---")
st.subheader(f"{symbol} GEX Distribution")

strikes = [19000, 19100, 19200, 19300, 19400, 19500]
call_gex = [50000, 75000, 100000, 125000, 80000, 60000]
put_gex = [-40000, -60000, -90000, -70000, -50000, -30000]

fig = go.Figure()
fig.add_trace(go.Bar(x=strikes, y=call_gex, name='Call GEX', marker_color='green'))
fig.add_trace(go.Bar(x=strikes, y=put_gex, name='Put GEX', marker_color='red'))
fig.update_layout(barmode='relative', height=400, xaxis_title="Strike Price", yaxis_title="GEX")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Strike Data Table")
df = pd.DataFrame({
    'Strike': strikes,
    'Call GEX': call_gex,
    'Put GEX': put_gex,
    'Net GEX': [c + p for c, p in zip(call_gex, put_gex)]
})

st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, f"{symbol}_gex_data.csv", "text/csv")

st.write("---")
st.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write("Subscribe to NYZTrade on YouTube for trading analytics!")
