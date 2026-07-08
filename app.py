import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Page Config: Dark mode professional layout
st.set_page_config(page_title="Advanced Indian Equity Screener & Valuation Model", layout="wide")
st.title("Indian Equity Valuation Platform")

# --- DATA FUNCTIONS ---

# 1. Simulated Data (for Tab 1 Advanced Matrix)
@st.cache_data
def fetch_mock_data():
    nifty_tickers = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ITC", "LT", 
        "ASIANPAINT", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN"
    ]
    data = []
    np.random.seed(42)
    for ticker in nifty_tickers:
        hist_roe = np.random.uniform(8, 35)
        fwd_pe = np.random.uniform(15, 80)
        data.append({
            "Ticker": ticker,
            "Market_Cap_Cr": np.random.uniform(50000, 1500000),
            "Historical_ROE": round(hist_roe, 2),
            "Forward_ROE": round(hist_roe * np.random.uniform(0.9, 1.2), 2),
            "Growth_5Y": round(np.random.uniform(5, 25), 2),
            "TTM_PE": round(fwd_pe * np.random.uniform(0.8, 1.3), 2),
            "Forward_PE": round(fwd_pe, 2),
            "Leverage_DE": round(np.random.uniform(0.0, 2.5), 2)
        })
    return pd.DataFrame(data)

# 2. Live Data Fetcher (for Tab 2 Live Screener)
@st.cache_data(show_spinner=False)
def pull_live_market_data(ticker_list, min_mcap_cr=1000):
    data = []
    min_mcap_absolute = min_mcap_cr * 10000000 
    
    for ticker in ticker_list:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker: continue
        
        try:
            stock = yf.Ticker(f"{clean_ticker}.NS")
            info = stock.info
            mcap = info.get('marketCap', 0)
            
            if mcap >= min_mcap_absolute:
                # Fetch annual income statement
                financials = stock.financials
                if financials is not None and not financials.empty:
                    # Net Income is usually the first row in annual financials
                    net_income = financials.loc['Net Income']
                    # Calculate YoY Growth: (Current Year - Previous Year) / Previous Year
                    if len(net_income) >= 2:
                        yearly_growth = ((net_income.iloc[0] - net_income.iloc[1]) / net_income.iloc[1]) * 100
                    else:
                        yearly_growth = None
                else:
                    yearly_growth = None
                
                roe = info.get('returnOnEquity', None)
                data.append({
                    "Ticker": clean_ticker,
                    "Market_Cap_Cr": round(mcap / 10000000, 2),
                    "TTM_PE": info.get('trailingPE', None),
                    "ROE_Pct": round(roe * 100, 2) if roe else None,
                    "Yearly_Profit_Growth_Pct": round(yearly_growth, 2) if yearly_growth is not None else None
                })
        except Exception:
            continue
            
    return pd.DataFrame(data).dropna(subset=["TTM_PE", "ROE_Pct"])

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["📊 Valuation Matrix & ML Engine", "⚡ Live Market Screener"])

# ==========================================
# TAB 1: ADVANCED VALUATION & ML
# ==========================================
with tab1:
    st.markdown("### Cross-Sectional Analysis (Historical & Forward Estimates)")
    
    df_mock = fetch_mock_data()
    
    # Filters specific to Tab 1
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        min_hist_roe = st.slider("Min Historical ROE (%)", 0.0, 40.0, 15.0, key="t1_roe")
    with col_b:
        min_growth_5y = st.slider("Min 5Y Hist Growth (%)", 0.0, 30.0, 10.0, key="t1_growth")
    with col_c:
        max_ttm_pe = st.slider("Max TTM P/E", 10.0, 150.0, 100.0, key="t1_pe")

    filtered_mock = df_mock[
        (df_mock["Historical_ROE"] >= min_hist_roe) &
        (df_mock["Growth_5Y"] >= min_growth_5y) &
        (df_mock["TTM_PE"] <= max_ttm_pe)
    ]

    if not filtered_mock.empty:
        fig1 = px.scatter(
            filtered_mock, x="Growth_5Y", y="Forward_ROE", size="Market_Cap_Cr", color="Forward_PE",
            hover_name="Ticker", color_continuous_scale=["#1e40af", "#ef4444"], 
            title="Quality vs Valuation Matrix"
        )
        fig1.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#f8fafc"),
            xaxis=dict(showgrid=True, gridcolor="#334155", zerolinecolor="#64748b"),
            yaxis=dict(showgrid=True, gridcolor="#334155", zerolinecolor="#64748b")
        )
        fig1.update_traces(marker=dict(line=dict(width=1.5, color="#ffffff")))
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No stocks match the criteria.")

    st.markdown("---")
    st.subheader("Valuation Engine: Forward P/E Predictor")
    
    if len(filtered_mock) > 5:
        features = ["Historical_ROE", "Growth_5Y", "Leverage_DE"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(filtered_mock[features], filtered_mock["Forward_PE"])
        
        st.markdown("**Test a Hypothetical Company Profile:**")
        mc1, mc2, mc3 = st.columns(3)
        test_hist_roe = mc1.number_input("Historical ROE (%)", value=20.0)
        test_growth = mc2.number_input("5Y Growth (%)", value=15.0)
        test_lev = mc3.number_input("Leverage (D/E)", value=0.5)
            
        pred = model.predict([[test_hist_roe, test_growth, test_lev]])
        st.metric("Predicted Fair Forward P/E", f"{pred[0]:.2f}x")

# ==========================================
# TAB 2: LIVE MARKET SCREENER
# ==========================================
with tab2:
    st.markdown("### Live Market Data Pull (Yahoo Finance)")
    st.markdown("Fetches real-time Market Cap, TTM P/E, and ROE. Filters out companies below your Market Cap threshold.")
    
    # Input list of tickers
    default_tickers = "ITC, TCS, RELIANCE, INFY, HDFCBANK, ZOMATO, SUZLON, IDEA, TATAMOTORS, LICI"
    ticker_input = st.text_area("Enter NSE Tickers (comma separated):", value=default_tickers)
    min_mcap_input = st.number_input("Minimum Market Cap (in Crores):", value=1000, step=500)
    
    if st.button("Fetch Live Data", type="primary"):
        with st.spinner("Fetching data from Yahoo Finance..."):
            ticker_list = [t.strip() for t in ticker_input.split(",")]
            live_df = pull_live_market_data(ticker_list, min_mcap_cr=min_mcap_input)
            
            if not live_df.empty:
                st.success(f"Successfully pulled data for {len(live_df)} companies meeting the criteria.")
                
                # Dark theme plot for live data
                fig2 = px.scatter(
                    live_df, x="ROE_Pct", y="TTM_PE", size="Market_Cap_Cr", color="TTM_PE",
                    hover_name="Ticker", color_continuous_scale=["#1e40af", "#ef4444"], 
                    labels={"ROE_Pct": "Return on Equity (%)", "TTM_PE": "Trailing P/E Ratio"},
                    title=f"Live ROE vs P/E Matrix (Market Cap > {min_mcap_input} Cr)"
                )
                fig2.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font=dict(color="#f8fafc"),
                    xaxis=dict(showgrid=True, gridcolor="#334155", zerolinecolor="#64748b"),
                    yaxis=dict(showgrid=True, gridcolor="#334155", zerolinecolor="#64748b")
                )
                fig2.update_traces(marker=dict(line=dict(width=1.5, color="#ffffff")))
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Formatted DataFrame
                st.dataframe(
                    live_df.style.format({
                        "Market_Cap_Cr": "{:,.0f} Cr", 
                        "TTM_PE": "{:.2f}x", 
                        "ROE_Pct": "{:.2f}%"
                    }), 
                    use_container_width=True
                )
            else:
                st.error("No companies found or none met the >1000 Cr Market Cap threshold.")
