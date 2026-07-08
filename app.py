import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Page Config: Professional layout
st.set_page_config(page_title="Advanced Indian Equity Screener & Valuation Model", layout="wide")
st.title("Indian Equity Valuation Platform")

# --- DATA FUNCTIONS ---

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

# FIX 1: Custom clean text for the loading spinner
@st.cache_data(show_spinner="Fetching live market data from Yahoo Finance...") 
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
                financials = stock.financials
                yearly_growth = None
                if financials is not None and not financials.empty and 'Net Income' in financials.index:
                    net_income = financials.loc['Net Income']
                    if len(net_income) >= 2:
                        yearly_growth = ((net_income.iloc[0] - net_income.iloc[1]) / net_income.iloc[1]) * 100
                
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
            
    df = pd.DataFrame(data)
    
    # CRITICAL FIX: Clean data before it hits the cache
    if not df.empty:
        df["TTM_PE"] = pd.to_numeric(df["TTM_PE"], errors='coerce')
        df["ROE_Pct"] = pd.to_numeric(df["ROE_Pct"], errors='coerce')
        df["Yearly_Profit_Growth_Pct"] = pd.to_numeric(df["Yearly_Profit_Growth_Pct"], errors='coerce')
        df = df.dropna(subset=["TTM_PE", "ROE_Pct"])
        
    return df

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["📊 Valuation Matrix & ML Engine", "⚡ Live Market Screener"])

# ==========================================
# TAB 1: ADVANCED VALUATION & ML
# ==========================================
with tab1:
    st.markdown("### Cross-Sectional Analysis (Historical & Forward Estimates)")
    
    df_mock = fetch_mock_data()
    
    # Collapsible Filters
    with st.expander("⚙️ Advanced Filters", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            min_hist_roe = st.slider("Min Historical ROE (%)", 0.0, 40.0, 15.0, key="t1_roe")
        with col_b:
            min_growth_5y = st.slider("Min 5Y Hist Growth (%)", 0.0, 30.0, 10.0, key="t1_growth")
        with col_c:
            max_ttm_pe = st.slider("Max TTM P/E", 10.0, 150.0, 100.0, key="t1_pe")

    # Explicit condition filtering
    filtered_mock = df_mock[
        (df_mock["Historical_ROE"] >= min_hist_roe) &
        (df_mock["Growth_5Y"] >= min_growth_5y) &
        (df_mock["TTM_PE"] <= max_ttm_pe)
    ]

    if not filtered_mock.empty:
        fig1 = px.scatter(
            filtered_mock, x="Growth_5Y", y="Forward_ROE", size="Market_Cap_Cr", color="Forward_PE",
            hover_name="Ticker", 
            color_continuous_scale=["#0284c7", "#991b1b"], 
            title="Quality vs Valuation Matrix"
        )
        # Minimalist white background with crisp blue formatting
        fig1.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"),
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", tickfont=dict(color="#1e3a8a", size=13)),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", tickfont=dict(color="#1e3a8a", size=13))
        )
        fig1.update_traces(marker=dict(line=dict(width=1.5, color="#1e3a8a")))
        
        # Override Streamlit's theme to maintain absolute color visibility
        st.plotly_chart(fig1, use_container_width=True, theme=None)
    else:
        st.warning("No stocks match the criteria.")

    st.markdown("---")
    
    # Collapsible ML Engine
    with st.expander("🤖 Valuation Engine: Forward P/E Predictor", expanded=False):
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
        else:
            st.info("Not enough data points to train the model. Broaden your filter criteria.")

# ==========================================
# TAB 2: LIVE MARKET SCREENER
# ==========================================
# ==========================================
# TAB 2: LIVE MARKET SCREENER
# ==========================================
# ==========================================
# TAB 2: LIVE MARKET SCREENER (Bulletproof Filter)
# ==========================================
with tab2:
    st.markdown("### Live Market Data Pull (Yahoo Finance)")
    
    default_tickers = "ITC, TCS, RELIANCE, INFY, HDFCBANK, ZOMATO, TATAMOTORS, ADANIENT, JSWSTEEL, ASIANPAINT, ETERNAL, ADANIGREEN, CGPOWER"
    ticker_input = st.text_area("Enter NSE Tickers (comma separated):", value=default_tickers)
    min_mcap_input = st.number_input("Minimum Market Cap (in Crores):", value=1000, step=500)
    
    # UI Layout for Buttons
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        fetch_clicked = st.button("Fetch Live Data", type="primary")
    with col_btn2:
        if st.button("🔄 Force Refresh (Clear Cache)"):
            st.cache_data.clear()
            if 'live_df' in st.session_state:
                del st.session_state['live_df']
            st.rerun()
            
    if fetch_clicked:
        ticker_list = [t.strip() for t in ticker_input.split(",")]
        st.session_state['live_df'] = pull_live_market_data(ticker_list, min_mcap_cr=min_mcap_input)
    
    if 'live_df' in st.session_state and not st.session_state['live_df'].empty:
        df = st.session_state['live_df'].copy()
        
        # 1. ABSOLUTE FORCE TO NUMERIC (Coerce turns any rogue strings into NaN)
        df["TTM_PE"] = pd.to_numeric(df["TTM_PE"], errors='coerce')
        df["ROE_Pct"] = pd.to_numeric(df["ROE_Pct"], errors='coerce')
        df["Yearly_Profit_Growth_Pct"] = pd.to_numeric(df["Yearly_Profit_Growth_Pct"], errors='coerce')
        
        # Drop rows where P/E or ROE became NaN
        df = df.dropna(subset=["TTM_PE", "ROE_Pct"])
        
        # Collapsible Live Data Filters
        with st.expander("⚙️ Screen & Filter Live Data", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                # Using the true min/max of the cleaned numeric column
                min_pe, max_pe = st.slider(
                    "P/E Ratio", 
                    float(df["TTM_PE"].min()), 
                    float(df["TTM_PE"].max()), 
                    (float(df["TTM_PE"].min()), float(df["TTM_PE"].max())), 
                    key="live_pe"
                )
            with col2:
                min_roe, max_roe = st.slider(
                    "ROE (%)", 
                    float(df["ROE_Pct"].min()), 
                    float(df["ROE_Pct"].max()), 
                    (float(df["ROE_Pct"].min()), float(df["ROE_Pct"].max())), 
                    key="live_roe"
                )
            with col3:
                growth_data = df["Yearly_Profit_Growth_Pct"].dropna()
                if not growth_data.empty:
                    min_growth, max_growth = st.slider(
                        "Yearly Profit Growth (%)", 
                        float(growth_data.min()), 
                        float(growth_data.max()), 
                        (float(growth_data.min()), float(growth_data.max())), 
                        key="live_growth"
                    )
                else:
                    min_growth, max_growth = -100.0, 100.0 

        # 2. STEP-BY-STEP FILTERING (This prevents logical operators from failing)
        filtered_df = df.copy()
        
        # Filter PE
        filtered_df = filtered_df[filtered_df["TTM_PE"] >= float(min_pe)]
        filtered_df = filtered_df[filtered_df["TTM_PE"] <= float(max_pe)]
        
        # Filter ROE
        filtered_df = filtered_df[filtered_df["ROE_Pct"] >= float(min_roe)]
        filtered_df = filtered_df[filtered_df["ROE_Pct"] <= float(max_roe)]
        
        # Filter Growth (Allow NaNs to stay if they exist, or filter if numeric)
        filtered_df = filtered_df[
            ((filtered_df["Yearly_Profit_Growth_Pct"] >= float(min_growth)) & 
             (filtered_df["Yearly_Profit_Growth_Pct"] <= float(max_growth))) | 
             filtered_df["Yearly_Profit_Growth_Pct"].isna()
        ]
        
        if not filtered_df.empty:
            # High-Contrast 3-Color Scale Chart
            fig2 = px.scatter(
                filtered_df, 
                x="Yearly_Profit_Growth_Pct", 
                y="ROE_Pct", 
                size="Market_Cap_Cr", 
                color="TTM_PE", 
                hover_name="Ticker",
                color_continuous_scale=[
                    [0.0, "#0284c7"],  
                    [0.45, "#0284c7"], 
                    [0.5, "#94a3b8"],  
                    [0.55, "#dc2626"], 
                    [1.0, "#dc2626"]   
                ],
                range_color=[float(filtered_df["TTM_PE"].min()), float(filtered_df["TTM_PE"].max())],
                title=f"Filtered Matrix: {len(filtered_df)} Stocks"
            )
            
            fig2.update_layout(
                height=700, 
                plot_bgcolor="white", 
                paper_bgcolor="white", 
                font=dict(color="#1e3a8a"),
                xaxis=dict(
                    showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1",
                    tickfont=dict(color="#1e3a8a", size=13),
                    title_font=dict(color="#1e3a8a", size=15)
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1",
                    tickfont=dict(color="#1e3a8a", size=13),
                    title_font=dict(color="#1e3a8a", size=15)
                )
            )
            fig2.update_traces(marker=dict(line=dict(width=1.5, color="#1e3a8a")))
            
            st.plotly_chart(fig2, use_container_width=True, theme=None)
            
            # Collapsible Data Table (Using absolute display values)
            with st.expander("📋 View Screened Data Table", expanded=True):
                st.dataframe(
                    filtered_df.style.format({
                        "Market_Cap_Cr": "{:,.0f} Cr", 
                        "TTM_PE": "{:.2f}x", 
                        "ROE_Pct": "{:.2f}%",
                        "Yearly_Profit_Growth_Pct": "{:.2f}%"
                    }), 
                    use_container_width=True
                )
        else:
            st.warning("No stocks match the selected filter criteria.")
