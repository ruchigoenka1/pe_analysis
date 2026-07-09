import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Advanced Indian Equity Screener & Valuation Model", layout="wide")
st.title("Indian Equity Valuation Platform")

# Professional minimal styling for new charts (Blue/White theme)
CHART_THEME = {
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'font': {'color': '#1e3a8a'},
    'xaxis': {'showgrid': True, 'gridcolor': '#e2e8f0', 'zerolinecolor': '#cbd5e1'},
    'yaxis': {'showgrid': True, 'gridcolor': '#e2e8f0', 'zerolinecolor': '#cbd5e1'}
}

# ==========================================
# DATA FUNCTIONS
# ==========================================

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
                    if len(net_income) >= 2 and net_income.iloc[1] != 0:
                        yearly_growth = ((net_income.iloc[0] - net_income.iloc[1]) / net_income.iloc[1]) * 100
                
                roe = info.get('returnOnEquity', None)
                beta = info.get('beta', 1.0)
                de_ratio = info.get('debtToEquity', 0) / 100 
                ev_ebitda = info.get('enterpriseToEbitda', None)
                
                # New fields for Value Creation Math
                total_debt = info.get('totalDebt', 0)
                ebitda = info.get('ebitda', 0)
                price_to_book = info.get('priceToBook', 1)
                ebit_growth = info.get('earningsQuarterlyGrowth', 0) * 100
                
                # Unlevered Beta Calculation
                unlevered_beta = beta / (1 + (1 - 0.25) * de_ratio) if beta else None

                data.append({
                    "Ticker": clean_ticker,
                    "Market_Cap_Cr": round(mcap / 10000000, 2),
                    "Total_Debt_Cr": round(total_debt / 10000000, 2),
                    "EBITDA_Cr": round(ebitda / 10000000, 2),
                    "Price_to_Book": price_to_book,
                    "TTM_PE": info.get('trailingPE', None),
                    "ROE_Pct": round(roe * 100, 2) if roe else None,
                    "Yearly_Profit_Growth_Pct": round(yearly_growth, 2) if yearly_growth is not None else None,
                    "EBIT_Growth_Pct": round(ebit_growth, 2) if ebit_growth else 0,
                    "Beta": round(beta, 2) if beta else 1.0,
                    "Unlevered_Beta": round(unlevered_beta, 2) if unlevered_beta else None,
                    "EV_EBITDA": round(ev_ebitda, 2) if ev_ebitda else None
                })
        except Exception:
            continue
            
    return pd.DataFrame(data)

# ==========================================
# TABS LAYOUT
# ==========================================
tab1, tab2 = st.tabs(["📊 Valuation Matrix & ML Engine", "⚡ Live Market Screener & Value Creation"])

# ==========================================
# TAB 1: ADVANCED VALUATION & ML
# ==========================================
with tab1:
    st.markdown("### Cross-Sectional Analysis (Historical & Forward Estimates)")
    
    df_mock = fetch_mock_data()
    
    with st.expander("⚙️ Advanced Filters", expanded=True):
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
            hover_name="Ticker", 
            color_continuous_scale=["#0284c7", "#991b1b"], 
            title="Quality vs Valuation Matrix"
        )
        fig1.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"),
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", tickfont=dict(color="#1e3a8a", size=13)),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", tickfont=dict(color="#1e3a8a", size=13))
        )
        fig1.update_traces(marker=dict(line=dict(width=1.5, color="#1e3a8a")))
        
        st.plotly_chart(fig1, use_container_width=True, theme=None)
    else:
        st.warning("No stocks match the criteria.")

    st.markdown("---")
    
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
# TAB 2: LIVE MARKET SCREENER & SPREAD
# ==========================================
with tab2:
    st.markdown("### Live Market Data Pull (Yahoo Finance)")
    
    default_tickers = "ITC, TCS, RELIANCE, INFY, HDFCBANK, ZOMATO, TATAMOTORS, ADANIENT, JSWSTEEL, ASIANPAINT, ETERNAL, ADANIGREEN, CGPOWER"
    ticker_input = st.text_area("Enter NSE Tickers (comma separated):", value=default_tickers)
    min_mcap_input = st.number_input("Minimum Market Cap (in Crores):", value=1000, step=500)
    
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
        
        # 1. Absolute Force to Numeric
        num_cols = ["TTM_PE", "ROE_Pct", "Yearly_Profit_Growth_Pct", "Beta", "Unlevered_Beta", "EV_EBITDA", "EBIT_Growth_Pct"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=["TTM_PE", "ROE_Pct"])

        # --- DYNAMIC WACC & ROCE CALCULATIONS (NEW) ---
        with st.expander("⚖️ Macro Assumptions & Cost of Capital (CAPM)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            risk_free_rate = c1.number_input("Risk-Free Rate (%)", value=7.0, step=0.1)
            erp = c2.number_input("Equity Risk Premium (%)", value=5.5, step=0.1)
            cost_of_debt = c3.number_input("Pre-Tax Cost of Debt (%)", value=8.5, step=0.1)
            tax_rate = c4.number_input("Corporate Tax Rate (%)", value=25.0, step=1.0) / 100

        # Math Logic for new features
        df["EBIT_Cr"] = df["EBITDA_Cr"] * 0.9
        df["Book_Equity_Cr"] = np.where(df["Price_to_Book"] > 0, df["Market_Cap_Cr"] / df["Price_to_Book"], df["Market_Cap_Cr"])
        df["Capital_Employed_Cr"] = df["Total_Debt_Cr"] + df["Book_Equity_Cr"]
        
        df["ROCE_Pct"] = np.where(df["Capital_Employed_Cr"] > 0, (df["EBIT_Cr"] / df["Capital_Employed_Cr"]) * 100, 0)
        df["Ke_Pct"] = risk_free_rate + (df["Beta"].fillna(1.0) * erp)
        df["Weight_Equity"] = df["Market_Cap_Cr"] / (df["Market_Cap_Cr"] + df["Total_Debt_Cr"])
        df["Weight_Debt"] = df["Total_Debt_Cr"] / (df["Market_Cap_Cr"] + df["Total_Debt_Cr"])
        
        df["WACC_Pct"] = (df["Weight_Equity"] * df["Ke_Pct"]) + (df["Weight_Debt"] * cost_of_debt * (1 - tax_rate))
        df["ROCE_WACC_Spread"] = df["ROCE_Pct"] - df["WACC_Pct"]

        # 2. Original Filters
        with st.expander("⚙️ Screen & Filter Live Data", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                min_pe, max_pe = st.slider("P/E Ratio", float(df["TTM_PE"].min()), float(df["TTM_PE"].max()), (float(df["TTM_PE"].min()), float(df["TTM_PE"].max())), key="live_pe")
            with col2:
                min_roe, max_roe = st.slider("ROE (%)", float(df["ROE_Pct"].min()), float(df["ROE_Pct"].max()), (float(df["ROE_Pct"].min()), float(df["ROE_Pct"].max())), key="live_roe")
            with col3:
                growth_data = df["Yearly_Profit_Growth_Pct"].dropna()
                if not growth_data.empty:
                    min_growth, max_growth = st.slider("Yearly Profit Growth (%)", float(growth_data.min()), float(growth_data.max()), (float(growth_data.min()), float(growth_data.max())), key="live_growth")
                else:
                    min_growth, max_growth = -100.0, 100.0 

        # Step-by-Step Filtering
        filtered_df = df.copy()
        filtered_df = filtered_df[(filtered_df["TTM_PE"] >= min_pe) & (filtered_df["TTM_PE"] <= max_pe)]
        filtered_df = filtered_df[(filtered_df["ROE_Pct"] >= min_roe) & (filtered_df["ROE_Pct"] <= max_roe)]
        filtered_df = filtered_df[
            ((filtered_df["Yearly_Profit_Growth_Pct"] >= min_growth) & 
             (filtered_df["Yearly_Profit_Growth_Pct"] <= max_growth)) | 
             filtered_df["Yearly_Profit_Growth_Pct"].isna()
        ]
        
        if not filtered_df.empty:
            
            # ---------------------------------------------------------
            # ORIGINAL GRAPH 1: Heatmap
            # ---------------------------------------------------------
            fig2 = px.scatter(
                filtered_df, x="Yearly_Profit_Growth_Pct", y="ROE_Pct", size="Market_Cap_Cr", color="TTM_PE", hover_name="Ticker",
                color_continuous_scale=[
                    [0.0, "#1e3a8a"], [0.2, "#0ea5e9"], [0.4, "#93c5fd"], [0.5, "#e2e8f0"],
                    [0.6, "#fca5a5"], [0.8, "#ef4444"], [1.0, "#7f1d1d"]
                ],
                range_color=[float(filtered_df["TTM_PE"].min()), float(filtered_df["TTM_PE"].max())],
                title=f"Filtered Matrix: {len(filtered_df)} Stocks"
            )
            fig2.update_layout(
                height=700, plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"),
                xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
                yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
            )
            fig2.update_traces(marker=dict(line=dict(width=1.5, color="#1e3a8a")))
            st.plotly_chart(fig2, use_container_width=True, theme=None)
            
            st.markdown("---")
            
            # ---------------------------------------------------------
            # ORIGINAL GRAPH 2: Operational Risk Toggle (Beta)
            # ---------------------------------------------------------
            show_risk_plot = st.checkbox("🔍 View Operational Risk (Unlevered Beta Comparison)")
            if show_risk_plot:
                risk_df = filtered_df.dropna(subset=["Beta", "Unlevered_Beta"]).sort_values("Unlevered_Beta")
                if not risk_df.empty:
                    melted_risk = risk_df.melt(id_vars=["Ticker"], value_vars=["Beta", "Unlevered_Beta"], var_name="Risk_Metric", value_name="Value")
                    fig_risk = px.bar(
                        melted_risk, x="Ticker", y="Value", color="Risk_Metric", barmode="group",
                        title="Levered Beta vs. Unlevered Beta (Operational Risk)",
                        color_discrete_map={"Beta": "#94a3b8", "Unlevered_Beta": "#0284c7"}
                    )
                    fig_risk.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"),
                        xaxis=dict(showgrid=False, zerolinecolor="#cbd5e1", title=""),
                        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", title="Beta Value")
                    )
                    st.plotly_chart(fig_risk, use_container_width=True, theme=None)
                else:
                    st.info("Beta data is not available.")
            
            # ---------------------------------------------------------
            # ORIGINAL GRAPH 3: Unlevered Valuation Toggle (EV/EBITDA)
            # ---------------------------------------------------------
            show_valuation_plot = st.checkbox("📊 View Unlevered Valuation (EV/EBITDA vs P/E)")
            if show_valuation_plot:
                val_df = filtered_df.dropna(subset=["TTM_PE", "EV_EBITDA"]).sort_values("TTM_PE")
                if not val_df.empty:
                    melted_val = val_df.melt(id_vars=["Ticker"], value_vars=["TTM_PE", "EV_EBITDA"], var_name="Valuation_Metric", value_name="Multiple")
                    fig_val = px.bar(
                        melted_val, x="Ticker", y="Multiple", color="Valuation_Metric", barmode="group",
                        title="Levered (P/E) vs. Unlevered (EV/EBITDA) Valuation",
                        color_discrete_map={"TTM_PE": "#94a3b8", "EV_EBITDA": "#0284c7"}
                    )
                    fig_val.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"),
                        xaxis=dict(showgrid=False, zerolinecolor="#cbd5e1", title=""),
                        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", title="Multiple (x)")
                    )
                    st.plotly_chart(fig_val, use_container_width=True, theme=None)
                else:
                    st.info("EV/EBITDA data is not available.")
            
            st.markdown("---")

            # ---------------------------------------------------------
            # NEW GRAPHS: Advanced Value Creation Analysis (Hidden behind toggle)
            # ---------------------------------------------------------
            show_advanced_charts = st.checkbox("📈 View Advanced Value Creation Charts (P/E vs ROE, Spread vs EV/EBITDA)")
            if show_advanced_charts:
                st.markdown("#### 1. Equity Valuation: P/E vs. ROE")
                fig_pe_roe = px.scatter(
                    filtered_df, x="ROE_Pct", y="TTM_PE", 
                    size=filtered_df["Yearly_Profit_Growth_Pct"].clip(lower=1), 
                    color="Yearly_Profit_Growth_Pct",
                    hover_name="Ticker", text="Ticker", color_continuous_scale="Blues",
                    title="P/E relative to ROE (Bubble Size = Net Income Growth)"
                )
                fig_pe_roe.update_traces(textposition='top center', marker=dict(line=dict(width=1, color="#0D47A1")))
                fig_pe_roe.update_layout(**CHART_THEME)
                st.plotly_chart(fig_pe_roe, use_container_width=True, theme=None)

                st.markdown("#### 2. Unlevered Valuation: EV/EBITDA vs. ROCE")
                fig_ev_roce = px.scatter(
                    filtered_df, x="ROCE_Pct", y="EV_EBITDA", 
                    size=filtered_df["EBIT_Growth_Pct"].clip(lower=1), 
                    color="EBIT_Growth_Pct",
                    hover_name="Ticker", text="Ticker", color_continuous_scale="Blues",
                    title="EV/EBITDA relative to ROCE (Bubble Size = EBIT Growth)"
                )
                fig_ev_roce.update_traces(textposition='top center', marker=dict(line=dict(width=1, color="#0D47A1")))
                fig_ev_roce.update_layout(**CHART_THEME)
                st.plotly_chart(fig_ev_roce, use_container_width=True, theme=None)

                st.markdown("#### 3. True Value Creation: ROCE vs WACC Spread")
                fig_spread = px.scatter(
                    filtered_df, x="ROCE_WACC_Spread", y="EV_EBITDA", 
                    size="Market_Cap_Cr", hover_name="Ticker", text="Ticker",
                    title="EV/EBITDA vs. Value Creation Spread (Bubble Size = Market Cap)"
                )
                fig_spread.update_traces(textposition='top center', marker=dict(color="#1E88E5", line=dict(width=1.5, color="#0D47A1")))
                fig_spread.add_vline(x=0, line_width=2, line_dash="dash", line_color="red", annotation_text="Value Destruction Zone", annotation_position="top left")
                fig_spread.update_layout(**CHART_THEME)
                st.plotly_chart(fig_spread, use_container_width=True, theme=None)

            st.markdown("---")

            # ---------------------------------------------------------
            # ORIGINAL GRAPH 4: Collapsible Data Table
            # ---------------------------------------------------------
            with st.expander("📋 View Screened Data Table", expanded=True):
                st.dataframe(
                    filtered_df.style.format({
                        "Market_Cap_Cr": "{:,.0f} Cr", 
                        "TTM_PE": "{:.2f}x", 
                        "ROE_Pct": "{:.2f}%",
                        "Yearly_Profit_Growth_Pct": "{:.2f}%",
                        "Beta": "{:.2f}",
                        "Unlevered_Beta": "{:.2f}",
                        "EV_EBITDA": "{:.2f}x",
                        "ROCE_Pct": "{:.2f}%",
                        "WACC_Pct": "{:.2f}%",
                        "ROCE_WACC_Spread": "{:.2f}%"
                    }), 
                    use_container_width=True
                )
        else:
            st.warning("No stocks match the selected filter criteria.")
