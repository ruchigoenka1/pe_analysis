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

# Original Heatmap color scale
HEATMAP_COLORS = [
    [0.0, "#1e3a8a"], [0.2, "#0ea5e9"], [0.4, "#93c5fd"], [0.5, "#e2e8f0"],
    [0.6, "#fca5a5"], [0.8, "#ef4444"], [1.0, "#7f1d1d"]
]

# ==========================================
# DATA FUNCTIONS
# ==========================================

@st.cache_data
def fetch_mock_data():
    nifty_tickers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ITC", "LT", "ASIANPAINT", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN"]
    data = []
    np.random.seed(42)
    for ticker in nifty_tickers:
        hist_roe = np.random.uniform(8, 35)
        fwd_pe = np.random.uniform(15, 80)
        data.append({
            "Ticker": ticker, "Market_Cap_Cr": np.random.uniform(50000, 1500000), "Historical_ROE": round(hist_roe, 2),
            "Forward_ROE": round(hist_roe * np.random.uniform(0.9, 1.2), 2), "Growth_5Y": round(np.random.uniform(5, 25), 2),
            "TTM_PE": round(fwd_pe * np.random.uniform(0.8, 1.3), 2), "Forward_PE": round(fwd_pe, 2), "Leverage_DE": round(np.random.uniform(0.0, 2.5), 2)
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
                
                roe = info.get('returnOnEquity', 0)
                beta = info.get('beta', 1.0)
                de_ratio = info.get('debtToEquity', 0) / 100 
                ev_ebitda = info.get('enterpriseToEbitda', None)
                
                # Metrics for ROCE and WACC
                total_debt = info.get('totalDebt', 0)
                ebitda = info.get('ebitda', 0)
                price_to_book = info.get('priceToBook', 1)
                ebit_growth = info.get('earningsQuarterlyGrowth', 0) * 100
                ev = info.get('enterpriseValue', mcap + total_debt)
                
                unlevered_beta = beta / (1 + (1 - 0.25) * de_ratio) if beta else None

                data.append({
                    "Ticker": clean_ticker,
                    "Market_Cap_Cr": round(mcap / 10000000, 2),
                    "Enterprise_Value_Cr": round(ev / 10000000, 2),
                    "Total_Debt_Cr": round(total_debt / 10000000, 2),
                    "EBITDA_Cr": round(ebitda / 10000000, 2),
                    "Price_to_Book": price_to_book,
                    "TTM_PE": info.get('trailingPE', None),
                    "ROE_Pct": round(roe * 100, 2) if roe else None,
                    "Yearly_Profit_Growth_Pct": round(yearly_growth, 2) if yearly_growth is not None else None,
                    "EBIT_Growth_Pct": round(ebit_growth, 2) if ebit_growth else None,
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
        with col_a: min_hist_roe = st.slider("Min Historical ROE (%)", 0.0, 40.0, 15.0, key="t1_roe")
        with col_b: min_growth_5y = st.slider("Min 5Y Hist Growth (%)", 0.0, 30.0, 10.0, key="t1_growth")
        with col_c: max_ttm_pe = st.slider("Max TTM P/E", 10.0, 150.0, 100.0, key="t1_pe")

    filtered_mock = df_mock[(df_mock["Historical_ROE"] >= min_hist_roe) & (df_mock["Growth_5Y"] >= min_growth_5y) & (df_mock["TTM_PE"] <= max_ttm_pe)]

    if not filtered_mock.empty:
        fig1 = px.scatter(
            filtered_mock, x="Growth_5Y", y="Forward_ROE", size="Market_Cap_Cr", color="Forward_PE",
            hover_name="Ticker", color_continuous_scale=["#0284c7", "#991b1b"], title="Quality vs Valuation Matrix"
        )
        fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"))
        st.plotly_chart(fig1, use_container_width=True)

    with st.expander("🤖 Valuation Engine: Forward P/E Predictor", expanded=False):
        if len(filtered_mock) > 5:
            features = ["Historical_ROE", "Growth_5Y", "Leverage_DE"]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(filtered_mock[features], filtered_mock["Forward_PE"])
            
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
    default_tickers = "ITC, TCS, RELIANCE, INFY, HDFCBANK, ZOMATO, TATAMOTORS, ADANIENT, JSWSTEEL, ASIANPAINT"
    ticker_input = st.text_area("Enter NSE Tickers (comma separated):", value=default_tickers)
    min_mcap_input = st.number_input("Minimum Market Cap (in Crores):", value=1000, step=500)
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1: fetch_clicked = st.button("Fetch Live Data", type="primary")
    with col_btn2:
        if st.button("🔄 Force Refresh"):
            st.cache_data.clear()
            st.rerun()
            
    if fetch_clicked:
        ticker_list = [t.strip() for t in ticker_input.split(",")]
        st.session_state['live_df'] = pull_live_market_data(ticker_list, min_mcap_cr=min_mcap_input)
    
    if 'live_df' in st.session_state and not st.session_state['live_df'].empty:
        df = st.session_state['live_df'].copy()
        
        # Absolute Force to Numeric
        num_cols = ["TTM_PE", "ROE_Pct", "Yearly_Profit_Growth_Pct", "EBIT_Growth_Pct", "EV_EBITDA", "Beta", "Unlevered_Beta"]
        for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=["TTM_PE", "ROE_Pct"])

        # --- RESTORED: MACRO ASSUMPTIONS ---
        with st.expander("⚖️ Macro Assumptions & Cost of Capital", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            risk_free_rate = c1.number_input("Risk-Free Rate (%)", value=7.0, step=0.1)
            erp = c2.number_input("Equity Risk Premium (%)", value=5.5, step=0.1)
            cost_of_debt = c3.number_input("Pre-Tax Cost of Debt (%)", value=8.5, step=0.1)
            tax_rate = c4.number_input("Corporate Tax Rate (%)", value=25.0, step=1.0) / 100

        # --- RESTORED: LIVE DATA FILTERS ---
        with st.expander("⚙️ Screen & Filter Live Data", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1: min_pe, max_pe = st.slider("P/E Ratio", float(df["TTM_PE"].min()), float(df["TTM_PE"].max()), (float(df["TTM_PE"].min()), float(df["TTM_PE"].max())), key="live_pe")
            with col2: min_roe, max_roe = st.slider("ROE (%)", float(df["ROE_Pct"].min()), float(df["ROE_Pct"].max()), (float(df["ROE_Pct"].min()), float(df["ROE_Pct"].max())), key="live_roe")
            with col3:
                growth_data = df["Yearly_Profit_Growth_Pct"].dropna()
                if not growth_data.empty:
                    min_growth, max_growth = st.slider("Yearly Profit Growth (%)", float(growth_data.min()), float(growth_data.max()), (float(growth_data.min()), float(growth_data.max())), key="live_growth")
                else:
                    min_growth, max_growth = -100.0, 100.0 

        # Apply Filters
        filtered_df = df.copy()
        filtered_df = filtered_df[(filtered_df["TTM_PE"] >= min_pe) & (filtered_df["TTM_PE"] <= max_pe)]
        filtered_df = filtered_df[(filtered_df["ROE_Pct"] >= min_roe) & (filtered_df["ROE_Pct"] <= max_roe)]
        filtered_df = filtered_df[
            ((filtered_df["Yearly_Profit_Growth_Pct"] >= min_growth) & (filtered_df["Yearly_Profit_Growth_Pct"] <= max_growth)) | 
             filtered_df["Yearly_Profit_Growth_Pct"].isna()
        ]

        if not filtered_df.empty:
            
            # --- VALUE CREATION MATH (Applied to Filtered Data) ---
            filtered_df["EBIT_Cr"] = filtered_df["EBITDA_Cr"] * 0.9
            filtered_df["Book_Equity_Cr"] = np.where(filtered_df["Price_to_Book"] > 0, filtered_df["Market_Cap_Cr"] / filtered_df["Price_to_Book"], filtered_df["Market_Cap_Cr"])
            filtered_df["Capital_Employed_Cr"] = filtered_df["Total_Debt_Cr"] + filtered_df["Book_Equity_Cr"]
            
            filtered_df["ROCE_Pct"] = np.where(filtered_df["Capital_Employed_Cr"] > 0, (filtered_df["EBIT_Cr"] / filtered_df["Capital_Employed_Cr"]) * 100, 0)
            filtered_df["Ke_Pct"] = risk_free_rate + (filtered_df["Beta"].fillna(1.0) * erp)
            
            filtered_df["Weight_Equity"] = filtered_df["Market_Cap_Cr"] / (filtered_df["Market_Cap_Cr"] + filtered_df["Total_Debt_Cr"])
            filtered_df["Weight_Debt"] = filtered_df["Total_Debt_Cr"] / (filtered_df["Market_Cap_Cr"] + filtered_df["Total_Debt_Cr"])
            filtered_df["WACC_Pct"] = (filtered_df["Weight_Equity"] * filtered_df["Ke_Pct"]) + (filtered_df["Weight_Debt"] * cost_of_debt * (1 - tax_rate))
            
            filtered_df["ROE_Ke_Spread"] = filtered_df["ROE_Pct"] - filtered_df["Ke_Pct"]
            filtered_df["ROCE_WACC_Spread"] = filtered_df["ROCE_Pct"] - filtered_df["WACC_Pct"]

            filtered_df["Bubble_MCap"] = filtered_df["Market_Cap_Cr"].clip(lower=1)
            filtered_df["Bubble_EV"] = filtered_df["Enterprise_Value_Cr"].clip(lower=1)

            # ==========================================
            # THE 4 EXACT REQUESTED GRAPHS
            # ==========================================
            st.markdown("### 1. The Equity View (Market Cap & P/E Focus)")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig1 = px.scatter(
                    filtered_df, x="Yearly_Profit_Growth_Pct", y="ROE_Pct", size="Bubble_MCap", color="TTM_PE", 
                    hover_name="Ticker", color_continuous_scale=HEATMAP_COLORS, title="Profit Growth vs. ROE"
                )
                fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"))
                fig1.update_traces(marker=dict(line=dict(width=1, color="#1e3a8a")))
                st.plotly_chart(fig1, use_container_width=True)

            with col_g2:
                fig2 = px.scatter(
                    filtered_df, x="Yearly_Profit_Growth_Pct", y="ROE_Ke_Spread", size="Bubble_MCap", color="TTM_PE", 
                    hover_name="Ticker", color_continuous_scale=HEATMAP_COLORS, title="Profit Growth vs. Value Spread (ROE - Ke)"
                )
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"))
                fig2.update_traces(marker=dict(line=dict(width=1, color="#1e3a8a")))
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 2. The Enterprise View (Enterprise Value & EV/EBITDA Focus)")
            col_g3, col_g4 = st.columns(2)
            with col_g3:
                fig3 = px.scatter(
                    filtered_df, x="EBIT_Growth_Pct", y="ROCE_Pct", size="Bubble_EV", color="EV_EBITDA", 
                    hover_name="Ticker", color_continuous_scale=HEATMAP_COLORS, title="EBIT Growth vs. ROCE"
                )
                fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"))
                fig3.update_traces(marker=dict(line=dict(width=1, color="#1e3a8a")))
                st.plotly_chart(fig3, use_container_width=True)

            with col_g4:
                fig4 = px.scatter(
                    filtered_df, x="EBIT_Growth_Pct", y="ROCE_WACC_Spread", size="Bubble_EV", color="EV_EBITDA", 
                    hover_name="Ticker", color_continuous_scale=HEATMAP_COLORS, title="EBIT Growth vs. Value Spread (ROCE - WACC)"
                )
                fig4.add_hline(y=0, line_dash="dash", line_color="red")
                fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"))
                fig4.update_traces(marker=dict(line=dict(width=1, color="#1e3a8a")))
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown("---")

            # --- RESTORED: THE ORIGINAL BAR CHART TOGGLES ---
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
                    fig_risk.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"))
                    st.plotly_chart(fig_risk, use_container_width=True)
                else:
                    st.info("Beta data is not available.")

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
                    fig_val.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#1e3a8a"))
                    st.plotly_chart(fig_val, use_container_width=True)
                else:
                    st.info("EV/EBITDA data is not available.")
            
            st.markdown("---")

            with st.expander("📋 View Screened Data Table", expanded=True):
                st.dataframe(
                    filtered_df[["Ticker", "Market_Cap_Cr", "Enterprise_Value_Cr", "TTM_PE", "EV_EBITDA", "Yearly_Profit_Growth_Pct", "ROE_Pct", "ROE_Ke_Spread", "EBIT_Growth_Pct", "ROCE_Pct", "ROCE_WACC_Spread", "Beta", "Unlevered_Beta"]].style.format({
                        "Market_Cap_Cr": "{:,.0f} Cr", "Enterprise_Value_Cr": "{:,.0f} Cr", 
                        "TTM_PE": "{:.1f}x", "EV_EBITDA": "{:.1f}x",
                        "Yearly_Profit_Growth_Pct": "{:.1f}%", "ROE_Pct": "{:.1f}%", "ROE_Ke_Spread": "{:.1f}%",
                        "EBIT_Growth_Pct": "{:.1f}%", "ROCE_Pct": "{:.1f}%", "ROCE_WACC_Spread": "{:.1f}%",
                        "Beta": "{:.2f}", "Unlevered_Beta": "{:.2f}"
                    }), 
                    use_container_width=True
                )
        else:
            st.warning("No stocks match the selected filter criteria.")
