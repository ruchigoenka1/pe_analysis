import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Page Config: Set layout to wide
st.set_page_config(page_title="Advanced Indian Equity Screener & Valuation Model", layout="wide")

# --- 1. DATA ACQUISITION & SIMULATION ---
@st.cache_data
def fetch_stock_data():
    nifty_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ITC.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJFINANCE.NS",
        "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "TATASTEEL.NS",
        "NESTLEIND.NS", "WIPRO.NS", "M&M.NS", "HCLTECH.NS", "POWERGRID.NS",
        "PIDILITIND.NS", "APOLLOHOSP.NS", "HINDALCO.NS", "DIVISLAB.NS", "GRASIM.NS"
    ]
    
    data = []
    np.random.seed(42) # For reproducible mock data
    
    for ticker in nifty_tickers:
        try:
            hist_roe = np.random.uniform(8, 35)
            fwd_pe = np.random.uniform(15, 80)
            
            data.append({
                "Ticker": ticker.replace(".NS", ""),
                "Market_Cap_Cr": np.random.uniform(50000, 1500000),
                "Historical_ROE": round(hist_roe, 2),
                "Forward_ROE": round(hist_roe * np.random.uniform(0.9, 1.2), 2),
                "Growth_3Y": round(np.random.uniform(5, 30), 2),
                "Growth_5Y": round(np.random.uniform(5, 25), 2),
                "Growth_10Y": round(np.random.uniform(5, 20), 2),
                "TTM_PE": round(fwd_pe * np.random.uniform(0.8, 1.3), 2),
                "Forward_PE": round(fwd_pe, 2),
                "Forward_EPS": round(np.random.uniform(20, 200), 2),
                "Leverage_DE": round(np.random.uniform(0.0, 2.5), 2)
            })
        except Exception:
            continue
            
    return pd.DataFrame(data)

df = fetch_stock_data()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("Fundamental Filters")

min_hist_roe = st.sidebar.slider("Min Historical ROE (%)", 0.0, 40.0, 15.0)
min_fwd_roe = st.sidebar.slider("Min Forward ROE (%)", 0.0, 40.0, 15.0)
min_growth_5y = st.sidebar.slider("Min 5Y Hist Growth (%)", 0.0, 30.0, 10.0)
max_leverage = st.sidebar.slider("Max Leverage (D/E Ratio)", 0.0, 3.0, 2.0)
max_ttm_pe = st.sidebar.slider("Max TTM P/E", 10.0, 150.0, 100.0)
max_fwd_pe = st.sidebar.slider("Max Forward P/E", 10.0, 150.0, 100.0)
min_fwd_eps = st.sidebar.number_input("Min Forward EPS (₹)", value=10.0)

# Apply Filters
filtered_df = df[
    (df["Historical_ROE"] >= min_hist_roe) &
    (df["Forward_ROE"] >= min_fwd_roe) &
    (df["Growth_5Y"] >= min_growth_5y) &
    (df["Leverage_DE"] <= max_leverage) &
    (df["TTM_PE"] <= max_ttm_pe) &
    (df["Forward_PE"] <= max_fwd_pe) &
    (df["Forward_EPS"] >= min_fwd_eps)
]

# --- 3. MAIN DASHBOARD VISUALIZATION ---
st.title("Indian Equity Valuation Matrix")
st.markdown("### Cross-Sectional Analysis & Forward P/E Prediction")

if not filtered_df.empty:
    fig = px.scatter(
        filtered_df,
        x="Growth_5Y",
        y="Forward_ROE",
        size="Market_Cap_Cr",
        color="Forward_PE",
        hover_name="Ticker",
        # Custom high-contrast scale for dark mode: solid blue to vibrant coral-red
        color_continuous_scale=["#1e40af", "#ef4444"], 
        labels={
            "Growth_5Y": "5-Year Historical Growth (%)",
            "Forward_ROE": "Consensus Forward ROE (%)",
            "Forward_PE": "Fwd P/E"
        },
        title="Quality vs Valuation Matrix (Size = Market Cap, Color = Fwd P/E)"
    )
    
    # Apply deep dark theme formatting
    fig.update_layout(
        plot_bgcolor="#0f172a",   # Slate 900 background for the graph plot
        paper_bgcolor="#0f172a",  # Match entire component canvas area
        font=dict(color="#f8fafc"), # Crisp white text labels
        coloraxis_colorbar=dict(title="Fwd P/E", title_font=dict(color="#f8fafc"), tickfont=dict(color="#f8fafc")),
        xaxis=dict(
            showgrid=True, 
            gridcolor="#334155",  # Subtle grid lines
            zeroline=True, 
            zerolinecolor="#64748b",
            tickfont=dict(color="#cbd5e1")
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="#334155", 
            zeroline=True, 
            zerolinecolor="#64748b",
            tickfont=dict(color="#cbd5e1")
        ),
    )
    
    # Give bubbles a glowing border to make them distinctly pop
    fig.update_traces(marker=dict(line=dict(width=1.5, color="#ffffff")))
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No stocks match the current filter criteria.")

# Formatted Data Table
st.dataframe(
    filtered_df.style.format({
        "Market_Cap_Cr": "{:,.0f}", 
        "Historical_ROE": "{:.1f}%", 
        "Forward_ROE": "{:.1f}%", 
        "Growth_3Y": "{:.1f}%", 
        "Growth_5Y": "{:.1f}%", 
        "Growth_10Y": "{:.1f}%", 
        "Leverage_DE": "{:.2f}x"
    }), 
    use_container_width=True
)

# --- 4. MACHINE LEARNING VALUATION ENGINE ---
st.markdown("---")
st.subheader("Valuation Engine: Forward P/E Predictor")
st.markdown("This Random Forest model trains dynamically on your filtered cross-section to gauge consensus market pricing trends based on your criteria.")

features = ["Historical_ROE", "Forward_ROE", "Growth_5Y", "Leverage_DE"]
target = "Forward_PE"

if len(filtered_df) > 5:
    X = filtered_df[features]
    y = filtered_df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    st.markdown("**Test a Hypothetical Company Profile:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        test_hist_roe = st.number_input("Historical ROE (%)", value=20.0)
    with col2:
        test_fwd_roe = st.number_input("Forward ROE (%)", value=22.0)
    with col3:
        test_growth = st.number_input("5Y Growth (%)", value=15.0)
    with col4:
        test_lev = st.number_input("Leverage (D/E)", value=0.5)
        
    prediction = model.predict([[test_hist_roe, test_fwd_roe, test_growth, test_lev]])
    
    st.metric(
        label="Predicted Fair Forward P/E", 
        value=f"{prediction[0]:.2f}x",
        delta="Based on current cross-sectional data regression",
        delta_color="off"
    )
    
    # Feature Importance Summary
    importance = pd.DataFrame({
        'Metric': features,
        'Importance Weight': model.feature_importances_
    }).sort_values(by='Importance Weight', ascending=False)
    
    st.markdown("**What fundamental metrics are driving valuation variance in this cohort?**")
    st.bar_chart(importance.set_index('Metric'), color="#3b82f6")

else:
    st.info("Expand your filters above. The machine learning model requires at least 5 companies in the current viewport to calculate weights properly.")
