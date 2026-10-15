import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta

st.set_page_config(
    page_title="RetailPulse - AI Analytics Platform",
    page_icon="🛍️",
    layout="wide"
)

# -------------------------------------------------------------
# Data & Model Loaders
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/retail_transactions.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    rfm = pd.read_csv("data/rfm_segmented.csv")
    return df, rfm

@st.cache_resource
def load_models():
    forecast_model = joblib.load("models/forecast_model.pkl")
    churn_model = joblib.load("models/churn_model.pkl")
    return forecast_model, churn_model

df, rfm = load_data()
forecast_model, churn_model = load_models()

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.title("🛍️ RetailPulse: AI-Powered Analytics & Forecasting")
st.markdown("Automated retail intelligence for inventory demand, customer segments, and churn mitigation.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${df['TotalAmount'].sum():,.2f}")
col2.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")
col3.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
col4.metric("Avg Order Value", f"${df.groupby('InvoiceNo')['TotalAmount'].sum().mean():.2f}")

st.markdown("---")

# -------------------------------------------------------------
# Tabs Navigation
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Demand Forecasting", 
    "👥 Customer Segmentation (RFM)", 
    "⚠️ Churn Risk Predictor"
])

# -------------------------------------------------------------
# Tab 1: Demand Forecasting
# -------------------------------------------------------------
with tab1:
    st.subheader("Daily Sales Trend & Demand Forecasting")
    
    daily_sales = df.groupby('InvoiceDate')['Quantity'].sum().reset_index()
    
    fig_sales = px.line(
        daily_sales, 
        x='InvoiceDate', 
        y='Quantity', 
        title="Historical Daily Units Sold",
        template="plotly_white"
    )
    st.plotly_chart(fig_sales, use_container_width=True)

    st.markdown("#### 7-Day Future Demand Prediction")
    if st.button("Generate Next 7 Days Forecast"):
        last_date = daily_sales['InvoiceDate'].max()
        last_qty = daily_sales['Quantity'].iloc[-1]
        lag_7_val = daily_sales['Quantity'].iloc[-7]

        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        preds = []

        curr_lag1 = last_qty
        for d in future_dates:
            feat = pd.DataFrame([{
                'DayOfWeek': d.dayofweek,
                'Month': d.month,
                'Day': d.day,
                'Lag_1': curr_lag1,
                'Lag_7': lag_7_val
            }])
            pred = forecast_model.predict(feat)[0]
            preds.append(round(pred))
            curr_lag1 = pred

        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Units': preds})

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(forecast_df, use_container_width=True)
        with col_b:
            fig_fc = px.bar(
                forecast_df, 
                x='Date', 
                y='Predicted Units', 
                title="Forecasted Daily Unit Sales",
                text='Predicted Units',
                template="plotly_white"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

# -------------------------------------------------------------
# Tab 2: RFM Segmentation
# -------------------------------------------------------------
with tab2:
    st.subheader("Customer Clusters via RFM Analysis")
    
    col_x, col_y = st.columns(2)
    with col_x:
        fig_scatter = px.scatter(
            rfm, 
            x='Recency', 
            y='Monetary', 
            color='Segment', 
            size='Frequency',
            hover_data=['CustomerID'],
            title="Recency vs Monetary (Size = Frequency)",
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_y:
        segment_counts = rfm['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Customer Count']
        fig_pie = px.pie(
            segment_counts, 
            names='Segment', 
            values='Customer Count', 
            title="Customer Distribution by Segment",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Segment Profiles")
    segment_summary = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    st.dataframe(segment_summary.style.format({'Recency': '{:.1f} days', 'Frequency': '{:.1f} orders', 'Monetary': '${:,.2f}'}), use_container_width=True)

# -------------------------------------------------------------
# Tab 3: Churn Prediction
# -------------------------------------------------------------
with tab3:
    st.subheader("Predict Single Customer Churn Risk")
    
    st.write("Enter customer engagement metrics to predict likelihood of churning:")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        in_freq = st.number_input("Total Orders Placed (Frequency)", min_value=1, max_value=50, value=3)
    with col_in2:
        in_monetary = st.number_input("Total Spend ($ Monetary)", min_value=10.0, max_value=10000.0, value=150.0)

    if st.button("Predict Churn Status"):
        input_data = pd.DataFrame([{'Frequency': in_freq, 'Monetary': in_monetary}])
        churn_prob = churn_model.predict_proba(input_data)[0][1]
        is_churn = churn_prob >= 0.5

        if is_churn:
            st.error(f"⚠️ High Churn Risk (Probability: {churn_prob * 100:.1f}%)")
            st.info("Recommended Action: Trigger loyalty discounts or re-engagement email sequence.")
        else:
            st.success(f"✅ Active & Engaged (Probability of Churn: {churn_prob * 100:.1f}%)")
            st.info("Recommended Action: Upsell premium catalog items.")