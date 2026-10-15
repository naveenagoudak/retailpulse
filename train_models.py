import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("1/4 Loading dataset...")
df = pd.read_csv("data/retail_transactions.csv")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


print("2/4 Training Demand Forecasting Model...")
daily_sales = df.groupby('InvoiceDate')['Quantity'].sum().reset_index()
daily_sales['DayOfWeek'] = daily_sales['InvoiceDate'].dt.dayofweek
daily_sales['Month'] = daily_sales['InvoiceDate'].dt.month
daily_sales['Day'] = daily_sales['InvoiceDate'].dt.day
daily_sales['Lag_1'] = daily_sales['Quantity'].shift(1)
daily_sales['Lag_7'] = daily_sales['Quantity'].shift(7)
daily_sales.dropna(inplace=True)

X_fc = daily_sales[['DayOfWeek', 'Month', 'Day', 'Lag_1', 'Lag_7']]
y_fc = daily_sales['Quantity']

forecast_model = RandomForestRegressor(n_estimators=100, random_state=42)
forecast_model.fit(X_fc, y_fc)
joblib.dump(forecast_model, "models/forecast_model.pkl")


print("3/4 Segmenting Customers (RFM)...")
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'TotalAmount': 'sum'                                     # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Cluster label mapping
cluster_names = {0: 'Regular', 1: 'Loyal/High Value', 2: 'At Risk'}
rfm['Segment'] = rfm['Cluster'].map(cluster_names)
rfm.to_csv("data/rfm_segmented.csv", index=False)
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/rfm_scaler.pkl")

print("4/4 Training Churn Prediction Model...")

rfm['Churn'] = (rfm['Recency'] > 90).astype(int)

X_churn = rfm[['Frequency', 'Monetary']]
y_churn = rfm['Churn']

churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
churn_model.fit(X_churn, y_churn)
joblib.dump(churn_model, "models/churn_model.pkl")

print(" All models trained and saved inside models/ directory successfully!")