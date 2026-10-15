import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

n_records = 15000
n_customers = 600

customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]
products = {
    "Electronics": ["Wireless Mouse", "USB-C Hub", "Mechanical Keyboard", "Monitor Stand", "Webcam HD"],
    "Apparel": ["Cotton T-Shirt", "Slim Jeans", "Hoodie", "Sports Cap", "Running Socks"],
    "Home & Kitchen": ["Coffee Mug", "Desk Lamp", "Water Bottle", "Air Purifier Filter", "Scented Candle"],
    "Fitness": ["Yoga Mat", "Dumbbell 5kg", "Resistance Bands", "Foam Roller", "Jump Rope"]
}

prod_flat = [(cat, item) for cat, items in products.items() for item in items]
base_date = datetime(2025, 1, 1)

data = []
for i in range(n_records):
    cust = np.random.choice(customer_ids)
    cat, prod = prod_flat[np.random.randint(0, len(prod_flat))]
    qty = np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03])
    
    price_ranges = {
        "Electronics": (15.0, 95.0),
        "Apparel": (10.0, 45.0),
        "Home & Kitchen": (8.0, 35.0),
        "Fitness": (12.0, 50.0)
    }
    low, high = price_ranges[cat]
    unit_price = round(np.random.uniform(low, high), 2)
    
    # 550 ದಿನಗಳ ಹಿಸ್ಟಾರಿಕಲ್ ಸೇಲ್ಸ್
    days_offset = np.random.randint(0, 550)
    inv_date = base_date + timedelta(days=days_offset)
    inv_no = f"INV{100000 + (i % 8000)}"
    
    data.append({
        "InvoiceNo": inv_no,
        "CustomerID": cust,
        "Category": cat,
        "Product": prod,
        "Quantity": qty,
        "UnitPrice": unit_price,
        "TotalAmount": round(qty * unit_price, 2),
        "InvoiceDate": inv_date.strftime("%Y-%m-%d")
    })

df = pd.DataFrame(data)
df.sort_values(by="InvoiceDate", inplace=True)
df.to_csv("data/retail_transactions.csv", index=False)
print(f"Dataset generated successfully with {len(df)} rows in data/retail_transactions.csv")