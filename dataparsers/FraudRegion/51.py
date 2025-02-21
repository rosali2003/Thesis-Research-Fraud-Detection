import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define the total number of orders
n_orders = 1000

# Define the proportion and number of orders from the fraud region
fraud_region_percentage = 0.20
n_fraud_orders = int(n_orders * fraud_region_percentage)
n_nonfraud_orders = n_orders - n_fraud_orders

# Define the fraud region and a list of non-fraud regions
fraud_region = "Fraud Region"
nonfraud_regions = ["North", "South", "East", "West", "Central"]

# Generate billing address regions for non-fraud orders by randomly choosing from nonfraud_regions
nonfraud_order_regions = np.random.choice(nonfraud_regions, n_nonfraud_orders)

# Create an array for fraud orders: all are from the fraud region
fraud_order_regions = np.array([fraud_region] * n_fraud_orders)

# Combine both arrays
all_regions = np.concatenate([nonfraud_order_regions, fraud_order_regions])

# Shuffle the combined array to mix fraud and non-fraud orders randomly
np.random.shuffle(all_regions)

# Create a DataFrame with order_id and billing_address_region
df = pd.DataFrame({
    "order_id": range(1, n_orders + 1),
    "billing_address_region": all_regions
})

# Add a fraud history column: if billing_address_region is the fraud region, label as 1 (fraud), else 0 (non-fraud)
df["fraud"] = df["billing_address_region"].apply(lambda x: 1 if x == fraud_region else 0)

# Display a sample of the dataset
print(df.head(10))
print("\nFraud distribution:")
print(df["fraud"].value_counts())
