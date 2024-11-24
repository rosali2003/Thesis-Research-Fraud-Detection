import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

# Set random seed
np.random.seed(42)

# Generate synthetic data
n_orders = 1000
n_products = 50

# Create product catalog
products = pd.DataFrame({
    'product_id': [f'PROD_{i:03d}' for i in range(n_products)],
    'retail_price': np.random.uniform(10, 1000, n_products),
    'resale_value_ratio': np.random.uniform(0.3, 0.8, n_products),  # Ratio of resale value to retail price
    'category': np.random.choice(['electronics', 'clothing', 'jewelry', 'home_goods', 'sports'], n_products)
})

# Lists to store order data
order_ids = []
customer_ids = []
product_lists = []
total_values = []
avg_resale_ratios = []
category_counts = []
is_fraudulent = []

# Generate normal order patterns (80%)
for i in range(800):
    order_id = f"ORDER_{i:04d}"
    customer_id = f"CUST_{np.random.randint(0, 400):04d}"
    
    # Normal customers tend to buy 1-5 items per order
    n_items = np.random.randint(1, 6)
    
    # Select products (typically mixed categories, moderate values)
    order_products = products.sample(n=n_items)
    
    order_ids.append(order_id)
    customer_ids.append(customer_id)
    product_lists.append(order_products['product_id'].tolist())
    total_values.append(order_products['retail_price'].sum())
    avg_resale_ratios.append(order_products['resale_value_ratio'].mean())
    category_counts.append(len(order_products['category'].unique()))
    is_fraudulent.append(0)

# Generate suspicious patterns (20%)
for i in range(200):
    order_id = f"ORDER_{i+800:04d}"
    customer_id = f"CUST_{np.random.randint(400, 500):04d}"
    
    # Fraudulent pattern type (0: high value, 1: low resale value)
    fraud_type = np.random.randint(0, 2)
    
    if fraud_type == 0:
        # High value items
        n_items = np.random.randint(3, 8)
        order_products = products[products['retail_price'] > products['retail_price'].quantile(0.8)].sample(n=n_items, replace=True)
    else:
        # Low resale value items
        n_items = np.random.randint(5, 10)
        order_products = products[products['resale_value_ratio'] < products['resale_value_ratio'].quantile(0.2)].sample(n=n_items, replace=True)
    
    order_ids.append(order_id)
    customer_ids.append(customer_id)
    product_lists.append(order_products['product_id'].tolist())
    total_values.append(order_products['retail_price'].sum())
    avg_resale_ratios.append(order_products['resale_value_ratio'].mean())
    category_counts.append(len(order_products['category'].unique()))
    is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': customer_ids,
    'products': product_lists,
    'total_value': total_values,
    'avg_resale_ratio': avg_resale_ratios,
    'unique_categories': category_counts,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['items_count'] = df['products'].apply(len)
df['avg_item_value'] = df['total_value'] / df['items_count']
df['customer_order_count'] = df.groupby('customer_id')['order_id'].transform('count')

# Prepare features and target
X = df.drop(['is_fraudulent', 'order_id', 'products', 'customer_id'], axis=1)
y = df['is_fraudulent']

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print dataset information
print("Dataset shape:", df.shape)
print("\nFeature columns:", X.columns.tolist())
print("\nClass distribution:")
print(df['is_fraudulent'].value_counts(normalize=True))
