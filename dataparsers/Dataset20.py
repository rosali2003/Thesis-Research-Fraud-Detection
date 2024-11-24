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

# Lists to store shipping data
order_ids = []
customer_ids = []
item_values = []
address_types = []  # residential, warehouse, hotel, po_box
customer_ages = []  # days since account creation
shipping_frequencies = []  # Number of shipments to this address
is_fraudulent = []

start_date = datetime(2024, 1, 1)

# Generate legitimate orders (75%)
for i in range(750):
    order_ids.append(f"ORD_{i:04d}")
    customer_ids.append(f"CUST_{random.randint(1, 300):04d}")
    
    # Mostly moderate value items to residential addresses
    item_values.append(round(random.uniform(50, 500), 2))
    address_types.append(np.random.choice(['residential', 'po_box', 'hotel', 'warehouse'], 
                                        p=[0.85, 0.05, 0.05, 0.05]))
    
    # Established customers
    customer_ages.append(random.randint(60, 1000))
    shipping_frequencies.append(random.randint(1, 20))
    is_fraudulent.append(0)

# Generate suspicious patterns (25%)
for i in range(250):
    order_ids.append(f"ORD_{i+750:04d}")
    customer_ids.append(f"CUST_{random.randint(301, 400):04d}")
    
    # High-value items to non-residential addresses
    item_values.append(round(random.uniform(800, 5000), 2))
    address_types.append(np.random.choice(['warehouse', 'hotel', 'po_box'], 
                                        p=[0.4, 0.3, 0.3]))
    
    # Newer accounts
    customer_ages.append(random.randint(1, 30))
    shipping_frequencies.append(random.randint(10, 50))
    is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': customer_ids,
    'item_value': item_values,
    'address_type': address_types,
    'account_age_days': customer_ages,
    'address_shipping_frequency': shipping_frequencies,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['avg_order_value'] = df.groupby('customer_id')['item_value'].transform('mean')
df['non_residential_ratio'] = df.groupby('customer_id')['address_type'].transform(
    lambda x: (x != 'residential').mean()
)

# Prepare features and target
X = df.drop(['is_fraudulent', 'order_id'], axis=1)
y = df['is_fraudulent']

# Encode categorical variables
le = LabelEncoder()
X['customer_id'] = le.fit_transform(X['customer_id'])
X['address_type'] = le.fit_transform(X['address_type'])

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['item_value', 'account_age_days', 'address_shipping_frequency', 
                 'avg_order_value', 'non_residential_ratio']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print dataset information
print("Dataset shape:", df.shape)
print("\nFeature columns:", X.columns.tolist())
print("\nClass distribution:")
print(df['is_fraudulent'].value_counts(normalize=True))
