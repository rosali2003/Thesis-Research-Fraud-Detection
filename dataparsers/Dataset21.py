import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic data
np.random.seed(42)
n_transactions = 1000

# Lists to store transaction data
transaction_ids = []
customer_ids = []
ip_countries = []
billing_countries = []
shipping_countries = []
order_amounts = []
is_fraudulent = []

# List of countries for data generation
countries = ['US', 'CA', 'GB', 'FR', 'DE', 'ES', 'IT', 'AU', 'JP', 'BR']

# Generate legitimate transactions (80%)
for i in range(800):
    transaction_ids.append(f"TXN_{i:04d}")
    customer_ids.append(f"CUST_{random.randint(1, 400):04d}")
    
    # For legitimate transactions, locations usually match
    country = np.random.choice(countries)
    ip_countries.append(country)
    
    # Small chance of different but reasonable shipping/billing addresses
    if random.random() < 0.1:
        billing_countries.append(np.random.choice(countries))
    else:
        billing_countries.append(country)
        
    if random.random() < 0.15:
        shipping_countries.append(np.random.choice(countries))
    else:
        shipping_countries.append(country)
    
    order_amounts.append(round(random.uniform(50, 500), 2))
    is_fraudulent.append(0)

# Generate suspicious transactions (20%)
for i in range(200):
    transaction_ids.append(f"TXN_{i+800:04d}")
    customer_ids.append(f"CUST_{random.randint(401, 500):04d}")
    
    # For suspicious transactions, IP location differs significantly
    billing_country = np.random.choice(countries)
    shipping_country = np.random.choice(countries)
    
    # Ensure IP country is different from both billing and shipping
    available_countries = [c for c in countries if c not in [billing_country, shipping_country]]
    ip_country = np.random.choice(available_countries)
    
    ip_countries.append(ip_country)
    billing_countries.append(billing_country)
    shipping_countries.append(shipping_country)
    
    # Slightly higher order amounts for suspicious transactions
    order_amounts.append(round(random.uniform(200, 1000), 2))
    is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'transaction_id': transaction_ids,
    'customer_id': customer_ids,
    'ip_country': ip_countries,
    'billing_country': billing_countries,
    'shipping_country': shipping_countries,
    'order_amount': order_amounts,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['ip_billing_match'] = (df['ip_country'] == df['billing_country']).astype(int)
df['ip_shipping_match'] = (df['ip_country'] == df['shipping_country']).astype(int)
df['billing_shipping_match'] = (df['billing_country'] == df['shipping_country']).astype(int)

# Calculate historical patterns per customer
df['customer_country_mismatch_rate'] = df.groupby('customer_id').apply(
    lambda x: ((x['ip_country'] != x['billing_country']) | 
               (x['ip_country'] != x['shipping_country'])).mean()
).reset_index(level=0, drop=True)

# Prepare features and target
X = df.drop(['is_fraudulent', 'transaction_id'], axis=1)
y = df['is_fraudulent']

# Encode categorical variables
le = LabelEncoder()
X['customer_id'] = le.fit_transform(X['customer_id'])
X['ip_country'] = le.fit_transform(X['ip_country'])
X['billing_country'] = le.fit_transform(X['billing_country'])
X['shipping_country'] = le.fit_transform(X['shipping_country'])

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['order_amount', 'customer_country_mismatch_rate']
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
