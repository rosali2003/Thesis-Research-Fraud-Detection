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
start_date = datetime(2024, 1, 1)

# Lists to store order data
order_ids = []
customer_ids = []
original_addresses = []
final_addresses = []
address_changes = []
order_edit_counts = []
days_to_changes = []
is_fraudulent = []

# Generate addresses
street_names = ['Main St', 'Oak Ave', 'Maple Dr', 'Cedar Ln', 'Pine Rd']
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
states = ['NY', 'CA', 'IL', 'TX', 'AZ']

def generate_address():
    return f"{random.randint(100, 9999)} {random.choice(street_names)}, {random.choice(cities)}, {random.choice(states)}"

# Generate normal order patterns (80%)
for i in range(800):
    order_id = f"ORDER_{i:04d}"
    customer_id = f"CUST_{random.randint(0, 400):04d}"
    
    # Normal orders have few or no changes
    original_addr = generate_address()
    n_changes = np.random.choice([0, 1], p=[0.8, 0.2])
    edit_count = np.random.randint(0, 3)
    
    if n_changes == 0:
        final_addr = original_addr
    else:
        final_addr = generate_address()
        
    order_ids.append(order_id)
    customer_ids.append(customer_id)
    original_addresses.append(original_addr)
    final_addresses.append(final_addr)
    address_changes.append(n_changes)
    order_edit_counts.append(edit_count)
    days_to_changes.append(random.randint(0, 3) if n_changes > 0 else 0)
    is_fraudulent.append(0)

# Generate suspicious patterns (20%)
for i in range(200):
    order_id = f"ORDER_{i+800:04d}"
    customer_id = f"CUST_{random.randint(401, 500):04d}"
    
    # Suspicious orders have multiple changes
    original_addr = generate_address()
    n_changes = random.randint(2, 5)
    edit_count = random.randint(3, 8)
    final_addr = generate_address()
    
    order_ids.append(order_id)
    customer_ids.append(customer_id)
    original_addresses.append(original_addr)
    final_addresses.append(final_addr)
    address_changes.append(n_changes)
    order_edit_counts.append(edit_count)
    days_to_changes.append(random.randint(4, 10))
    is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': customer_ids,
    'original_address': original_addresses,
    'final_address': final_addresses,
    'address_changes': address_changes,
    'order_edit_count': order_edit_counts,
    'days_to_changes': days_to_changes,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['address_changed'] = (df['original_address'] != df['final_address']).astype(int)
df['customer_address_change_rate'] = df.groupby('customer_id')['address_changed'].transform('mean')
df['customer_edit_rate'] = df.groupby('customer_id')['order_edit_count'].transform('mean')

# Prepare features and target
X = df.drop(['is_fraudulent', 'order_id', 'original_address', 'final_address'], axis=1)
y = df['is_fraudulent']

# Encode categorical variables
le = LabelEncoder()
X['customer_id'] = le.fit_transform(X['customer_id'])

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['address_changes', 'order_edit_count', 'days_to_changes', 
                 'customer_address_change_rate', 'customer_edit_rate']
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
