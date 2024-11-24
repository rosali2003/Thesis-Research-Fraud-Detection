import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Generate synthetic data
n_samples = 10000
transaction_ids = []
customer_ids = []
attempt_times = []
card_numbers = []
transaction_amounts = []
is_approved = []
is_fraudulent = []

# Generate legitimate transactions (70% of samples)
for i in range(int(n_samples * 0.7)):
    customer_id = f"C{random.randint(1000, 9999)}"
    base_time = datetime.now() - timedelta(days=random.randint(1, 180))
    
    # Single successful transaction
    transaction_ids.append(f"T{i}")
    customer_ids.append(customer_id)
    attempt_times.append(base_time)
    card_numbers.append(f"CARD_{random.randint(1000, 9999)}")
    transaction_amounts.append(random.uniform(10, 1000))
    is_approved.append(1)
    is_fraudulent.append(0)

# Generate fraudulent patterns (30% of samples)
for i in range(int(n_samples * 0.3)):
    customer_id = f"C{random.randint(1000, 9999)}"
    base_time = datetime.now() - timedelta(days=random.randint(1, 180))
    
    # Multiple attempts with different cards in quick succession
    n_attempts = random.randint(3, 6)
    for j in range(n_attempts):
        transaction_ids.append(f"T{len(transaction_ids)}")
        customer_ids.append(customer_id)
        attempt_times.append(base_time + timedelta(minutes=random.randint(1, 15)))
        card_numbers.append(f"CARD_{random.randint(1000, 9999)}")
        transaction_amounts.append(random.uniform(100, 2000))
        is_approved.append(1 if j == n_attempts-1 else 0)
        is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'transaction_id': transaction_ids,
    'customer_id': customer_ids,
    'attempt_time': attempt_times,
    'card_number': card_numbers,
    'transaction_amount': transaction_amounts,
    'is_approved': is_approved,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['attempt_hour'] = df['attempt_time'].dt.hour
df['attempts_per_customer'] = df.groupby('customer_id')['transaction_id'].transform('count')
df['unique_cards_per_customer'] = df.groupby('customer_id')['card_number'].transform('nunique')
df['card_switch_rate'] = df['unique_cards_per_customer'] / df['attempts_per_customer']
df['approval_rate'] = df.groupby('customer_id')['is_approved'].transform('mean')

# Calculate time differences between attempts
df = df.sort_values(['customer_id', 'attempt_time'])
df['time_since_last_attempt'] = df.groupby('customer_id')['attempt_time'].diff().dt.total_seconds().fillna(0)

# Prepare features and target
X = df.drop(['is_fraudulent', 'transaction_id', 'attempt_time', 'card_number'], axis=1)
y = df['is_fraudulent']

# Encode categorical variables
le = LabelEncoder()
X['customer_id'] = le.fit_transform(X['customer_id'])

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['transaction_amount', 'attempts_per_customer', 'unique_cards_per_customer',
                 'card_switch_rate', 'approval_rate', 'time_since_last_attempt', 'attempt_hour']
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
