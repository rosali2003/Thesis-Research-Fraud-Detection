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

# Generate base transaction data
transaction_ids = []
user_ids = []
transaction_types = []  # 'purchase', 'return', 'cancellation'
amounts = []
time_to_return = []  # Time between purchase and return/cancellation
is_fraudulent = []

# Generate legitimate transactions (70%)
for i in range(700):
    transaction_ids.append(f"TXN_{i:04d}")
    user_ids.append(f"USER_{random.randint(1, 200):03d}")
    
    # Mostly purchases, some legitimate returns
    tx_type = np.random.choice(['purchase', 'return'], p=[0.8, 0.2])
    transaction_types.append(tx_type)
    
    amount = round(random.uniform(50, 500), 2)
    amounts.append(amount)
    
    # Legitimate returns usually happen after several days
    if tx_type == 'return':
        time_to_return.append(random.randint(3, 30))
    else:
        time_to_return.append(0)
    
    is_fraudulent.append(0)

# Generate suspicious patterns (30%)
for i in range(300):
    transaction_ids.append(f"TXN_{i+700:04d}")
    
    # Suspicious users make multiple returns/cancellations
    user_ids.append(f"USER_{random.randint(201, 250):03d}")
    
    # Higher proportion of returns and cancellations
    tx_type = np.random.choice(['purchase', 'return', 'cancellation'], p=[0.3, 0.4, 0.3])
    transaction_types.append(tx_type)
    
    amount = round(random.uniform(100, 1000), 2)
    amounts.append(amount)
    
    # Suspicious returns/cancellations happen very quickly
    if tx_type in ['return', 'cancellation']:
        time_to_return.append(random.randint(0, 2))
    else:
        time_to_return.append(0)
    
    is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'transaction_id': transaction_ids,
    'user_id': user_ids,
    'transaction_type': transaction_types,
    'amount': amounts,
    'time_to_return': time_to_return,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['return_frequency'] = df.groupby('user_id')['transaction_type'].transform(
    lambda x: (x == 'return').mean()
)
df['cancellation_frequency'] = df.groupby('user_id')['transaction_type'].transform(
    lambda x: (x == 'cancellation').mean()
)

# Prepare features and target
X = df.drop(['is_fraudulent', 'transaction_id'], axis=1)
y = df['is_fraudulent']

# Encode categorical variables
le = LabelEncoder()
X['user_id'] = le.fit_transform(X['user_id'])
X['transaction_type'] = le.fit_transform(X['transaction_type'])

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['amount', 'time_to_return', 'return_frequency', 'cancellation_frequency']
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
