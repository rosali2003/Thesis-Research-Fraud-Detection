import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic data
np.random.seed(42)
n_accounts = 1000

# Generate account creation dates over last 180 days
creation_dates = [datetime.now() - timedelta(days=np.random.randint(1, 180)) 
                 for _ in range(n_accounts)]

# Generate transaction data
account_ids = []
transaction_dates = []
amounts = []
is_fraudulent = []

# Normal accounts (80%)
for i in range(int(n_accounts * 0.8)):
    creation_date = creation_dates[i]
    num_transactions = np.random.randint(1, 10)
    
    for _ in range(num_transactions):
        # Transactions spread out over time after account creation
        days_after = np.random.randint(7, 90)
        transaction_dates.append(creation_date + timedelta(days=days_after))
        account_ids.append(f"ACC_{i:04d}")
        amounts.append(round(np.random.uniform(50, 500), 2))
        is_fraudulent.append(0)

# Suspicious new accounts with high-value transactions (20%)
for i in range(int(n_accounts * 0.8), n_accounts):
    creation_date = creation_dates[i]
    num_transactions = np.random.randint(3, 7)
    
    for _ in range(num_transactions):
        # Transactions very soon after account creation
        days_after = np.random.randint(0, 5)
        transaction_dates.append(creation_date + timedelta(days=days_after))
        account_ids.append(f"ACC_{i:04d}")
        amounts.append(round(np.random.uniform(2000, 10000), 2))
        is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'account_id': account_ids,
    'creation_date': [creation_dates[int(acc_id.split('_')[1])] for acc_id in account_ids],
    'transaction_date': transaction_dates,
    'amount': amounts,
    'is_fraudulent': is_fraudulent
})

# Calculate time since account creation for each transaction
df['days_since_creation'] = (df['transaction_date'] - df['creation_date']).dt.total_seconds() / (24 * 3600)

# Drop datetime columns and account_id
X = df.drop(['is_fraudulent', 'creation_date', 'transaction_date', 'account_id'], axis=1)
y = df['is_fraudulent']

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
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
