import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic data
np.random.seed(42)
n_accounts = 500
n_days = 90

# Lists to store transaction data
account_ids = []
transaction_dates = []
amounts = []
transaction_frequencies = []  # Daily transaction count
is_fraudulent = []

start_date = datetime(2024, 1, 1)

# Generate normal transaction patterns (80%)
for i in range(400):
    account_id = f"ACC_{i:04d}"
    
    # Normal spending pattern parameters
    avg_daily_transactions = random.uniform(0.5, 2)  # Average transactions per day
    avg_amount = random.uniform(50, 200)  # Average transaction amount
    amount_std = avg_amount * 0.2  # Standard deviation for amount
    
    # Generate transactions for this account
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        
        # Random number of transactions for this day
        n_transactions = np.random.poisson(avg_daily_transactions)
        
        for _ in range(n_transactions):
            account_ids.append(account_id)
            transaction_dates.append(current_date)
            amounts.append(max(10, np.random.normal(avg_amount, amount_std)))
            transaction_frequencies.append(n_transactions)
            is_fraudulent.append(0)

# Generate suspicious patterns (20%)
for i in range(100):
    account_id = f"ACC_{i+400:04d}"
    
    # Normal period
    avg_daily_transactions = random.uniform(0.5, 2)
    avg_amount = random.uniform(50, 200)
    amount_std = avg_amount * 0.2
    
    # Generate normal transactions for first 60 days
    for day in range(60):
        current_date = start_date + timedelta(days=day)
        n_transactions = np.random.poisson(avg_daily_transactions)
        
        for _ in range(n_transactions):
            account_ids.append(account_id)
            transaction_dates.append(current_date)
            amounts.append(max(10, np.random.normal(avg_amount, amount_std)))
            transaction_frequencies.append(n_transactions)
            is_fraudulent.append(0)
    
    # Suspicious period - increased frequency and amounts
    for day in range(60, n_days):
        current_date = start_date + timedelta(days=day)
        n_transactions = np.random.poisson(avg_daily_transactions * 3)  # Triple frequency
        
        for _ in range(n_transactions):
            account_ids.append(account_id)
            transaction_dates.append(current_date)
            amounts.append(max(10, np.random.normal(avg_amount * 2.5, amount_std * 2)))  # Higher amounts
            transaction_frequencies.append(n_transactions)
            is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'account_id': account_ids,
    'transaction_date': transaction_dates,
    'amount': amounts,
    'daily_transaction_count': transaction_frequencies,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['amount_7day_mean'] = df.groupby('account_id')['amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
df['freq_7day_mean'] = df.groupby('account_id')['daily_transaction_count'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

# Prepare features and target
X = df.drop(['is_fraudulent', 'account_id', 'transaction_date'], axis=1)
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
