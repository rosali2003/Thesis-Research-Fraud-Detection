import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Generate synthetic transaction data
np.random.seed(42)
n_transactions = 1000

# Generate timestamps over a 30 day period
timestamps = [datetime.now() - timedelta(days=np.random.randint(0,30), 
                                       hours=np.random.randint(0,24), 
                                       minutes=np.random.randint(0,60)) 
             for _ in range(n_transactions)]

# Define payment methods with probabilities
payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'cryptocurrency', 'digital_wallet']
payment_weights = [0.4, 0.3, 0.15, 0.05, 0.1]  # Make crypto and digital wallet less common

# Generate user IDs with some users having multiple transactions
n_users = 200
user_ids = np.random.randint(1, n_users+1, n_transactions)

# Generate transaction data
data = {
    'timestamp': timestamps,
    'user_id': user_ids,
    'payment_method': np.random.choice(payment_methods, n_transactions, p=payment_weights),
    'amount': np.random.uniform(10, 1000, n_transactions).round(2)
}

df = pd.DataFrame(data)

# Sort by user and timestamp
df = df.sort_values(['user_id', 'timestamp'])

# Calculate time difference and payment method changes between consecutive transactions
df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
df['payment_changed'] = df.groupby('user_id')['payment_method'].transform(lambda x: x != x.shift())

# Flag suspicious patterns
df['suspicious'] = False

# Flag rapid payment method switches (multiple changes within 1 hour)
user_groups = df.groupby('user_id')
for user_id, group in user_groups:
    mask = (group['time_diff'] < 3600) & (group['payment_changed'])
    if mask.sum() >= 2:  # At least 2 payment method changes within an hour
        df.loc[group.index, 'suspicious'] = True
        
# Flag unusual payment method usage
unusual_methods = ['cryptocurrency', 'digital_wallet']
df.loc[df['payment_method'].isin(unusual_methods), 'suspicious'] = True

# Convert timestamp to numerical features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day'] = pd.to_datetime(df['timestamp']).dt.day
df['month'] = pd.to_datetime(df['timestamp']).dt.month

# Drop original timestamp column
df = df.drop('timestamp', axis=1)

# Encode categorical variables
le = LabelEncoder()
df['payment_method'] = le.fit_transform(df['payment_method'])

# Prepare features and target
X = df.drop('suspicious', axis=1)
y = df['suspicious']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['amount', 'time_diff', 'hour', 'day', 'month']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("\nFeatures:", list(X_train.columns))
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))