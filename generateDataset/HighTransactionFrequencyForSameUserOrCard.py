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
timestamps = []
user_ids = []
card_numbers = []
amounts = []
is_fraudulent = []

# Configuration
num_normal = 10000
num_suspicious = 200

# Generate base set of users and cards
base_users = [f"USER_{i:04d}" for i in range(500)]
base_cards = [f"CARD_{i:04d}" for i in range(800)]

# Generate normal transactions
start_date = datetime(2024, 1, 1)
for _ in range(num_normal):
    timestamps.append(start_date + timedelta(
        days=random.randint(0, 90),
        hours=random.randint(6, 23),
        minutes=random.randint(0, 59)
    ))
    user_ids.append(random.choice(base_users))
    card_numbers.append(random.choice(base_cards))
    amounts.append(round(random.uniform(10, 1000), 2))
    is_fraudulent.append(0)

# Generate suspicious velocity attack patterns
for _ in range(num_suspicious):
    base_time = start_date + timedelta(
        days=random.randint(0, 90),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    
    attack_type = random.choice(['user', 'card'])
    num_burst = random.randint(4, 8)
    suspicious_user = random.choice(base_users)
    suspicious_card = random.choice(base_cards)
    base_amount = random.uniform(50, 500)
    
    for i in range(num_burst):
        time_offset = random.randint(1, 15)
        timestamps.append(base_time + timedelta(minutes=time_offset))
        
        if attack_type == 'user':
            user_ids.append(suspicious_user)
            card_numbers.append(random.choice(base_cards))
        else:
            user_ids.append(random.choice(base_users))
            card_numbers.append(suspicious_card)
        
        amount_variation = random.uniform(-20, 20)
        amounts.append(round(base_amount + amount_variation, 2))
        is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'user_id': user_ids,
    'card_number': card_numbers,
    'amount': amounts,
    'is_fraudulent': is_fraudulent
})

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Prepare features and target
X = df.drop(['is_fraudulent', 'timestamp'], axis=1)
y = df['is_fraudulent']

# One-hot encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Impute missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)
print("X_train shape:", X_train.head())
print("y_train shape:", y_train.head())
print("X_test shape:", X_test.head())
print("y_test shape:", y_test.head())

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

# Save to CSV (optional)
df.to_csv("high_frequency_transactions.csv", index=False)
