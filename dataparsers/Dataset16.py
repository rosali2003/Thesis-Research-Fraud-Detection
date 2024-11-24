import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_transactions = 1000

# Generate user data
user_ids = [f"USER_{i:04d}" for i in range(200)]
billing_countries = ['US', 'UK', 'CA', 'AU', 'DE']
user_billing_country = {user: np.random.choice(billing_countries) for user in user_ids}

# Lists to store transaction data
transaction_data = []

# Generate legitimate transactions
for _ in range(800):
    user_id = np.random.choice(user_ids)
    billing_country = user_billing_country[user_id]
    
    # Access country usually matches billing country with small variation
    if np.random.random() < 0.9:
        access_country = billing_country
    else:
        access_country = np.random.choice([c for c in billing_countries if c != billing_country])
    
    transaction_data.append({
        'user_id': user_id,
        'billing_country': billing_country,
        'access_country': access_country,
        'using_vpn': False,
        'ip_type': 'residential',
        'transaction_amount': round(np.random.uniform(10, 1000), 2),
        'is_fraudulent': 0
    })

# Generate suspicious transactions
vpn_countries = ['RU', 'CN', 'BR', 'NL', 'SG']  # Common VPN server locations
for _ in range(200):
    user_id = np.random.choice(user_ids)
    billing_country = user_billing_country[user_id]
    access_country = np.random.choice(vpn_countries)
    
    transaction_data.append({
        'user_id': user_id,
        'billing_country': billing_country,
        'access_country': access_country,
        'using_vpn': True,
        'ip_type': np.random.choice(['datacenter', 'proxy']),
        'transaction_amount': round(np.random.uniform(10, 1000), 2),
        'is_fraudulent': 1
    })

# Create DataFrame
df = pd.DataFrame(transaction_data)

# Encode categorical variables
le_dict = {}
categorical_cols = ['user_id', 'billing_country', 'access_country', 'ip_type']
for col in categorical_cols:
    le_dict[col] = LabelEncoder()
    df[col] = le_dict[col].fit_transform(df[col])

# Prepare features and target
X = df.drop('is_fraudulent', axis=1)
y = df['is_fraudulent']

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['transaction_amount']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print dataset information
print("Dataset shape:", df.shape)
print("\nFeature columns:", X.columns.tolist())
print("\nClass distribution:")
print(df['is_fraudulent'].value_counts(normalize=True))
