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
n_users = 500
n_days = 30

# Lists to store login data
user_ids = []
login_dates = []
device_types = ['desktop', 'mobile', 'tablet']
browsers = ['chrome', 'firefox', 'safari', 'edge']
operating_systems = ['windows', 'macos', 'android', 'ios', 'linux']

device_list = []
browser_list = []
os_list = []
is_fraudulent = []

start_date = datetime(2024, 1, 1)

# Generate normal user patterns (80%)
for i in range(400):
    user_id = f"USER_{i:04d}"
    
    # Normal users tend to use 1-2 devices consistently
    preferred_devices = np.random.choice(device_types, size=np.random.randint(1, 3), replace=False)
    preferred_browser = np.random.choice(browsers)
    preferred_os = np.random.choice(operating_systems)
    
    # Generate logins for this user
    n_logins = np.random.randint(15, 45)  # Random number of logins over the period
    
    for _ in range(n_logins):
        login_date = start_date + timedelta(days=np.random.randint(0, n_days))
        user_ids.append(user_id)
        login_dates.append(login_date)
        
        # High probability of using preferred device/browser/os
        device_list.append(np.random.choice(preferred_devices))
        browser_list.append(preferred_browser if np.random.random() < 0.9 else np.random.choice(browsers))
        os_list.append(preferred_os if np.random.random() < 0.9 else np.random.choice(operating_systems))
        is_fraudulent.append(0)

# Generate suspicious patterns (20%)
for i in range(100):
    user_id = f"USER_{i+400:04d}"
    
    # Suspicious users frequently change their configuration
    n_logins = np.random.randint(30, 60)  # More frequent logins
    
    for _ in range(n_logins):
        login_date = start_date + timedelta(days=np.random.randint(0, n_days))
        user_ids.append(user_id)
        login_dates.append(login_date)
        
        # Randomly select device/browser/os with high variability
        device_list.append(np.random.choice(device_types))
        browser_list.append(np.random.choice(browsers))
        os_list.append(np.random.choice(operating_systems))
        is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'user_id': user_ids,
    'login_date': login_dates,
    'device_type': device_list,
    'browser': browser_list,
    'operating_system': os_list,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['daily_login_count'] = df.groupby(['user_id', 'login_date']).transform('count')['is_fraudulent']
df['unique_devices_24h'] = df.groupby('user_id').rolling('24H', on='login_date')['device_type'].transform('nunique')
df['unique_browsers_24h'] = df.groupby('user_id').rolling('24H', on='login_date')['browser'].transform('nunique')
df['unique_os_24h'] = df.groupby('user_id').rolling('24H', on='login_date')['operating_system'].transform('nunique')

# Fill NaN values with 1 (for first login where rolling window is empty)
df = df.fillna(1)

# Prepare features and target
X = df.drop(['is_fraudulent', 'login_date'], axis=1)
y = df['is_fraudulent']

# Encode categorical variables
le_dict = {}
categorical_cols = ['user_id', 'device_type', 'browser', 'operating_system']
for col in categorical_cols:
    le_dict[col] = LabelEncoder()
    X[col] = le_dict[col].fit_transform(X[col])

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['daily_login_count', 'unique_devices_24h', 'unique_browsers_24h', 'unique_os_24h']
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
