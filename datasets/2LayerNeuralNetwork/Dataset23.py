import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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

# Lists to store login attempt data
user_ids = []
attempt_dates = []
success = []
ip_addresses = []
is_fraudulent = []

start_date = datetime(2024, 1, 1)

# Generate normal login patterns (80%)
for i in range(400):
    user_id = f"USER_{i:04d}"
    base_ip = f"192.168.{random.randint(0,255)}"

    # Generate login attempts for this user
    n_attempts = np.random.randint(10, 30)  # Random number of login attempts

    for _ in range(n_attempts):
        attempt_date = start_date + timedelta(days=np.random.randint(0, n_days))
        user_ids.append(user_id)
        attempt_dates.append(attempt_date)

        # Normal users mostly have successful logins with occasional failures
        success.append(1 if random.random() < 0.95 else 0)
        # Normal users mostly use same IP
        ip_addresses.append(f"{base_ip}.{random.randint(0,10)}")
        is_fraudulent.append(0)

# Generate suspicious patterns (20%)
for i in range(100):
    user_id = f"USER_{i+400:04d}"

    # Generate series of failed attempts followed by success
    n_series = np.random.randint(3, 6)  # Number of attack series

    for _ in range(n_series):
        attempt_date = start_date + timedelta(days=np.random.randint(0, n_days))
        n_failures = np.random.randint(5, 15)  # Number of failed attempts in series

        # Generate multiple failed attempts
        for j in range(n_failures):
            user_ids.append(user_id)
            attempt_dates.append(attempt_date + timedelta(minutes=random.randint(1, 10)))
            success.append(0)
            # Attackers try from different IPs
            ip_addresses.append(f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}")
            is_fraudulent.append(1)

        # Add successful attempt after failures
        user_ids.append(user_id)
        attempt_dates.append(attempt_date + timedelta(minutes=random.randint(11, 20)))
        success.append(1)
        ip_addresses.append(f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}")
        is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'user_id': user_ids,
    'attempt_date': attempt_dates,
    'success': success,
    'ip_address': ip_addresses,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['attempt_hour'] = df['attempt_date'].dt.hour
df['daily_attempts'] = df.groupby(['user_id', df['attempt_date'].dt.date])['success'].transform('count')
df['daily_failures'] = df.groupby(['user_id', df['attempt_date'].dt.date])['success'].transform(lambda x: (x == 0).sum())
df['failure_rate'] = df['daily_failures'] / df['daily_attempts']
df['unique_ips'] = df.groupby(['user_id', df['attempt_date'].dt.date])['ip_address'].transform('nunique')

# Prepare features and target
X = df.drop(['is_fraudulent', 'attempt_date', 'ip_address'], axis=1)
y = df['is_fraudulent']

# Encode categorical variables
le = LabelEncoder()
X['user_id'] = le.fit_transform(X['user_id'])

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['daily_attempts', 'daily_failures', 'failure_rate', 'unique_ips', 'attempt_hour']
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


# Print dataset information
print("Dataset shape:", df.shape)
print("\nFeature columns:", X.columns.tolist())
print("\nClass distribution:")
print(df['is_fraudulent'].value_counts(normalize=True))

# 1. Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"Precision: {precision_score(y_test, lr_pred):.4f}")
print(f"Recall: {recall_score(y_test, lr_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, lr_pred):.4f}")

# 2. Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
lin_reg_pred = (lin_reg_model.predict(X_test) > 0.5).astype(int)

print("\nLinear Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, lin_reg_pred):.4f}")
print(f"Precision: {precision_score(y_test, lin_reg_pred):.4f}")
print(f"Recall: {recall_score(y_test, lin_reg_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, lin_reg_pred):.4f}")

# 3. 1-Layer Neural Network
class OneLayerNN(nn.Module):
    def __init__(self, input_dim):
        super(OneLayerNN, self).__init__()
        self.layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.layer(x))

input_dim = X_train.shape[1]
one_layer_model = OneLayerNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(one_layer_model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        outputs = one_layer_model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

one_layer_model.eval()
with torch.no_grad():
    one_layer_pred = (one_layer_model(X_test_tensor).squeeze() > 0.5).cpu().numpy()

print("\n1-Layer Neural Network Results:")
print(f"Accuracy: {accuracy_score(y_test, one_layer_pred):.4f}")
print(f"Precision: {precision_score(y_test, one_layer_pred):.4f}")
print(f"Recall: {recall_score(y_test, one_layer_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, one_layer_pred):.4f}")

# 4. 2-Layer Neural Network
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.sigmoid(self.layer2(x))

hidden_dim = 32
two_layer_model = TwoLayerNN(input_dim, hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(two_layer_model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        outputs = two_layer_model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

two_layer_model.eval()
with torch.no_grad():
    two_layer_pred = (two_layer_model(X_test_tensor).squeeze() > 0.5).cpu().numpy()

print("\n2-Layer Neural Network Results:")
print(f"Accuracy: {accuracy_score(y_test, two_layer_pred):.4f}")
print(f"Precision: {precision_score(y_test, two_layer_pred):.4f}")
print(f"Recall: {recall_score(y_test, two_layer_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, two_layer_pred):.4f}")

# 5. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"Recall: {recall_score(y_test, rf_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, rf_pred):.4f}")

# Feature importance for Random Forest
feature_importance = rf_model.feature_importances_
feature_names = X.columns.tolist()
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nRandom Forest Feature Importance:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Visualize feature importance
plt.figure(figsize=(12, 6))
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
