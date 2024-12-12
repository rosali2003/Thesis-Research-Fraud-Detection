import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
