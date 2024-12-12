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
n_orders = 1000
n_products = 50

# Create product catalog
products = pd.DataFrame({
    'product_id': [f'PROD_{i:03d}' for i in range(n_products)],
    'retail_price': np.random.uniform(10, 1000, n_products),
    'resale_value_ratio': np.random.uniform(0.3, 0.8, n_products),  # Ratio of resale value to retail price
    'category': np.random.choice(['electronics', 'clothing', 'jewelry', 'home_goods', 'sports'], n_products)
})

# Lists to store order data
order_ids = []
customer_ids = []
product_lists = []
total_values = []
avg_resale_ratios = []
category_counts = []
is_fraudulent = []

# Generate normal order patterns (80%)
for i in range(800):
    order_id = f"ORDER_{i:04d}"
    customer_id = f"CUST_{np.random.randint(0, 400):04d}"

    # Normal customers tend to buy 1-5 items per order
    n_items = np.random.randint(1, 6)

    # Select products (typically mixed categories, moderate values)
    order_products = products.sample(n=n_items)

    order_ids.append(order_id)
    customer_ids.append(customer_id)
    product_lists.append(order_products['product_id'].tolist())
    total_values.append(order_products['retail_price'].sum())
    avg_resale_ratios.append(order_products['resale_value_ratio'].mean())
    category_counts.append(len(order_products['category'].unique()))
    is_fraudulent.append(0)

# Generate suspicious patterns (20%)
for i in range(200):
    order_id = f"ORDER_{i+800:04d}"
    customer_id = f"CUST_{np.random.randint(400, 500):04d}"

    # Fraudulent pattern type (0: high value, 1: low resale value)
    fraud_type = np.random.randint(0, 2)

    if fraud_type == 0:
        # High value items
        n_items = np.random.randint(3, 8)
        order_products = products[products['retail_price'] > products['retail_price'].quantile(0.8)].sample(n=n_items, replace=True)
    else:
        # Low resale value items
        n_items = np.random.randint(5, 10)
        order_products = products[products['resale_value_ratio'] < products['resale_value_ratio'].quantile(0.2)].sample(n=n_items, replace=True)

    order_ids.append(order_id)
    customer_ids.append(customer_id)
    product_lists.append(order_products['product_id'].tolist())
    total_values.append(order_products['retail_price'].sum())
    avg_resale_ratios.append(order_products['resale_value_ratio'].mean())
    category_counts.append(len(order_products['category'].unique()))
    is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': customer_ids,
    'products': product_lists,
    'total_value': total_values,
    'avg_resale_ratio': avg_resale_ratios,
    'unique_categories': category_counts,
    'is_fraudulent': is_fraudulent
})

# Calculate additional features
df['items_count'] = df['products'].apply(len)
df['avg_item_value'] = df['total_value'] / df['items_count']
df['customer_order_count'] = df.groupby('customer_id')['order_id'].transform('count')

# Prepare features and target
X = df.drop(['is_fraudulent', 'order_id', 'products', 'customer_id'], axis=1)
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
