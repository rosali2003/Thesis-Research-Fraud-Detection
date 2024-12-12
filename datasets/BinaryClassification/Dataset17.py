import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Set up the model
input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    y_pred_class = (y_pred > 0.5).float()

    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)

print("\nTest Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Feature importance
with torch.no_grad():
    feature_importance = model.linear.weight.abs().squeeze().numpy()
    feature_names = X.columns.tolist()
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")
