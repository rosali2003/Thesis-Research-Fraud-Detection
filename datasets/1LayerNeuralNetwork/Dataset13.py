import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Initialize lists to store transaction data
timestamps = []
user_ids = []
card_numbers = []
amounts = []
is_fraudulent = []

# Configuration
num_normal = 10000
num_suspicious = 200

# Generate base set of users and cards (increased for more realistic distribution)
base_users = [f"USER_{i:04d}" for i in range(500)]
base_cards = [f"CARD_{i:04d}" for i in range(800)]

# Generate normal transactions with realistic patterns
start_date = datetime(2024, 1, 1)
for _ in range(num_normal):
    # Normal users typically make 1-2 transactions per day
    timestamps.append(start_date + timedelta(
        days=random.randint(0, 90),  # Extended to 90 days for better distribution
        hours=random.randint(6, 23),  # Most transactions occur between 6 AM and 11 PM
        minutes=random.randint(0, 59)
    ))
    user_ids.append(random.choice(base_users))
    card_numbers.append(random.choice(base_cards))
    amounts.append(round(random.uniform(10, 1000), 2))
    is_fraudulent.append(0)

# Generate suspicious velocity attack patterns
for _ in range(num_suspicious):
    # Pick a random base timestamp
    base_time = start_date + timedelta(
        days=random.randint(0, 90),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    
    # Velocity attack patterns:
    # 1. Many transactions in very short time (5-15 minutes)
    # 2. Usually similar amounts
    # 3. Can be either same user or same card
    
    attack_type = random.choice(['user', 'card'])
    num_burst = random.randint(4, 8)  # More transactions in burst
    suspicious_user = random.choice(base_users)
    suspicious_card = random.choice(base_cards)
    base_amount = random.uniform(50, 500)  # Base amount for similar transactions
    
    for i in range(num_burst):
        # Transactions happen within 5-15 minutes
        time_offset = random.randint(1, 15)
        timestamps.append(base_time + timedelta(minutes=time_offset))
        
        if attack_type == 'user':
            # Same user, different cards
            user_ids.append(suspicious_user)
            card_numbers.append(random.choice(base_cards))
        else:
            # Same card, could be different users (stolen card)
            user_ids.append(random.choice(base_users))
            card_numbers.append(suspicious_card)
        
        # Similar amounts with small variations
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

# Save to CSV
output_file = "high_frequency_transactions.csv"
df.to_csv(output_file, index=False)

# Display detailed statistics
print(f"Dataset generated and saved to {output_file}")
print("\nDataset Statistics:")
print(f"Total transactions: {len(df)}")
print(f"Normal transactions: {len(df[df['is_fraudulent'] == 0])}")
print(f"Suspicious transactions: {len(df[df['is_fraudulent'] == 1])}")
print(f"Fraud ratio: {len(df[df['is_fraudulent'] == 1]) / len(df):.2%}")

# Display time span of the dataset
print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")


# Convert timestamp to numerical features
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['user_id', 'card_number'])

# Select features and target
X = df_encoded.drop(['timestamp', 'is_fraudulent'], axis=1)
y = df_encoded['is_fraudulent']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
import torch
import torch.nn as nn
import torch.optim as optim

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

# Define the neural network architecture
class OneLayerNet(nn.Module):
    def __init__(self, input_size):
        super(OneLayerNet, self).__init__()
        self.layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.layer(x))
        return x

# Initialize the model
model = OneLayerNet(X_train_scaled.shape[1])

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 100
batch_size = 32

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
print("\nTraining Neural Network:")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_tensor).squeeze()
    y_pred = (y_pred_proba >= 0.5).float()
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f'\nNeural Network Test Accuracy: {accuracy:.4f}')
    
    # Calculate and print additional metrics
    from sklearn.metrics import classification_report, accuracy_score
    y_pred_np = y_pred.numpy()
    y_test_np = y_test_tensor.numpy()
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_np, y_pred_np))
    
    accuracy = accuracy_score(y_test_np, y_pred_np)
    print(f"Accuracy Score: {accuracy:.4f}")
