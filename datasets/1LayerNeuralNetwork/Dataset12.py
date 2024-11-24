import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Define sample size
n_samples = 1000

# Define high fraud risk regions and their fraud probabilities
high_risk_regions = {
    'Region A': 0.8,  # 80% chance of fraud
    'Region B': 0.7,  # 70% chance of fraud 
    'Region C': 0.6   # 60% chance of fraud
}

medium_risk_regions = {
    'Region D': 0.4,  # 40% chance of fraud
    'Region E': 0.3,  # 30% chance of fraud
    'Region F': 0.25  # 25% chance of fraud
}

low_risk_regions = {
    'Region G': 0.1,  # 10% chance of fraud
    'Region H': 0.05, # 5% chance of fraud
    'Region I': 0.02  # 2% chance of fraud
}

# Create lists to store data
regions = []
fraud_labels = []

# Generate data for high risk regions
for region, fraud_prob in high_risk_regions.items():
    size = int(n_samples * 0.2)  # 20% of samples from high risk regions
    regions.extend([region] * size)
    fraud_labels.extend(np.random.choice(
        [0, 1], 
        size=size, 
        p=[1-fraud_prob, fraud_prob]
    ))

# Generate data for medium risk regions
for region, fraud_prob in medium_risk_regions.items():
    size = int(n_samples * 0.3)  # 30% of samples from medium risk regions
    regions.extend([region] * size)
    fraud_labels.extend(np.random.choice(
        [0, 1], 
        size=size, 
        p=[1-fraud_prob, fraud_prob]
    ))

# Generate data for low risk regions
for region, fraud_prob in low_risk_regions.items():
    size = int(n_samples * 0.5)  # 50% of samples from low risk regions
    regions.extend([region] * size)
    fraud_labels.extend(np.random.choice(
        [0, 1], 
        size=size, 
        p=[1-fraud_prob, fraud_prob]
    ))

# Create DataFrame
df = pd.DataFrame({
    'region': regions,
    'is_fraud': fraud_labels
})

# Add some noise to labels (flip some labels randomly to simulate real-world noise)
noise_idx = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
df.loc[noise_idx, 'is_fraud'] = 1 - df.loc[noise_idx, 'is_fraud']

# One-hot encode regions
df_encoded = pd.get_dummies(df, columns=['region'], prefix=['region'])

# Convert to numpy arrays
X = df_encoded.drop('is_fraud', axis=1).values
y = df_encoded['is_fraud'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print some statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(df)}")
print("\nFraud distribution by region:")
for region in high_risk_regions.keys():
    mask = df['region'] == region
    fraud_rate = df[mask]['is_fraud'].mean()
    print(f"{region}: {fraud_rate:.2%} fraud rate")
for region in medium_risk_regions.keys():
    mask = df['region'] == region
    fraud_rate = df[mask]['is_fraud'].mean()
    print(f"{region}: {fraud_rate:.2%} fraud rate")
for region in low_risk_regions.keys():
    mask = df['region'] == region
    fraud_rate = df[mask]['is_fraud'].mean()
    print(f"{region}: {fraud_rate:.2%} fraud rate")


# Custom dataset class
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create neural network class
class OneLayerNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=32):
        super(OneLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Create data loaders
train_dataset = FraudDataset(X_train_scaled, y_train)
test_dataset = FraudDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, loss function and optimizer
model = OneLayerNN(input_size=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
    # Evaluate on test set
    if (epoch + 1) % 10 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X).squeeze()
                predicted = (outputs >= 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{n_epochs}], Test Accuracy: {accuracy:.4f}')

# Final evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X).squeeze()
        predicted = (outputs >= 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
final_accuracy = correct / total
print(f'\nFinal Test Accuracy: {final_accuracy:.4f}')
