import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample size
n_samples = 1000

# Generate legitimate transactions (centered around different distances)
legitimate_dist = np.concatenate([
    np.random.normal(100, 50, size=int(n_samples * 0.3)),   # Local
    np.random.normal(500, 200, size=int(n_samples * 0.3)),  # Regional
    np.random.normal(900, 300, size=int(n_samples * 0.4))   # Some near threshold
])

# Generate fraudulent transactions
fraudulent_dist = np.concatenate([
    np.random.normal(1100, 300, size=int(n_samples * 0.4)),  # Just over threshold
    np.random.normal(2000, 500, size=int(n_samples * 0.3)),  # Different country
    np.random.normal(3000, 800, size=int(n_samples * 0.3))   # International
])

# Ensure positive distances
legitimate_dist = np.abs(legitimate_dist)
fraudulent_dist = np.abs(fraudulent_dist)

# Create initial labels
legitimate_labels = np.zeros(len(legitimate_dist))
fraudulent_labels = np.ones(len(fraudulent_dist))

# Combine data
distances = np.concatenate([legitimate_dist, fraudulent_dist])
labels = np.concatenate([legitimate_labels, fraudulent_labels])

# Add intentional misclassifications (noise in labels)
noise_idx = np.random.choice(len(distances), size=int(len(distances) * 0.1), replace=False)
labels[noise_idx] = 1 - labels[noise_idx]  # Flip labels for selected indices

# Create DataFrame
df = pd.DataFrame({
    'distance_km': distances,
    'is_fraud': labels
})

# Add noise to distances
df['distance_km'] += np.random.normal(0, 100, size=len(df))
df['distance_km'] = np.abs(df['distance_km'])

# Convert data to PyTorch tensors
X = df[['distance_km']].values
y = df['is_fraud'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
class SimpleNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super(SimpleNN, self).__init__()
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

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Training loop
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    # Evaluate on test set
    model.eval()
    epoch_test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            epoch_test_loss += loss.item()
    
    train_losses.append(epoch_train_loss / len(train_loader))
    test_losses.append(epoch_test_loss / len(test_loader))
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# Evaluate model
model.eval()
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test_scaled)).numpy()
y_pred_binary = (y_pred >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(y_pred_binary == y_test.reshape(-1, 1))
print(f'\nTest Accuracy: {accuracy:.4f}')

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.savefig('training_curves.png')
plt.close()

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label='Predicted')
plt.xlabel('Distance (km)')
plt.ylabel('Fraud Probability')
plt.title('Neural Network: Actual vs Predicted Values')
plt.legend()
plt.savefig('nn_predictions.png')
plt.close()

# Plot decision boundary
plt.figure(figsize=(10, 6))
X_plot = np.linspace(df['distance_km'].min(), df['distance_km'].max(), 300).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
with torch.no_grad():
    y_plot = model(torch.FloatTensor(X_plot_scaled)).numpy()

plt.scatter(X[y == 0], y[y == 0], color='blue', alpha=0.5, label='Legitimate')
plt.scatter(X[y == 1], y[y == 1], color='red', alpha=0.5, label='Fraud')
plt.plot(X_plot, y_plot, color='green', label='Decision Boundary')
plt.axvline(x=1000, color='black', linestyle='--', label='1000km Threshold')
plt.xlabel('Distance (km)')
plt.ylabel('Fraud Probability')
plt.title('Neural Network Decision Boundary')
plt.legend()
plt.savefig('nn_decision_boundary.png')
plt.close()