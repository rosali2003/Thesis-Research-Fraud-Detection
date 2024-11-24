import numpy as np
import pandas as pd
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


# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

# Define the neural network architecture
class TwoLayerNet(nn.Module):
    def __init__(self, input_size):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the model
model = TwoLayerNet(X_train_scaled.shape[1])

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
    
