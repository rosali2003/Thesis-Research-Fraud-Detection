
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
from sklearn.ensemble import RandomForestClassifier

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

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'\nRandom Forest Test Accuracy: {accuracy:.4f}')