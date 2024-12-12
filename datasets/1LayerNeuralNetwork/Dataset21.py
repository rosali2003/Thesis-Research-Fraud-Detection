import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
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

# generate data code here


# Print dataset information
print("Dataset shape:", df.shape)
print("\nFeature columns:", X.columns.tolist())
print("\nClass distribution:")
print(df['is_fraudulent'].value_counts(normalize=True))


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

