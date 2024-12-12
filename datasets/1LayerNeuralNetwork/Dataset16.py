import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# ... (keep all the data preparation code as is) ...

# Define the 1-layer neural network model
class OneLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(OneLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Set up the model
input_dim = X_train.shape[1]
hidden_dim = 32  # You can adjust this
model = OneLayerNN(input_dim, hidden_dim)

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

    accuracy = accuracy_score(y_test, y_pred_class.numpy())
    precision = precision_score(y_test, y_pred_class.numpy())
    recall = recall_score(y_test, y_pred_class.numpy())
    f1 = f1_score(y_test, y_pred_class.numpy())

print("\nTest Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred.numpy(), alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("1-Layer NN: Predicted vs Actual Values")
plt.show()

# Feature importance (approximation)
with torch.no_grad():
    # Use the weights of the first layer as a proxy for feature importance
    feature_importance = model.layer1.weight.abs().mean(dim=0).numpy()
    feature_names = X.columns.tolist()
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nApproximate Feature Importance:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")
