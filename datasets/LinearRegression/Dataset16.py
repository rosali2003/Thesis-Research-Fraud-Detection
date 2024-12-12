import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# ... (keep all the data preparation code as is) ...

# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Set up the model
input_dim = X_train.shape[1]
model = LinearRegression(input_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
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
    mse = mean_squared_error(y_test, y_pred.numpy())
    r2 = r2_score(y_test, y_pred.numpy())

    # Convert predictions to binary classification
    y_pred_class = (y_pred > 0.5).float()
    accuracy = accuracy_score(y_test, y_pred_class.numpy())
    precision = precision_score(y_test, y_pred_class.numpy())
    recall = recall_score(y_test, y_pred_class.numpy())
    f1 = f1_score(y_test, y_pred_class.numpy())

print("\nTest Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Accuracy (using 0.5 threshold): {accuracy:.4f}")
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

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred.numpy(), alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predicted vs Actual Values")
plt.show()
