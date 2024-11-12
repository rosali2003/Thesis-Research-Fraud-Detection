class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)  # No sigmoid, outputs raw values

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
linear_model = LinearRegressionModel(input_size)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.SGD(linear_model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = linear_model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


from sklearn.metrics import classification_report, precision_recall_curve, auc

# Evaluation
linear_model.eval()
with torch.no_grad():
    # Get raw predictions
    y_pred_linear = linear_model(X_test_tensor).squeeze().numpy()

    # Apply a threshold to convert to binary predictions
    y_pred_binary = (y_pred_linear >= 0.5).astype(int)

print("Linear Regression Classification Report (Thresholded):\n", classification_report(y_test, y_pred_binary))

# Calculate Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_binary)
pr_auc_linear = auc(recall, precision)
print(f"Linear Regression Precision-Recall AUC: {pr_auc_linear:.4f}")
