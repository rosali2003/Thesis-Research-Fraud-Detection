import torch.nn as nn
import torch.optim as optim

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
logistic_model = LogisticRegressionModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.SGD(logistic_model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = logistic_model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


from sklearn.metrics import classification_report, precision_recall_curve, auc

# Evaluation
logistic_model.eval()
with torch.no_grad():
    y_pred_logistic = logistic_model(X_test_tensor).squeeze().round()
    y_pred_logistic = y_pred_logistic.numpy()

print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logistic))

# Calculate Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_logistic)
pr_auc_logistic = auc(recall, precision)
print(f"Logistic Regression Precision-Recall AUC: {pr_auc_logistic:.4f}")
