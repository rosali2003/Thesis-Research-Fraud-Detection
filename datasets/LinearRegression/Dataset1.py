import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Mock dataset creation (replace this with your actual dataset)
num_samples = 1000
data = {
    "ip_address": ["192.168.1.1"] * num_samples,
    "shipping_address": ["1234 Main St, Anytown, State, 12345"] * num_samples,
    "email": ["user@example.com"] * num_samples,
    "order": ["order_value"] * num_samples,
    "transaction_amount": [100] * (num_samples // 2) + [1000] * (num_samples // 2),  # Example feature
    "transaction_time": [1] * (num_samples // 2) + [2] * (num_samples // 2)  # Example feature
}

# Create a DataFrame
df = pd.DataFrame(data)

# Add a binary target variable for fraud classification (0: non-fraudulent, 1: fraudulent)
df['is_fraud'] = [0] * (num_samples // 2) + [1] * (num_samples // 2)

# Select features and target
X = df.drop(['is_fraud', 'ip_address', 'shipping_address', 'email', 'order'], axis=1)
y = df['is_fraud']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using metrics suited for regression
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")
