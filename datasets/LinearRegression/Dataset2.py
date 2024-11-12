import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report

# Function to generate a list of IP addresses that change every 50 rows
def generate_ip_addresses(num_samples, change_interval=50):
    ip_addresses = []
    for i in range(num_samples):
        if i % change_interval == 0:
            # Generate a new IP address every 'change_interval' rows
            new_ip = f"192.168.{i // change_interval}.1"
        ip_addresses.append(new_ip)
    return ip_addresses

# Number of samples in the dataset
num_samples = 1000

# Create the dataset with changing IP addresses
data = {
    "ip_address": generate_ip_addresses(num_samples),
    "shipping_address": ["1234 Main St, Anytown, State, 12345"] * num_samples,
    "email": ["user@example.com"] * num_samples,
    "order": ["order_value"] * num_samples,
    "transaction_amount": [100] * (num_samples // 2) + [1000] * (num_samples // 2),  # Example feature
    "transaction_time": [1] * (num_samples // 2) + [2] * (num_samples // 2)  # Example feature
}

# Create a DataFrame
df = pd.DataFrame(data)

# Add a binary target variable for classification (0: non-fraudulent, 1: fraudulent)
df['is_fraud'] = [0] * (num_samples // 2) + [1] * (num_samples // 2)

# Select features and target
X = df.drop(['is_fraud', 'shipping_address', 'email', 'order'], axis=1)
y = df['is_fraud']

# Encode the IP address using one-hot encoding
X_encoded = pd.get_dummies(X, columns=['ip_address'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_continuous = model.predict(X_test)

# Since it's a classification task, convert the continuous predictions to binary (0 or 1)
y_pred = (y_pred_continuous > 0.5).astype(int)

# Evaluate the model using classification metrics
mse = mean_squared_error(y_test, y_pred_continuous)
r2 = r2_score(y_test, y_pred_continuous)
report = classification_report(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2 Score): {r2}")
print("Classification Report:")
print(report)
