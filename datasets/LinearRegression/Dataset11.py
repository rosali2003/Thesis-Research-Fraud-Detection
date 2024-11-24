import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

# Split and scale data
X = df[['distance_km']].values
y = df['is_fraud'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print metrics
print("\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Print model parameters
print("\nModel Parameters:")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label='Predicted')
plt.xlabel('Distance (km)')
plt.ylabel('Fraud Probability')
plt.title('Linear Regression: Actual vs Predicted Values')
plt.legend()
plt.savefig('linear_regression_predictions.png')
plt.close()

# Visualize regression line
plt.figure(figsize=(10, 6))
X_plot = np.linspace(df['distance_km'].min(), df['distance_km'].max(), 300).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_plot = model.predict(X_plot_scaled)

plt.scatter(X[y == 0], y[y == 0], color='blue', alpha=0.5, label='Legitimate')
plt.scatter(X[y == 1], y[y == 1], color='red', alpha=0.5, label='Fraud')
plt.plot(X_plot, y_plot, color='green', label='Regression Line')
plt.axvline(x=1000, color='black', linestyle='--', label='1000km Threshold')
plt.xlabel('Distance (km)')
plt.ylabel('Fraud Probability')
plt.title('Linear Regression Line')
plt.legend()
plt.savefig('linear_regression_line.png')
plt.close()

# Create residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('residual_plot.png')
plt.close()

# Save the dataset
df.to_csv('shipping_distance_fraud.csv', index=False)

# Print some statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(df)}")
print(f"Fraud cases: {len(df[df['is_fraud'] == 1])}")
print(f"Legitimate cases: {len(df[df['is_fraud'] == 0])}")
print("\nDistance Statistics (in kilometers):")
print(df.groupby('is_fraud')['distance_km'].describe())

# Print additional insights
print("\nPrediction Range:")
print(f"Min predicted value: {y_pred.min():.4f}")
print(f"Max predicted value: {y_pred.max():.4f}")
