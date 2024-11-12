import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample size
n_samples = 1000

# Generate distances (in kilometers)
distances = np.concatenate([
    # Legitimate transactions (under 1000 km)
    np.random.normal(5, 3, size=int(n_samples * 0.2)),     # Same city
    np.random.normal(50, 20, size=int(n_samples * 0.2)),   # Nearby cities
    np.random.normal(200, 100, size=int(n_samples * 0.2)), # Different cities
    np.random.normal(500, 200, size=int(n_samples * 0.2)), # Different regions
    # Fraudulent transactions (over 1000 km)
    np.random.normal(2000, 500, size=int(n_samples * 0.1)),  # Different countries
    np.random.normal(5000, 1000, size=int(n_samples * 0.1)) # International
])

# Ensure all distances are positive
distances = np.abs(distances)

# Create labels (0 for legitimate, 1 for fraudulent)
labels = (distances > 1000).astype(int)

# Create a DataFrame
df = pd.DataFrame({
    'distance_km': distances,
    'is_fraud': labels
})

# Add some noise to make the dataset more realistic
noise = np.random.normal(0, 20, size=len(df))
df['distance_km'] += noise

# Ensure distances remain positive after adding noise
df['distance_km'] = np.abs(df['distance_km'])

# Recalculate fraud labels after noise addition
df['is_fraud'] = (df['distance_km'] > 1000).astype(int)

# Optional: Scale the features
scaler = StandardScaler()
df['distance_km_scaled'] = scaler.fit_transform(df[['distance_km']])

# Save the dataset
df.to_csv('shipping_distance_fraud.csv', index=False)

# Print some statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(df)}")
print(f"Fraud cases: {len(df[df['is_fraud'] == 1])}")
print(f"Legitimate cases: {len(df[df['is_fraud'] == 0])}")
print("\nDistance Statistics (in kilometers):")
print(df.groupby('is_fraud')['distance_km'].describe())
