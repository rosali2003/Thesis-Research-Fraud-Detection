import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Define sample size
n_samples = 1000

# Define high fraud risk regions and their fraud probabilities
high_risk_regions = {
    'Region A': 0.8,  # 80% chance of fraud
    'Region B': 0.7,  # 70% chance of fraud 
    'Region C': 0.6   # 60% chance of fraud
}

medium_risk_regions = {
    'Region D': 0.4,  # 40% chance of fraud
    'Region E': 0.3,  # 30% chance of fraud
    'Region F': 0.25  # 25% chance of fraud
}

low_risk_regions = {
    'Region G': 0.1,  # 10% chance of fraud
    'Region H': 0.05, # 5% chance of fraud
    'Region I': 0.02  # 2% chance of fraud
}

# Create lists to store data
regions = []
fraud_labels = []

# Generate data for high risk regions
for region, fraud_prob in high_risk_regions.items():
    size = int(n_samples * 0.2)  # 20% of samples from high risk regions
    regions.extend([region] * size)
    fraud_labels.extend(np.random.choice(
        [0, 1], 
        size=size, 
        p=[1-fraud_prob, fraud_prob]
    ))

# Generate data for medium risk regions
for region, fraud_prob in medium_risk_regions.items():
    size = int(n_samples * 0.3)  # 30% of samples from medium risk regions
    regions.extend([region] * size)
    fraud_labels.extend(np.random.choice(
        [0, 1], 
        size=size, 
        p=[1-fraud_prob, fraud_prob]
    ))

# Generate data for low risk regions
for region, fraud_prob in low_risk_regions.items():
    size = int(n_samples * 0.5)  # 50% of samples from low risk regions
    regions.extend([region] * size)
    fraud_labels.extend(np.random.choice(
        [0, 1], 
        size=size, 
        p=[1-fraud_prob, fraud_prob]
    ))

# Create DataFrame
df = pd.DataFrame({
    'region': regions,
    'is_fraud': fraud_labels
})

# Add some noise to labels (flip some labels randomly to simulate real-world noise)
noise_idx = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
df.loc[noise_idx, 'is_fraud'] = 1 - df.loc[noise_idx, 'is_fraud']

# One-hot encode regions
df_encoded = pd.get_dummies(df, columns=['region'], prefix=['region'])

# Convert to numpy arrays
X = df_encoded.drop('is_fraud', axis=1).values
y = df_encoded['is_fraud'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print some statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(df)}")
print("\nFraud distribution by region:")
for region in high_risk_regions.keys():
    mask = df['region'] == region
    fraud_rate = df[mask]['is_fraud'].mean()
    print(f"{region}: {fraud_rate:.2%} fraud rate")
for region in medium_risk_regions.keys():
    mask = df['region'] == region
    fraud_rate = df[mask]['is_fraud'].mean()
    print(f"{region}: {fraud_rate:.2%} fraud rate")
for region in low_risk_regions.keys():
    mask = df['region'] == region
    fraud_rate = df[mask]['is_fraud'].mean()
    print(f"{region}: {fraud_rate:.2%} fraud rate")

# Create binary classification model
from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'\nLogistic Regression Test Accuracy: {accuracy:.4f}')

# Print classification report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_pred_proba = model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()
