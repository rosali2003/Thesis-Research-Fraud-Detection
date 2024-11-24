import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Initialize lists to store transaction data
timestamps = []
user_ids = []
card_numbers = []
amounts = []
is_fraudulent = []

# Configuration
num_normal = 10000
num_suspicious = 200

# Generate base set of users and cards (increased for more realistic distribution)
base_users = [f"USER_{i:04d}" for i in range(500)]
base_cards = [f"CARD_{i:04d}" for i in range(800)]

# Generate normal transactions with realistic patterns
start_date = datetime(2024, 1, 1)
for _ in range(num_normal):
    # Normal users typically make 1-2 transactions per day
    timestamps.append(start_date + timedelta(
        days=random.randint(0, 90),  # Extended to 90 days for better distribution
        hours=random.randint(6, 23),  # Most transactions occur between 6 AM and 11 PM
        minutes=random.randint(0, 59)
    ))
    user_ids.append(random.choice(base_users))
    card_numbers.append(random.choice(base_cards))
    amounts.append(round(random.uniform(10, 1000), 2))
    is_fraudulent.append(0)

# Generate suspicious velocity attack patterns
for _ in range(num_suspicious):
    # Pick a random base timestamp
    base_time = start_date + timedelta(
        days=random.randint(0, 90),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    
    # Velocity attack patterns:
    # 1. Many transactions in very short time (5-15 minutes)
    # 2. Usually similar amounts
    # 3. Can be either same user or same card
    
    attack_type = random.choice(['user', 'card'])
    num_burst = random.randint(4, 8)  # More transactions in burst
    suspicious_user = random.choice(base_users)
    suspicious_card = random.choice(base_cards)
    base_amount = random.uniform(50, 500)  # Base amount for similar transactions
    
    for i in range(num_burst):
        # Transactions happen within 5-15 minutes
        time_offset = random.randint(1, 15)
        timestamps.append(base_time + timedelta(minutes=time_offset))
        
        if attack_type == 'user':
            # Same user, different cards
            user_ids.append(suspicious_user)
            card_numbers.append(random.choice(base_cards))
        else:
            # Same card, could be different users (stolen card)
            user_ids.append(random.choice(base_users))
            card_numbers.append(suspicious_card)
        
        # Similar amounts with small variations
        amount_variation = random.uniform(-20, 20)
        amounts.append(round(base_amount + amount_variation, 2))
        is_fraudulent.append(1)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'user_id': user_ids,
    'card_number': card_numbers,
    'amount': amounts,
    'is_fraudulent': is_fraudulent
})

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Save to CSV
output_file = "high_frequency_transactions.csv"
df.to_csv(output_file, index=False)

# Display detailed statistics
print(f"Dataset generated and saved to {output_file}")
print("\nDataset Statistics:")
print(f"Total transactions: {len(df)}")
print(f"Normal transactions: {len(df[df['is_fraudulent'] == 0])}")
print(f"Suspicious transactions: {len(df[df['is_fraudulent'] == 1])}")
print(f"Fraud ratio: {len(df[df['is_fraudulent'] == 1]) / len(df):.2%}")

# Display time span of the dataset
print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")


# Convert timestamp to numerical features
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['user_id', 'card_number'])

# Select features and target
X = df_encoded.drop(['timestamp', 'is_fraudulent'], axis=1)
y = df_encoded['is_fraudulent']
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train binary classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42)
clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = clf.predict(X_test_scaled)

# Print metrics
from sklearn.metrics import classification_report, accuracy_score
print("\nBinary Classification Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
