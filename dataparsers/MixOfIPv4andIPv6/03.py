import random
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_ipv4():
    """Generate a random IPv4 address."""
    return '.'.join(str(random.randint(0, 255)) for _ in range(4))

def generate_ipv6():
    """Generate a random IPv6 address (8 groups of 4 hex digits)."""
    return ':'.join('{:04x}'.format(random.randint(0, 0xFFFF)) for _ in range(8))

def create_synthetic_dataset(num_rows=1000, ipv4_ratio=0.9, fraud_ratio=0.05):
    """
    Create a synthetic fraud detection dataset.
    
    Parameters:
        num_rows (int): Total number of rows in the dataset.
        ipv4_ratio (float): Fraction of rows that should have an IPv4 address.
        fraud_ratio (float): Fraction of transactions that are fraudulent.
        
    Returns:
        pd.DataFrame: A DataFrame containing the synthetic dataset.
    """
    ip_addresses = []
    transaction_amounts = []
    fraud_labels = []
    
    for _ in range(num_rows):
        # Determine IP address type based on the desired ratio.
        if random.random() < ipv4_ratio:
            ip_addresses.append(generate_ipv4())
        else:
            ip_addresses.append(generate_ipv6())
        
        # Create a random transaction amount between $1.00 and $1000.00.
        transaction_amounts.append(round(random.uniform(1.0, 1000.0), 2))
        
        # Assign a fraud label (1 for fraud, 0 for not fraud) based on fraud_ratio.
        fraud_labels.append(1 if random.random() < fraud_ratio else 0)
    
    # Create the DataFrame.
    df = pd.DataFrame({
        'ip_address': ip_addresses,
        'transaction_amount': transaction_amounts,
        'is_fraud': fraud_labels
    })

    # Separate features and target.
    X = df[['ip_address', 'transaction_amount']]
    y = df['is_fraud']

    # Proceed with your preprocessing, feature engineering (e.g., encoding the IP address), and training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
