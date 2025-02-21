import random
import pandas as pd

def generate_ip_address():
    """Generate a random IPv4 address."""
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def create_fraud_dataset(num_rows=1000, proxy_ratio=0.2):
    """
    Create a synthetic fraud detection dataset.
    
    Parameters:
        num_rows (int): Total number of rows in the dataset.
        proxy_ratio (float): Fraction of rows where the user is using a proxy.
    
    Returns:
        pd.DataFrame: A DataFrame with columns:
                      - ip_address: The generated IPv4 address.
                      - using_proxy: 1 if the row represents a user using a proxy, 0 otherwise.
                      - is_fraud: 1 if the row is fraudulent, 0 otherwise.
    """
    data = []
    for _ in range(num_rows):
        # Determine if this row uses a proxy (20% probability).
        using_proxy = 1 if random.random() < proxy_ratio else 0
        
        # Generate a random IP address.
        ip = generate_ip_address()
        
        # Assign a fraud probability based on proxy usage.
        # For example: 10% chance if using proxy, 5% chance otherwise.
        fraud_prob = 0.10 if using_proxy else 0.05
        is_fraud = 1 if random.random() < fraud_prob else 0
        
        data.append({
            "ip_address": ip,
            "using_proxy": using_proxy,
            "is_fraud": is_fraud
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate the dataset.
    df = create_fraud_dataset(num_rows=1000, proxy_ratio=0.4)
    
    # Optionally, save the dataset to a CSV file.
    df.to_csv("synthetic_fraud_dataset.csv", index=False)
    
    # Display the first few rows and the percentage of proxy usage.
    print("First 5 rows of the synthetic fraud detection dataset:")
    print(df.head())
    proxy_usage_percent = df['using_proxy'].mean() * 100
    print(f"\nPercentage of rows using a proxy: {proxy_usage_percent:.2f}%")
