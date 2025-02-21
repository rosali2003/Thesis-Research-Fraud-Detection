import random
import pandas as pd

def generate_ip_address():
    """Generate a random IPv4 address as a string."""
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def create_blacklist_dataset(num_rows=1000, blacklist_ratio=0.1):
    """
    Create a synthetic fraud detection dataset based on IP status.
    
    Parameters:
        num_rows (int): Total number of rows in the dataset.
        blacklist_ratio (float): Fraction of rows with blacklisted IP addresses.
    
    Returns:
        pd.DataFrame: A DataFrame with columns:
                      - ip_address: The generated IPv4 address.
                      - ip_status: "blacklisted" if the IP is blacklisted, "normal" otherwise.
                      - is_fraud: 1 if the row is fraudulent, 0 otherwise.
                                 Fraud probability is higher for blacklisted IPs.
    """
    data = []
    for _ in range(num_rows):
        # Determine if the IP is blacklisted (10% probability).
        if random.random() < blacklist_ratio:
            ip_status = "blacklisted"
            # For example, assign a higher fraud probability for blacklisted IPs.
            fraud_prob = 0.30
        else:
            ip_status = "normal"
            # For normal IPs, assign a lower fraud probability.
            fraud_prob = 0.05
        
        # Generate a random IP address.
        ip = generate_ip_address()
        
        # Determine fraud label.
        is_fraud = 1 if random.random() < fraud_prob else 0
        
        data.append({
            "ip_address": ip,
            "ip_status": ip_status,
            "is_fraud": is_fraud
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate the synthetic dataset.
    df = create_blacklist_dataset(num_rows=1000, blacklist_ratio=0.3)
    
    # Optionally, save the dataset to a CSV file.
    df.to_csv("synthetic_blacklist_fraud_dataset.csv", index=False)
    
    # Display the first few rows of the dataset.
    print("First 5 rows of the synthetic fraud detection dataset:")
    print(df.head())
    
    # Display the counts for blacklisted vs normal IP addresses.
    print("\nIP Status Counts:")
    print(df['ip_status'].value_counts())
    
    # Display overall fraud rate.
    fraud_rate = df['is_fraud'].mean() * 100
    print(f"\nOverall Fraud Rate: {fraud_rate:.2f}%")
