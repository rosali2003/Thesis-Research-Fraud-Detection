import random
import pandas as pd

def generate_ip_address():
    """Generate a random IPv4 address as a string."""
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def create_vpn_dataset(num_rows=1000, vpn_ratio=0.1):
    """
    Create a synthetic dataset for VPN usage.
    
    Parameters:
        num_rows (int): Total number of rows in the dataset.
        vpn_ratio (float): Fraction of rows where the user is using a VPN.
    
    Returns:
        pd.DataFrame: A DataFrame with columns:
                      - ip_address: The generated IPv4 address.
                      - using_vpn: 1 if the user is using a VPN, 0 otherwise.
    """
    data = []
    for _ in range(num_rows):
        # Determine if the user is using a VPN (10% probability by default).
        using_vpn = 1 if random.random() < vpn_ratio else 0
        
        # Generate a random IP address.
        ip = generate_ip_address()
        
        data.append({
            "ip_address": ip,
            "using_vpn": using_vpn
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate the synthetic VPN dataset.
    df = create_vpn_dataset(num_rows=1000, vpn_ratio=0.3)
    
    # Optionally, save the dataset to a CSV file.
    df.to_csv("synthetic_vpn_dataset.csv", index=False)
    
    # Display the first few rows.
    print("First 5 rows of the synthetic VPN dataset:")
    print(df.head())
    
    # Display the percentage of rows using VPN.
    vpn_usage_percent = df['using_vpn'].mean() * 100
    print(f"\nPercentage of rows using VPN: {vpn_usage_percent:.2f}%")
