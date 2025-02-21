import random
import pandas as pd

def is_private(ip):
    """
    Check if an IPv4 address (string) is in a private range.
    Private ranges:
      - 10.0.0.0 to 10.255.255.255
      - 172.16.0.0 to 172.31.255.255
      - 192.168.0.0 to 192.168.255.255
    """
    parts = list(map(int, ip.split('.')))
    if parts[0] == 10:
        return True
    if parts[0] == 172 and 16 <= parts[1] <= 31:
        return True
    if parts[0] == 192 and parts[1] == 168:
        return True
    return False

def generate_private_ip():
    """Generate a random private IPv4 address by choosing one of the private ranges."""
    range_choice = random.choice(['10', '172', '192'])
    
    if range_choice == '10':
        # 10.x.x.x
        return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
    elif range_choice == '172':
        # 172.16.x.x to 172.31.x.x
        return f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(0,255)}"
    else:
        # 192.168.x.x
        return f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"

def generate_public_ip():
    """Generate a random public IPv4 address by making sure it is not in a private range."""
    while True:
        ip = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
        if not is_private(ip):
            return ip

def create_synthetic_dataset(num_rows=1000, public_ratio=0.1):
    """
    Create a synthetic dataset of IPv4 addresses.
    
    Parameters:
      - num_rows: Total number of rows.
      - public_ratio: Fraction of rows that are public IP addresses.
    
    Returns:
      A pandas DataFrame with columns:
         - ip_address: The generated IPv4 address.
         - ip_type: "public" or "private" indicating the address type.
    """
    ip_addresses = []
    ip_types = []
    
    for _ in range(num_rows):
        # With probability equal to public_ratio, generate a public IP address.
        if random.random() < public_ratio:
            ip = generate_public_ip()
            ip_type = "public"
        else:
            ip = generate_private_ip()
            ip_type = "private"
        ip_addresses.append(ip)
        ip_types.append(ip_type)
    
    df = pd.DataFrame({
        'ip_address': ip_addresses,
        'ip_type': ip_types
    })
    
    return df

if __name__ == "__main__":
    # Generate the dataset with 1,000 rows.
    df = create_synthetic_dataset(num_rows=1000, public_ratio=0.3)
    
    # Optionally, save to CSV.
    df.to_csv("synthetic_ip_dataset.csv", index=False)
    
    # Show the first few rows.
    print("First 5 rows of the synthetic dataset:")
    print(df.head())
    
    # Verify the counts of public vs. private IP addresses.
    print("\nCounts by IP type:")
    print(df['ip_type'].value_counts())
