import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Define total number of records
n_records = 1000

# Define the common (fraudulent) address that appears in 80% of the records
common_address = '123 Fake St, Faketown, USA'

# Calculate the number of records with the common address (80%) and unique addresses (20%)
n_common = int(0.6 * n_records)
n_unique = n_records - n_common

# Generate unique addresses for the 20% of records.
# For demonstration, we'll generate a random address using a simple template.
street_names = ["Main", "Broad", "Oak", "Pine", "Maple", "Elm", "Cedar"]
cities = ["Springfield", "Centerville", "Franklin", "Greenville", "Madison", "Rivertown"]

def generate_random_address():
    street_number = np.random.randint(100, 999)
    street = np.random.choice(street_names)
    city = np.random.choice(cities)
    return f"{street_number} {street} St, {city}"

# Create the list of unique addresses
unique_addresses = [generate_random_address() for _ in range(n_unique)]

# Combine common and unique addresses into a single list.
# 80% common addresses followed by 20% unique addresses.
addresses = [common_address] * n_common + unique_addresses

# Shuffle the list so that the common and unique addresses are randomly distributed
np.random.shuffle(addresses)

# Create a DataFrame with a customer_id and address columns
df = pd.DataFrame({
    'customer_id': range(1, n_records + 1),
    'address': addresses
})

# Preview the first few rows
print(df.head(10))

# Optionally, save the dataset to a CSV file
df.to_csv('synthetic_fraud_dataset.csv', index=False)
