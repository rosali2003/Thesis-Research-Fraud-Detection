import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import DataLoader, TensorDataset

# Download latest version
path = kagglehub.dataset_download("mishra5001/credit-card")

print("Path to dataset files:", path)

# Load the Application Data set (replace with your actual file path)
data = pd.read_csv(f"{path}/application_data.csv")  # Adjust the path as needed

# Display basic information about the data
print("Data Shape:", data.shape)
print("Data Columns:", data.columns)
print(data.head())

# Separate features and target
X = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)  # Drop ID and target columns
y = data['TARGET']

# Handle categorical variables with one-hot encoding and label encoding for binary variables
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Impute missing values with median for numeric columns
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the dataset into training (20%) and testing (80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)