
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


repo_id = "Chandan2312/tourism-package-prediction"
repo_type = "dataset"

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Chandan2312/tourism-package-prediction/tourism.csv"
customer_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# 1. Dropped the unique identifier 'CustomerID' from column selections.
# 2. Dropped the unique identifier - first column in the dataset, one without any header.

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                       # Age of the customer
    'NumberOfPersonVisiting',    # Total number of people accompanying
    'PreferredPropertyStar',     # Preferred hotel rating
    'NumberOfTrips',             # Average annual trips
    'Passport',                  # Holds a valid passport (0/1)
    'OwnCar',                    # Owns a car (0/1)
    'NumberOfChildrenVisiting',  # Children below age 5 accompanying
    'MonthlyIncome',             # Gross monthly income
    'PitchSatisfactionScore',    # Satisfaction score with sales pitch
    'NumberOfFollowups',         # Total follow-ups after pitch
    'DurationOfPitch'            # Duration of sales pitch
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',             # Method of contact
    'CityTier',                  # City category
    'Occupation',                # Occupation type
    'Gender',                    # Gender
    'MaritalStatus',             # Marital status
    'Designation',               # Designation in organization
    'ProductPitched'             # Type of product pitched
]

# Define predictor matrix (X) using selected numeric and categorical features
X = customer_dataset[numeric_features + categorical_features]

# Define target variable
y = customer_dataset[target]

# Split dataset into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility
)

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type=repo_type,
    )
