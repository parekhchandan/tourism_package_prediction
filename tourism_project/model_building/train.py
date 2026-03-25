# for data manipulation
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer # Added for fixing the NaN error
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

repo_id = "Chandan2312/tourism-package-prediction"
repo_type = "dataset"

# Ensure folder exists
os.makedirs("tourism_project/model_building", exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Update Hugging Face dataset paths
Xtrain_path = hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type=repo_type)
Xtest_path = hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type=repo_type)
ytrain_path = hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type=repo_type)
ytest_path = hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type=repo_type)

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# Numerical features
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',
    'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]

# Categorical features
categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'MaritalStatus',
    'Designation', 'ProductPitched'
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# IMPROVED: Preprocessing pipeline with Imputers to prevent the 'isnan' error
numeric_transformer = make_pipeline(
    SimpleImputer(strategy='median'), # Handles missing numbers
    StandardScaler()
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'), # Handles missing strings
    OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False is safer for XGBoost
)

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features)
)

# Base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid (keys must match the step name from make_pipeline)
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save model locally
    model_path = "best_customer_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved and logged to MLflow.")

    # Upload to Hugging Face
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model",
    )
