import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the CSV data into a Pandas DataFrame
# file_path = 'replace with dataset path'
heart_df = pd.read_csv(file_path)

# Split the data into features and target variable
X = heart_df.iloc[:, :-1]
y = heart_df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an instance of the SVC class with the best hyperparameters
best_model = SVC(C=0.1, gamma='scale', kernel='linear', probability=True)

# Train the model
best_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Model trained successfully.")

# Saving the model and scaler
model_file_path = 'D:/BYTEWISE_2024/Tasks/Week8-9/BTW_Project2_Heart_disease_Prediction/heart_disease_model.pkl'
scaler_file_path = 'D:/BYTEWISE_2024/Tasks/Week8-9/BTW_Project2_Heart_disease_Prediction/scaler.pkl'

# Save the trained model and scaler
joblib.dump(best_model, model_file_path)
joblib.dump(scaler, scaler_file_path )
