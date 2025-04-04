import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
dataset_path = "/Users/dhaneshchintala/Desktop/RetailOptimization/datasets/demand_forecasting.csv"
df = pd.read_csv(dataset_path)

# Print available columns
print("Dataset Columns:", df.columns)

# Define Features (X) and Target (y)
X = df.drop(columns=['Sales Quantity', 'Date', 'Product ID', 'Store ID'])  # Exclude non-numeric columns
y = df['Sales Quantity']

# Convert categorical columns to numeric
X = pd.get_dummies(X)  # One-hot encoding for categorical data

# Split into Train & Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model training completed!")
