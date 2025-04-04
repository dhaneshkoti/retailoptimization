import os
import pandas as pd

# Load dataset
dataset_path = "/Users/dhaneshchintala/Desktop/RetailOptimization/datasets/inventory_monitoring.csv"
df_inventory = pd.read_csv(dataset_path)

# Print available columns
print("Available Columns:", df_inventory.columns)

# Check if 'Date' column exists
if 'Date' in df_inventory.columns:
    df_inventory['Date'] = pd.to_datetime(df_inventory['Date'])
else:
    print("⚠️ Warning: 'Date' column not found! Adding a generated timestamp.")
    df_inventory['Date'] = pd.to_datetime("today")  # Generate today's date

print(df_inventory.head())
