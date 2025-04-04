import pandas as pd

# Load dataset
dataset_path = "/Users/dhaneshchintala/Desktop/RetailOptimization/datasets/pricing_optimization.csv"
df_pricing = pd.read_csv(dataset_path)

# Print available columns
print("Available Columns:", df_pricing.columns)

# Forward fill missing values (Updated)
df_pricing.ffill(inplace=True)

# Expected feature columns
expected_features = ['Base Price', 'Competitor Price', 'Return Rate']

# Check if features exist
missing_features = [col for col in expected_features if col not in df_pricing.columns]

if missing_features:
    print(f"⚠️ Warning: Missing columns in dataset: {missing_features}")
    # Provide a default or alternative handling
    for col in missing_features:
        df_pricing[col] = 0  # Replace with a meaningful default value

# Now, extract feature set
X = df_pricing[expected_features]
print(X.head())
