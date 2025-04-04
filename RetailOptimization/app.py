import streamlit as st
import pandas as pd
import requests
import io

# Define GitHub raw URL for datasets
GITHUB_REPO = "https://raw.githubusercontent.com/dhaneshkoti/RetailOptimization/main/datasets/"

# Function to fetch CSV from GitHub
@st.cache_data
def load_data(file_name):
    """Fetch CSV file from GitHub and return as DataFrame"""
    file_url = GITHUB_REPO + file_name
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raise error for failed requests
        df = pd.read_csv(io.StringIO(response.text))  # Convert text to DataFrame
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network Error: {e}")
        return None
    except pd.errors.ParserError:
        st.error(f"⚠️ Error reading `{file_name}`. Check file format!")
        return None

# Load Datasets
df_inventory = load_data("inventory_monitoring.csv")
df_demand = load_data("demand_forecasting.csv")
df_pricing = load_data("pricing_optimization.csv")

# Check if data loaded correctly
if df_inventory is None or df_demand is None or df_pricing is None:
    st.error("❌ Failed to load one or more datasets. Check file paths or GitHub repository.")

# Standardize column names (removes spaces)
if df_inventory is not None:
    df_inventory.columns = df_inventory.columns.str.strip()
if df_demand is not None:
    df_demand.columns = df_demand.columns.str.strip()
if df_pricing is not None:
    df_pricing.columns = df_pricing.columns.str.strip()

# Convert date columns to datetime (Fixes ValueError)
for df in [df_inventory, df_demand, df_pricing]:
    if df is not None:
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[col] = df[col].astype(int) // 10**9  # Convert to timestamp (numeric)
                except:
                    pass  # Ignore columns that can't be converted

# Sidebar Navigation
st.sidebar.title("Retail Optimization Dashboard")
page = st.sidebar.radio("Choose a module:", ["🏬 Inventory Monitoring", "📈 Demand Forecasting", "💰 Pricing Optimization"])

# 🏬 Inventory Monitoring Page
if page == "🏬 Inventory Monitoring":
    st.title("📊 Inventory Monitoring")
    st.write("Monitor stock levels and identify low-stock products.")

    if df_inventory is not None:
        st.write("### Inventory Data Preview")
        st.dataframe(df_inventory.head())

        if "Stock Levels" in df_inventory.columns:
            low_stock = df_inventory[df_inventory["Stock Levels"] < 10]
            st.write("### ⚠️ Low Stock Alerts")
            st.dataframe(low_stock)
        else:
            st.error("❌ Column 'Stock Levels' is missing!")

# 📈 Demand Forecasting Page
elif page == "📈 Demand Forecasting":
    st.title("📉 Sales Demand Forecasting")
    st.write("Predict sales demand using AI-based forecasting.")

    if df_demand is not None and "Sales Quantity" in df_demand.columns:
        X = df_demand.drop(columns=["Sales Quantity"])  
        y = df_demand["Sales Quantity"]

        X = X.select_dtypes(include=['number'])

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        sample_product = X_test.iloc[0].values.reshape(1, -1)
        prediction = model.predict(sample_product)

        st.write("### 🎯 Predicted Sales Demand")
        st.write(f"Predicted demand for product: **{int(prediction[0])} units**")
    else:
        st.error("❌ Column 'Sales Quantity' is missing!")

# 💰 Pricing Optimization Page
elif page == "💰 Pricing Optimization":
    st.title("💰 AI-Based Pricing Optimization")
    st.write("Optimize pricing strategies based on market trends.")

    if df_pricing is not None and "Base Price" in df_pricing.columns and "Sales Quantity" in df_pricing.columns:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        X = df_pricing.select_dtypes(include=['number']).drop(columns=["Sales Quantity"], errors='ignore')
        y = df_pricing["Sales Quantity"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        sample_price = X_test.iloc[0].values.reshape(1, -1)
        price_prediction = model.predict(sample_price)

        st.write("### 💡 Suggested Optimal Price")
        st.write(f"Recommended Price for Product: **${round(price_prediction[0], 2)}**")
    else:
        st.error("❌ Column 'Base Price' or 'Sales Quantity' is missing! Please check the dataset.")

