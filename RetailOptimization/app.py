import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # If you're using a trained model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests


GITHUB_REPO = "https://raw.githubusercontent.com/dhaneshkoti/RetailOptimization/main/datasets/"

@st.cache_data
def load_data(file_name):
    """Fetch CSV from GitHub & return as DataFrame"""
    file_url = GITHUB_REPO + file_name
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raise error if request fails
        df = pd.read_csv(pd.compat.StringIO(response.text))  # Convert text to CSV
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network Error: {e}")
        return None
    except pd.errors.ParserError:
        st.error(f"⚠️ Error reading `{file_name}`. Check file format!")
        return None


# ✅ Load datasets from GitHub
df_inventory = load_data("inventory_monitoring.csv")
df_demand = load_data("demand_forecasting.csv")
df_pricing = load_data("pricing_optimization.csv")

# ✅ Standardize column names (removes spaces)
for df in [df_inventory, df_demand, df_pricing]:
    if df is not None:
        df.columns = df.columns.str.strip()

# ✅ Convert date columns to datetime (Fixes ValueError)
for df in [df_inventory, df_demand, df_pricing]:
    if df is not None:
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[col] = df[col].astype(int) // 10**9  # Convert to timestamp (numeric)
                except:
                    pass  # Ignore columns that can't be converted

# ✅ Sidebar Navigation
st.sidebar.title("Retail Optimization Dashboard")
page = st.sidebar.radio("Choose a module:", ["🏬 Inventory Monitoring", "📈 Demand Forecasting", "💰 Pricing Optimization"])

# 🏬 **Inventory Monitoring Page**
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

            # 📊 Stock Levels Chart
            st.write("### 📊 Stock Levels by Product")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Product ID", y="Stock Levels", data=df_inventory.head(10), ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.error("❌ Column 'Stock Levels' is missing!")

# 📈 **Demand Forecasting Page**
elif page == "📈 Demand Forecasting":
    st.title("📉 Sales Demand Forecasting")
    st.write("Predict sales demand using AI-based forecasting.")

    if df_demand is not None:
        st.write("### Demand Data Preview")
        st.dataframe(df_demand.head())

        if "Sales Quantity" in df_demand.columns:
            X = df_demand.drop(columns=["Sales Quantity"])  
            y = df_demand["Sales Quantity"]

            X = X.select_dtypes(include=['number'])  # Ensure numeric features

            # ✅ Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # ✅ Predict and display results
            sample_product = X_test.iloc[0].values.reshape(1, -1)
            prediction = model.predict(sample_product)
            
            st.write("### 🎯 Predicted Sales Demand")
            st.write(f"Predicted demand for product: **{int(prediction[0])} units**")
        else:
            st.error("❌ Column 'Sales Quantity' is missing!")

# 💰 **Pricing Optimization Page**
elif page == "💰 Pricing Optimization":
    st.title("💰 AI-Based Pricing Optimization")
    st.write("Optimize pricing strategies based on market trends.")

    if df_pricing is not None:
        st.write("### Pricing Data Preview")
        st.dataframe(df_pricing.head())

        st.write("🔍 Column Names in Pricing Data:", df_pricing.columns.tolist())

        if "Base Price" in df_pricing.columns and "Sales Quantity" in df_pricing.columns:
            # 📊 Price vs. Sales Chart
            st.write("### 🔥 Price vs. Sales Analysis")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(x=df_pricing["Base Price"], y=df_pricing["Sales Quantity"], ax=ax)
            st.pyplot(fig)

            # ✅ Train a simple pricing model
            X = df_pricing.select_dtypes(include=['number']).drop(columns=["Sales Quantity"], errors='ignore')
            y = df_pricing["Sales Quantity"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # ✅ Predict optimal pricing
            sample_price = X_test.iloc[0].values.reshape(1, -1)
            price_prediction = model.predict(sample_price)

            st.write("### 💡 Suggested Optimal Price")
            st.write(f"Recommended Price for Product: **${round(price_prediction[0], 2)}**")
        else:
            st.error("❌ Column 'Base Price' or 'Sales Quantity' is missing! Please check the dataset.")
