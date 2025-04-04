import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ✅ Set dataset directory
DATA_PATH = "datasets"

# ✅ Function to safely load datasets
def load_data(file_name):
    file_path = os.path.join(DATA_PATH, file_name)
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.warning(f"⚠️ `{file_name}` not found! Please upload it below.")
        return None

# ✅ Streamlit UI
st.sidebar.title("Retail Optimization Dashboard")
page = st.sidebar.radio("Choose a module:", ["🏬 Inventory Monitoring", "📈 Demand Forecasting", "💰 Pricing Optimization"])

# ✅ Uploaders for missing datasets
uploaded_inventory = st.file_uploader("Upload Inventory Data (CSV)", type="csv")
uploaded_demand = st.file_uploader("Upload Demand Data (CSV)", type="csv")
uploaded_pricing = st.file_uploader("Upload Pricing Data (CSV)", type="csv")

# ✅ Load datasets (Use uploaded files if available)
if uploaded_inventory:
    df_inventory = pd.read_csv(uploaded_inventory)
else:
    df_inventory = load_data("inventory_monitoring.csv")

if uploaded_demand:
    df_demand = pd.read_csv(uploaded_demand)
else:
    df_demand = load_data("demand_forecasting.csv")

if uploaded_pricing:
    df_pricing = pd.read_csv(uploaded_pricing)
else:
    df_pricing = load_data("pricing_optimization.csv")

# 🏬 **Inventory Monitoring Page**
if page == "🏬 Inventory Monitoring":
    st.title("📊 Inventory Monitoring")
    st.write("Monitor stock levels and identify low-stock products.")

    if df_inventory is not None:
        st.subheader("📋 Inventory Data Preview")
        st.dataframe(df_inventory.head())

        # Check if "Stock Levels" column exists
        if "Stock Levels" in df_inventory.columns:
            # Low Stock Alert
            low_stock = df_inventory[df_inventory["Stock Levels"] < 10]
            st.subheader("⚠️ Low Stock Alerts")
            st.dataframe(low_stock)

            # Stock Levels Chart
            st.subheader("📊 Stock Levels by Product")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=df_inventory["Product ID"], y=df_inventory["Stock Levels"], ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.error("❌ Column 'Stock Levels' missing! Please check dataset.")

# 📈 **Demand Forecasting Page**
elif page == "📈 Demand Forecasting":
    st.title("📉 Sales Demand Forecasting")
    st.write("Predict sales demand using AI-based forecasting.")

    if df_demand is not None:
        st.subheader("📋 Demand Data Preview")
        st.dataframe(df_demand.head())

        # Check if "Sales Quantity" column exists
        if "Sales Quantity" in df_demand.columns:
            # Train a simple model
            X = df_demand.drop(columns=["Sales Quantity"], errors='ignore')  
            y = df_demand["Sales Quantity"]

            if not X.empty:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Predict and display results
                sample_product = X_test.iloc[0].values.reshape(1, -1)
                prediction = model.predict(sample_product)

                st.subheader("🎯 Predicted Sales Demand")
                st.write(f"Predicted demand for product: **{int(prediction[0])} units**")
            else:
                st.error("❌ No valid features found for training! Please check dataset.")
        else:
            st.error("❌ Column 'Sales Quantity' missing! Please check dataset.")

# 💰 **Pricing Optimization Page**
elif page == "💰 Pricing Optimization":
    st.title("💰 AI-Based Pricing Optimization")
    st.write("Optimize pricing strategies based on market trends.")

    if df_pricing is not None:
        st.subheader("📋 Pricing Data Preview")
        st.dataframe(df_pricing.head())

        # Check if required columns exist
        if "Base Price" in df_pricing.columns and "Sales Quantity" in df_pricing.columns:
            # Display Price vs. Sales Chart
            st.subheader("🔥 Price vs. Sales Analysis")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(x=df_pricing["Base Price"], y=df_pricing["Sales Quantity"], ax=ax)
            st.pyplot(fig)

            # Train a simple pricing model
            X = df_pricing.drop(columns=["Sales Quantity"], errors='ignore')  
            y = df_pricing["Sales Quantity"]

            if not X.empty:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Predict optimal pricing
                sample_price = X_test.iloc[0].values.reshape(1, -1)
                price_prediction = model.predict(sample_price)

                st.subheader("💡 Suggested Optimal Price")
                st.write(f"Recommended Price for Product: **${round(price_prediction[0], 2)}**")
            else:
                st.error("❌ No valid features found for training! Please check dataset.")
        else:
            st.error("❌ Columns 'Base Price' or 'Sales Quantity' missing! Please check dataset.")

# ✅ **Run the app using:** `streamlit run app.py`
