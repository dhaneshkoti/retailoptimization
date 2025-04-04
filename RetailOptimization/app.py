import streamlit as st
import pandas as pd
import os

# ✅ Get the directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets")

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
    if df_inventory is not None:
        st.dataframe(df_inventory.head())

# 📈 **Demand Forecasting Page**
elif page == "📈 Demand Forecasting":
    st.title("📉 Sales Demand Forecasting")
    if df_demand is not None:
        st.dataframe(df_demand.head())

# 💰 **Pricing Optimization Page**
elif page == "💰 Pricing Optimization":
    st.title("💰 AI-Based Pricing Optimization")
    if df_pricing is not None:
        st.dataframe(df_pricing.head())
