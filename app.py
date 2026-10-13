
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Retail Sales ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Sales ML Dashboard")
st.write("Regression, Classification and Customer Clustering")

# File paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Sales_transactions_2022_2025.csv"
MODEL_DIR = BASE_DIR / "models"

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Load model bundles
@st.cache_resource
def load_models():
    regression = joblib.load(
        MODEL_DIR / "regression_bundle_compressed.joblib"
    )
    classification = joblib.load(
        MODEL_DIR / "classification_bundle.joblib"
    )
    clustering = joblib.load(
        MODEL_DIR / "clustering_bundle.joblib"
    )
    return regression, classification, clustering

# Sidebar navigation
page = st.sidebar.radio(
    "Choose ML Task",
    ["Overview", "Regression", "Classification", "Clustering"]
)

try:
    df = load_data()
    reg_bundle, clf_bundle, cluster_bundle = load_models()

    if page == "Overview":
        st.subheader("Dataset Overview")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.dataframe(df.head())

    elif page == "Regression":
        st.subheader("Profit Prediction")
        st.info("Next, we will connect your trained regression model to user inputs.")

    elif page == "Classification":
        st.subheader("Return Status Prediction")
        st.info("Next, we will connect your trained classification model to user inputs.")

    elif page == "Clustering":
        st.subheader("Customer Segmentation")
        st.info("Next, we will connect your trained clustering model to user inputs.")

except Exception as e:
    st.error(f"Could not load the dataset or models: {e}")
    st.info(
        "Check that the CSV and all three model files exist "
        "at the paths shown in the repository structure."
    )
