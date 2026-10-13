
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Retail Sales ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Sales Machine Learning Dashboard")
st.write("Analyze retail sales using machine learning.")

# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Regression",
        "Classification",
        "Clustering"
    ]
)

# Upload dataset
uploaded_file = st.sidebar.file_uploader(
    "Upload retail sales CSV",
    type=["csv"]
)

df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if page == "Overview":
    st.header("Dataset Overview")

    if df is not None:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Total Features", len(df.columns))
        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Column Information")
        st.write(df.dtypes.astype(str))

    else:
        st.info("Upload your retail sales CSV using the sidebar.")

elif page == "Regression":
    st.header("Regression Analysis")
    st.write("Regression models and evaluation metrics go here.")

elif page == "Classification":
    st.header("Classification Analysis")
    st.write("Return_Flag prediction and classification metrics go here.")

elif page == "Clustering":
    st.header("Customer Segmentation")
    st.write("Clustering models and customer groups go here.")
