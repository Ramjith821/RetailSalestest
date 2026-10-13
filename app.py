
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Retail Sales ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales ML Dashboard")
st.write("Regression, Classification and Customer Clustering")

# ---------------- FILE PATHS ----------------

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Sales_transactions_2022_2025.csv"
MODEL_DIR = BASE_DIR / "models"

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ---------------- LOAD MODELS ----------------

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


# ---------------- SIDEBAR NAVIGATION ----------------

page = st.sidebar.radio(
    "Choose ML Task",
    ["Overview", "Regression", "Classification", "Clustering"]
)


# ---------------- MAIN APPLICATION ----------------

try:
    df = load_data()
    reg_bundle, clf_bundle, cluster_bundle = load_models()

    # ==================================================
    # OVERVIEW
    # ==================================================

    if page == "Overview":

        st.subheader("Dataset Overview")

        col1, col2 = st.columns(2)

        col1.metric("Total Rows", f"{df.shape[0]:,}")
        col2.metric("Total Columns", df.shape[1])

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(include="all").T)

    # ==================================================
    # REGRESSION
    # ==================================================

    elif page == "Regression":

        st.subheader("💰 Profit Prediction")

        st.write(
            "Enter transaction details to estimate the profit "
            "using your trained regression model."
        )

        features = reg_bundle["features"]
        models = reg_bundle["models"]
        preprocessor = reg_bundle["preprocessor"]
        bounds = reg_bundle["iqr_bounds"]

        # ---------- CATEGORY OPTIONS ----------

        def category_options(column):
            if column in df.columns:
                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if values:
                    return sorted(values)

            return ["Unknown"]

        # ---------- INPUT FORM: 10 FEATURES ----------

        with st.form("regression_form"):

            st.markdown("### Transaction Details")

            col1, col2 = st.columns(2)

            with col1:

                quantity = st.number_input(
                    "Quantity",
                    min_value=1,
                    value=2
                )

                unit_price = st.number_input(
                    "Unit Price",
                    min_value=0.0,
                    value=100.0,
                    step=10.0
                )

                discount = st.number_input(
                    "Discount Percentage",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0
                )

                customer_age = st.number_input(
                    "Customer Age",
                    min_value=1,
                    max_value=100,
                    value=30
                )

                delivery_days = st.number_input(
                    "Delivery Days",
                    min_value=0,
                    value=3
                )

            with col2:

                rating = st.slider(
                    "Customer Rating",
                    min_value=1.0,
                    max_value=5.0,
                    value=4.0,
                    step=0.5
                )

                product_category = st.selectbox(
                    "Product Category",
                    category_options("Product_Category")
                )

                customer_segment = st.selectbox(
                    "Customer Segment",
                    category_options("Customer_Segment")
                )

                sales_channel = st.selectbox(
                    "Sales Channel",
                    category_options("Sales_Channel")
                )

                payment_method = st.selectbox(
                    "Payment Method",
                    category_options("Payment_Method")
                )

            model_name = st.selectbox(
                "Choose Regression Model",
                list(models.keys()),
                index=(
                    list(models.keys()).index(
                        "Random Forest Regressor"
                    )
                    if "Random Forest Regressor" in models
                    else 0
                )
            )

            submitted = st.form_submit_button(
                "Predict Profit",
                use_container_width=True
            )

        # ---------- MAKE PREDICTION ----------

        if submitted:

            try:
                # Fill all model features with representative values.
                # The 10 inputs below will replace their corresponding defaults.

                input_row = {}

                for feature in features:

                    if feature not in df.columns:
                        raise ValueError(
                            f"Feature '{feature}' is missing from the dataset."
                        )

                    series = df[feature].dropna()

                    if pd.api.types.is_numeric_dtype(df[feature]):

                        input_row[feature] = (
                            float(series.median())
                            if not series.empty
                            else 0.0
                        )

                    else:

                        mode_values = series.mode()

                        input_row[feature] = (
                            mode_values.iloc[0]
                            if not mode_values.empty
                            else "Unknown"
                        )

                # Replace defaults with user inputs

                input_row.update({
                    "Quantity": quantity,
                    "Unit_Price": unit_price,
                    "Discount_Percentage": discount,
                    "Customer_Age": customer_age,
                    "Delivery_Days": delivery_days,
                    "Customer_Rating": rating,
                    "Product_Category": product_category,
                    "Customer_Segment": customer_segment,
                    "Sales_Channel": sales_channel,
                    "Payment_Method": payment_method,
                })

                input_df = pd.DataFrame(
                    [input_row],
                    columns=features
                )

                # Apply the saved training-time outlier bounds

                for feature, limits in bounds.items():

                    if feature in input_df.columns:

                        input_df[feature] = input_df[feature].clip(
                            lower=limits[0],
                            upper=limits[1]
                        )

                # Apply saved preprocessing

                transformed_input = preprocessor.transform(input_df)

                # Predict profit

                selected_model = models[model_name]

                prediction = selected_model.predict(
                    transformed_input
                )[0]

                st.success(
                    f"### Predicted Profit: {prediction:,.2f}"
                )

                st.caption(
                    f"Model used: {model_name}"
                )

                st.warning(
                    "The other model features use representative values "
                    "from the dataset. Actual profit may differ. "
                    "This version has not retrained the model using "
                    "only the 10 displayed inputs."
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

    # ==================================================
    # CLASSIFICATION
    # ==================================================

    elif page == "Classification":

        st.subheader("Return Status Prediction")

        st.info(
            "The classification prediction form will be implemented next."
        )

        st.write(
            "This section will predict whether a transaction is likely "
            "to be returned."
        )

    # ==================================================
    # CLUSTERING
    # ==================================================

    elif page == "Clustering":

        st.subheader("👥 Customer Segmentation")

        st.info(
            "The customer clustering interface will be implemented next."
        )

        st.write(
            "This section will assign customers to groups based on "
            "their characteristics and purchasing behavior."
        )

# ---------------- ERROR HANDLING ----------------

except Exception as e:

    st.error(
        f"Could not load the dataset or models: {e}"
    )

    st.info(
        "Check that the CSV file and all three model files exist "
        "in the expected repository locations."
    )
