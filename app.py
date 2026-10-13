
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Retail Sales ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales ML Dashboard")
st.write("Regression, Classification and Customer Clustering")

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


# ---------------- HELPER FUNCTIONS ----------------

def category_options(df, column):
    if column in df.columns:
        values = df[column].dropna().astype(str).unique().tolist()
        if values:
            return sorted(values)

    return ["Unknown"]


def build_regression_input(
    df,
    features,
    quantity,
    unit_price,
    discount,
    customer_age,
    delivery_days,
    rating,
    product_category,
    customer_segment,
    sales_channel,
    payment_method
):
    now = datetime.now()

    gross_amount = quantity * unit_price
    discount_amount = gross_amount * discount / 100

    fallbacks = {
        "Customer_Age": customer_age,
        "Quantity": quantity,
        "Unit_Price": unit_price,
        "Discount_Percentage": discount,
        "Delivery_Days": delivery_days,
        "Customer_Rating": rating,
        "Product_Category": product_category,
        "Customer_Segment": customer_segment,
        "Sales_Channel": sales_channel,
        "Payment_Method": payment_method,
        "Gross_Amount_Before_Discount": gross_amount,
        "Discount_Amount": discount_amount,
        "Order_Year": now.year,
        "Order_Month": now.month,
        "Order_Day": now.day,
        "Order_DayOfWeek": now.weekday(),
        "Order_Quarter": (now.month - 1) // 3 + 1,
        "Order_Hour": now.hour,
        "Is_Weekend": int(now.weekday() >= 5),
    }

    row = {}

    for feature in features:
        if feature in fallbacks:
            row[feature] = fallbacks[feature]

        elif feature in df.columns:
            series = df[feature].dropna()

            if pd.api.types.is_numeric_dtype(df[feature]):
                row[feature] = (
                    float(series.median()) if not series.empty else 0.0
                )
            else:
                modes = series.mode()
                row[feature] = (
                    modes.iloc[0] if not modes.empty else "Unknown"
                )

        else:
            row[feature] = "Unknown"

    return pd.DataFrame([row], columns=features)


# ---------------- NAVIGATION ----------------

page = st.sidebar.radio(
    "Choose ML Task",
    ["Overview", "Regression", "Classification", "Clustering"]
)


# ---------------- MAIN APP ----------------

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

        st.subheader("Summary Statistics")
        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

    # ==================================================
    # REGRESSION
    # ==================================================

    elif page == "Regression":
        st.subheader("💰 Profit Prediction")
        st.write("Enter transaction details to estimate profit.")

        features = reg_bundle["features"]
        models = reg_bundle["models"]
        preprocessor = reg_bundle["preprocessor"]
        bounds = reg_bundle.get("iqr_bounds", {})

        with st.form("profit_prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                quantity = st.number_input(
                    "Quantity", min_value=1, value=2
                )
                unit_price = st.number_input(
                    "Unit Price", min_value=0.0, value=100.0,
                    step=10.0
                )
                discount = st.number_input(
                    "Discount Percentage", min_value=0.0,
                    max_value=100.0, value=10.0
                )
                customer_age = st.number_input(
                    "Customer Age", min_value=1, max_value=100,
                    value=30
                )
                delivery_days = st.number_input(
                    "Delivery Days", min_value=0, value=3
                )

            with col2:
                rating = st.slider(
                    "Customer Rating", min_value=1.0,
                    max_value=5.0, value=4.0, step=0.5
                )
                product_category = st.selectbox(
                    "Product Category",
                    category_options(df, "Product_Category")
                )
                customer_segment = st.selectbox(
                    "Customer Segment",
                    category_options(df, "Customer_Segment")
                )
                sales_channel = st.selectbox(
                    "Sales Channel",
                    category_options(df, "Sales_Channel")
                )
                payment_method = st.selectbox(
                    "Payment Method",
                    category_options(df, "Payment_Method")
                )

            # Exclude polynomial model until its specific
            # transformation pipeline is available.
            available_models = {
                name: model
                for name, model in models.items()
                if "Polynomial Regression" not in name
            }

            if not available_models:
                available_models = models

            model_name = st.selectbox(
                "Regression Model",
                list(available_models.keys()),
                index=(
                    list(available_models.keys()).index(
                        "Random Forest Regressor"
                    )
                    if "Random Forest Regressor" in available_models
                    else 0
                )
            )

            submitted = st.form_submit_button(
                "Predict Profit",
                use_container_width=True
            )

        if submitted:
            try:
                input_df = build_regression_input(
                    df,
                    features,
                    quantity,
                    unit_price,
                    discount,
                    customer_age,
                    delivery_days,
                    rating,
                    product_category,
                    customer_segment,
                    sales_channel,
                    payment_method
                )

                for feature, limits in bounds.items():
                    if feature in input_df.columns:
                        input_df[feature] = input_df[feature].clip(
                            lower=limits[0],
                            upper=limits[1]
                        )

                transformed_input = preprocessor.transform(input_df)
                prediction = available_models[model_name].predict(
                    transformed_input
                )[0]

                st.success(f"Predicted Profit: {prediction:,.2f}")
                st.caption(f"Model used: {model_name}")
                st.warning(
                    "Other features use representative or derived values. "
                    "The model was trained using its original feature set, "
                    "so these defaults can affect prediction accuracy."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ==================================================
    # CLASSIFICATION
    # ==================================================

    elif page == "Classification":
        st.subheader("Return Status Prediction")

        st.write("Classification bundle keys:")
        st.json(list(clf_bundle.keys()))

        st.write("Classification bundle details:")
        st.write(clf_bundle)

        st.info(
            "This diagnostic view helps us identify the saved "
            "classification model, preprocessing pipeline and features "
            "before building the prediction form."
        )

    # ==================================================
    # CLUSTERING
    # ==================================================

    elif page == "Clustering":
        st.subheader("👥 Customer Segmentation")

        st.info(
            "Customer clustering will be connected after inspecting "
            "the saved clustering bundle."
        )

        st.write("Clustering bundle keys:")
        st.json(list(cluster_bundle.keys()))

# ---------------- ERROR HANDLING ----------------

except Exception as e:
    st.error(f"Could not load the dataset or models: {e}")

    st.info(
        "Verify that the CSV file and all three model files exist "
        "in the expected repository locations."
    )
