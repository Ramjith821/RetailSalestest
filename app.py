```python
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Sales - Regression",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Retail Sales Prediction")
st.write("Predict the target value using a trained regression model.")

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "Sales_transactions_2022_2025.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "regression_bundle_compressed.joblib"
)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Regression model not found: {MODEL_PATH}")
    st.stop()

try:
    df = load_data(DATA_PATH)
    bundle = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load files: {e}")
    st.stop()

models = bundle.get("models", {})
preprocessor = bundle.get("preprocessor")
features = bundle.get("features", [])

if not models or preprocessor is None or not features:
    st.error("The regression bundle is missing required components.")
    st.stop()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

with st.expander("View Dataset"):
    st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Total Features", len(df.columns))
col3.metric("Missing Values", f"{df.isna().sum().sum():,}")

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_default(feature):
    if feature not in df.columns:
        return 0

    series = df[feature].dropna()

    if series.empty:
        return 0

    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())

    return series.mode().iloc[0]


def clip_to_bounds(feature, value):
    bounds = bundle.get("iqr_bounds", {})
    item = bounds.get(feature)

    if item is None:
        return value

    lower = None
    upper = None

    if isinstance(item, dict):
        lower = item.get("lower", item.get("lower_bound"))
        upper = item.get("upper", item.get("upper_bound"))
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        lower, upper = item[0], item[1]

    if (
        lower is not None
        and upper is not None
        and isinstance(value, (int, float, np.number))
    ):
        return float(np.clip(value, lower, upper))

    return value


# --------------------------------------------------
# REGRESSION INPUT FORM
# --------------------------------------------------

st.divider()
st.subheader("Enter Transaction Details")

model_name = st.selectbox(
    "Choose Regression Algorithm",
    list(models.keys())
)

values = {}

numeric_fields = {
    "Quantity": (1.0, 1.0),
    "Unit_Price": (100.0, 10.0),
    "Discount_Percentage": (0.0, 1.0),
    "Customer_Age": (30.0, 1.0),
    "Delivery_Days": (3.0, 1.0),
    "Customer_Rating": (4.0, 0.5),
}

categorical_fields = [
    "Product_Category",
    "Customer_Segment",
    "Sales_Channel",
    "Payment_Method",
]

with st.form("regression_form"):

    left, right = st.columns(2)

    for i, (feature, settings) in enumerate(numeric_fields.items()):
        if feature in features:
            default, step = settings

            if feature in df.columns:
                try:
                    default = float(get_default(feature))
                except (TypeError, ValueError):
                    pass

            with (left if i % 2 == 0 else right):
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=0.0,
                    value=default,
                    step=step,
                    key=f"reg_{feature}",
                )

    for i, feature in enumerate(categorical_fields):
        if feature in features:
            with (left if i % 2 == 0 else right):
                if feature in df.columns:
                    options = df[feature].dropna().unique().tolist()
                else:
                    options = []

                if options:
                    default = get_default(feature)
                    index = options.index(default) if default in options else 0

                    values[feature] = st.selectbox(
                        feature.replace("_", " "),
                        options,
                        index=index,
                        key=f"reg_cat_{feature}",
                    )
                else:
                    values[feature] = st.text_input(
                        feature.replace("_", " "),
                        key=f"reg_text_{feature}",
                    )

    submitted = st.form_submit_button("Predict Sales")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:
    try:
        # Fill unentered features with dataset defaults.
        for feature in features:
            if feature not in values:
                values[feature] = get_default(feature)

        # Derive common amount features when present.
        quantity = values.get("Quantity", get_default("Quantity"))
        unit_price = values.get("Unit_Price", get_default("Unit_Price"))
        discount = values.get(
            "Discount_Percentage",
            get_default("Discount_Percentage"),
        )

        gross_amount = quantity * unit_price
        discount_amount = gross_amount * discount / 100.0

        if "Gross_Amount_Before_Discount" in features:
            values["Gross_Amount_Before_Discount"] = gross_amount

        if "Discount_Amount" in features:
            values["Discount_Amount"] = discount_amount

        # Apply stored bounds to numeric inputs.
        for feature in features:
            values[feature] = clip_to_bounds(
                feature, values[feature]
            )

        # Match the exact feature order expected by training.
        input_df = pd.DataFrame(
            [[values[feature] for feature in features]],
            columns=features,
        )

        transformed = preprocessor.transform(input_df)
        model = models[model_name]
        prediction = model.predict(transformed)[0]

        st.divider()
        st.subheader("Prediction Result")
        st.metric("Predicted Value", f"{float(prediction):,.2f}")

        st.caption(
            "Features not entered manually use dataset defaults. "
            "The result is an estimate from the selected trained model."
        )

    except Exception as e:
        st.error(f"Prediction failed: {type(e).__name__}: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.title("Navigation")
st.sidebar.info("Regression module only")
```
