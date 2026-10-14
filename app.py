```python
import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Retail Sales ML Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "Sales_transactions_2022_2025.csv"
)

REG_PATH = os.path.join(
    BASE_DIR, "models", "regression_bundle_compressed.joblib"
)

CLS_PATH = os.path.join(
    BASE_DIR, "models", "classification_bundle.joblib"
)

st.title("Retail Sales ML Dashboard")
st.caption("Regression and Classification")


# ==================================================
# LOAD FILES
# ==================================================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


if not os.path.exists(DATA_PATH):
    st.error("Dataset CSV file not found.")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Dataset loading failed: {type(e).__name__}: {e}")
    st.stop()


def load_optional_bundle(path, module_name):
    if not os.path.exists(path):
        st.warning(f"{module_name} model file not found: {path}")
        return None

    try:
        return load_bundle(path)
    except Exception as e:
        st.error(
            f"{module_name} model loading failed: "
            f"{type(e).__name__}: {e}"
        )
        return None


reg_bundle = load_optional_bundle(REG_PATH, "Regression")
cls_bundle = load_optional_bundle(CLS_PATH, "Classification")


# ==================================================
# COMMON HELPER FUNCTIONS
# ==================================================

def get_default(feature):
    if feature not in df.columns:
        return 0

    series = df[feature].dropna()

    if series.empty:
        return 0

    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())

    mode = series.mode()

    if not mode.empty:
        return mode.iloc[0]

    return series.iloc[0]


def get_bounds(bundle, feature):
    bounds = bundle.get("iqr_bounds", {})
    item = bounds.get(feature)

    if item is None:
        return None

    if isinstance(item, dict):
        lower = item.get("lower", item.get("lower_bound"))
        upper = item.get("upper", item.get("upper_bound"))

        if lower is not None and upper is not None:
            return lower, upper

    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return item[0], item[1]

    return None


def prepare_features(bundle, values):
    features = bundle.get("features", [])

    row = {}

    for feature in features:
        value = values.get(feature, get_default(feature))
        bounds = get_bounds(bundle, feature)

        if (
            bounds is not None
            and isinstance(value, (int, float, np.number))
        ):
            value = float(
                np.clip(value, bounds[0], bounds[1])
            )

        row[feature] = value

    return pd.DataFrame([row], columns=features)


def predict_saved_model(bundle, model_name, input_df):
    """
    Supports a saved estimator with a separate preprocessor,
    or a saved sklearn Pipeline that preprocesses raw input itself.
    """

    model = bundle["models"][model_name]
    preprocessor = bundle.get("preprocessor")

    # Pipeline usually accepts the original feature DataFrame.
    if hasattr(model, "named_steps"):
        prediction = model.predict(input_df)
        probabilities = (
            model.predict_proba(input_df)[0]
            if hasattr(model, "predict_proba")
            else None
        )
        return prediction[0], probabilities, model

    if preprocessor is None:
        raise ValueError("The saved preprocessor is missing.")

    transformed = preprocessor.transform(input_df)

    prediction = model.predict(transformed)[0]

    probabilities = (
        model.predict_proba(transformed)[0]
        if hasattr(model, "predict_proba")
        else None
    )

    return prediction, probabilities, model


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Regression", "Classification"]
)


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Columns", len(df.columns))
    c3.metric("Missing Values", f"{df.isna().sum().sum():,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Numerical Summary")
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(
            missing.rename("Missing Count").to_frame(),
            use_container_width=True
        )


# ==================================================
# REGRESSION
# ==================================================

elif page == "Regression":

    st.header("Sales Regression")
    st.write("Select a saved model and enter transaction details.")

    if not reg_bundle:
        st.error("Regression bundle could not be loaded.")
        st.stop()

    reg_models = reg_bundle.get("models", {})
    reg_features = reg_bundle.get("features", [])

    if not reg_models or not reg_features:
        st.error("Regression bundle is missing models or features.")
        st.stop()

    model_name = st.selectbox(
        "Regression Algorithm",
        list(reg_models.keys())
    )

    values = {}

    numeric_fields = {
        "Quantity": (1.0, 1.0),
        "Unit_Price": (100.0, 10.0),
        "Discount_Percentage": (0.0, 1.0),
        "Customer_Age": (30.0, 1.0),
        "Delivery_Days": (3.0, 1.0),
        "Customer_Rating": (4.0, 0.5)
    }

    categorical_fields = [
        "Product_Category",
        "Customer_Segment",
        "Sales_Channel",
        "Payment_Method"
    ]

    with st.form("regression_form"):

        left, right = st.columns(2)

        for i, (feature, settings) in enumerate(
            numeric_fields.items()
        ):
            if feature not in reg_features:
                continue

            default, step = settings

            if feature in df.columns:
                try:
                    default = float(get_default(feature))
                except (ValueError, TypeError):
                    pass

            with (left if i % 2 == 0 else right):
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=0.0,
                    value=default,
                    step=step,
                    key=f"reg_{feature}"
                )

        for i, feature in enumerate(categorical_fields):
            if feature not in reg_features:
                continue

            with (left if i % 2 == 0 else right):
                options = (
                    df[feature].dropna().unique().tolist()
                    if feature in df.columns
                    else []
                )

                if options:
                    default = get_default(feature)
                    index = (
                        options.index(default)
                        if default in options
                        else 0
                    )

                    values[feature] = st.selectbox(
                        feature.replace("_", " "),
                        options,
                        index=index,
                        key=f"reg_cat_{feature}"
                    )

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            quantity = values.get(
                "Quantity", get_default("Quantity")
            )
            unit_price = values.get(
                "Unit_Price", get_default("Unit_Price")
            )
            discount = values.get(
                "Discount_Percentage",
                get_default("Discount_Percentage")
            )

            gross = quantity * unit_price
            discount_amount = gross * discount / 100.0

            if "Gross_Amount_Before_Discount" in reg_features:
                values["Gross_Amount_Before_Discount"] = gross

            if "Discount_Amount" in reg_features:
                values["Discount_Amount"] = discount_amount

            input_df = prepare_features(reg_bundle, values)

            prediction, _, _ = predict_saved_model(
                reg_bundle,
                model_name,
                input_df
            )

            st.subheader("Prediction Result")
            st.metric(
                "Predicted Value",
                f"{float(prediction):,.2f}"
            )

            st.caption(
                "Unentered features use dataset defaults. "
                "This is a model estimate, not a guaranteed result."
            )

        except Exception as e:
            st.error(
                f"Regression prediction failed: "
                f"{type(e).__name__}: {e}"
            )


# ==================================================
# CLASSIFICATION
# ==================================================

elif page == "Classification":

    st.header("Return Status Classification")
    st.write("Predict the return class using a saved classifier.")

    if not cls_bundle:
        st.error("Classification bundle could not be loaded.")
        st.stop()

    cls_models = cls_bundle.get("models", {})
    cls_features = cls_bundle.get("features", [])

    if not cls_models or not cls_features:
        st.error(
            "Classification bundle is missing models or features."
        )
        st.stop()

    # Display only models actually stored in the bundle.
    model_name = st.selectbox(
        "Classification Algorithm",
        list(cls_models.keys())
    )

    values = {}

    numeric_fields = {
        "Quantity": (1.0, 1.0),
        "Unit_Price": (100.0, 10.0),
        "Discount_Percentage": (0.0, 1.0),
        "Customer_Age": (30.0, 1.0),
        "Delivery_Days": (3.0, 1.0),
        "Customer_Rating": (4.0, 0.5)
    }

    categorical_fields = [
        "Product_Category",
        "Customer_Segment",
        "Sales_Channel",
        "Payment_Method"
    ]

    with st.form("classification_form"):

        left, right = st.columns(2)

        for i, (feature, settings) in enumerate(
            numeric_fields.items()
        ):
            if feature not in cls_features:
                continue

            default, step = settings

            if feature in df.columns:
                try:
                    default = float(get_default(feature))
                except (ValueError, TypeError):
                    pass

            with (left if i % 2 == 0 else right):
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=0.0,
                    value=default,
                    step=step,
                    key=f"cls_{feature}"
                )

        for i, feature in enumerate(categorical_fields):
            if feature not in cls_features:
                continue

            with (left if i % 2 == 0 else right):
                options = (
                    df[feature].dropna().unique().tolist()
                    if feature in df.columns
                    else []
                )

                if options:
                    default = get_default(feature)
                    index = (
                        options.index(default)
                        if default in options
                        else 0
                    )

                    values[feature] = st.selectbox(
                        feature.replace("_", " "),
                        options,
                        index=index,
                        key=f"cls_cat_{feature}"
                    )

        submitted = st.form_submit_button(
            "Predict Return Status"
        )

    if submitted:
        try:
            input_df = prepare_features(cls_bundle, values)

            prediction, probabilities, model = predict_saved_model(
                cls_bundle,
                model_name,
                input_df
            )

            st.subheader("Classification Result")
            st.success(f"Predicted class: {prediction}")

            if probabilities is not None:
                classes = getattr(model, "classes_", None)

                if classes is not None:
                    probability_df = pd.DataFrame({
                        "Class": classes,
                        "Probability (%)": probabilities * 100
                    }).sort_values(
                        "Probability (%)",
                        ascending=False
                    )

                    st.subheader("Class Probabilities")
                    st.dataframe(
                        probability_df,
                        use_container_width=True
                    )

            st.caption(
                "Unentered features use dataset defaults. "
                "The prediction depends on the saved model and "
                "the quality of its training data."
            )

        except Exception as e:
            st.error(
                f"Classification prediction failed: "
                f"{type(e).__name__}: {e}"
            )


# ==================================================
# FOOTER
# ==================================================

st.sidebar.divider()
st.sidebar.caption("Retail Sales ML Dashboard")
```
