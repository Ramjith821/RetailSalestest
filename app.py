
import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Retail Sales ML Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Sales_transactions_2022_2025.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

REG_PATH = os.path.join(MODEL_DIR, "regression_bundle_compressed.joblib")
CLS_PATH = os.path.join(MODEL_DIR, "classification_bundle.joblib")
CLUSTER_PATH = os.path.join(MODEL_DIR, "clustering_bundle.joblib")

st.title("Retail Sales ML Dashboard")
st.caption("Regression • Classification • Clustering")


# ============================================================
# LOAD FILES
# ============================================================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


if not os.path.exists(DATA_PATH):
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Dataset loading failed: {type(exc).__name__}: {exc}")
    st.stop()


def load_optional_bundle(path, name):
    if not os.path.exists(path):
        st.warning(f"{name} bundle not found: {path}")
        return None
    try:
        return load_bundle(path)
    except Exception as exc:
        st.error(f"{name} bundle loading failed: {type(exc).__name__}: {exc}")
        return None


reg_bundle = load_optional_bundle(REG_PATH, "Regression")
cls_bundle = load_optional_bundle(CLS_PATH, "Classification")
cluster_bundle = load_optional_bundle(CLUSTER_PATH, "Clustering")


# ============================================================
# COMMON HELPERS
# ============================================================
def get_default(feature):
    if feature not in df.columns:
        return 0

    series = df[feature].dropna()
    if series.empty:
        return 0

    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())

    modes = series.mode()
    return modes.iloc[0] if not modes.empty else series.iloc[0]


def get_bounds(bundle, feature):
    bounds = bundle.get("iqr_bounds", {})
    item = bounds.get(feature) if isinstance(bounds, dict) else None

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
            value = float(np.clip(value, bounds[0], bounds[1]))

        row[feature] = value

    return pd.DataFrame([row], columns=features)


def saved_prediction(bundle, model_name, input_df):
    """Predict with either a pipeline or estimator + separate preprocessor."""
    model = bundle["models"][model_name]

    if hasattr(model, "named_steps"):
        prediction = model.predict(input_df)[0]
        probabilities = (
            model.predict_proba(input_df)[0]
            if hasattr(model, "predict_proba")
            else None
        )
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            final_step = list(model.named_steps.values())[-1]
            classes = getattr(final_step, "classes_", None)
        return prediction, probabilities, classes

    preprocessor = bundle.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Saved preprocessor is missing from the bundle.")

    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)[0]
    probabilities = (
        model.predict_proba(transformed)[0]
        if hasattr(model, "predict_proba")
        else None
    )
    classes = getattr(model, "classes_", None)
    return prediction, probabilities, classes


def model_dict(bundle):
    if not isinstance(bundle, dict):
        return {}
    models = bundle.get("models", {})
    return models if isinstance(models, dict) else {}


def render_inputs(bundle, prefix, numeric_defaults, categorical_names):
    """Render common manual fields that exist in the bundle's feature list."""
    features = bundle.get("features", [])
    values = {}
    left, right = st.columns(2)

    for i, (feature, (default, step)) in enumerate(numeric_defaults.items()):
        if feature not in features:
            continue

        if feature in df.columns:
            try:
                default = float(get_default(feature))
            except (TypeError, ValueError):
                pass

        with (left if i % 2 == 0 else right):
            values[feature] = st.number_input(
                feature.replace("_", " "),
                min_value=0.0,
                value=float(default),
                step=float(step),
                key=f"{prefix}_{feature}",
            )

    for i, feature in enumerate(categorical_names):
        if feature not in features:
            continue

        options = (
            df[feature].dropna().unique().tolist()
            if feature in df.columns
            else []
        )
        with (left if i % 2 == 0 else right):
            if options:
                default_value = get_default(feature)
                try:
                    default_index = options.index(default_value)
                except ValueError:
                    default_index = 0
                values[feature] = st.selectbox(
                    feature.replace("_", " "),
                    options,
                    index=default_index,
                    key=f"{prefix}_{feature}",
                )

    return values


# ============================================================
# SIDEBAR
# ============================================================
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Regression", "Classification", "Clustering"],
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Dataset Statistics"):
        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True,
        )

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.subheader("Missing Values")
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing.rename("Missing Count").to_frame())


# ============================================================
# REGRESSION
# ============================================================
elif page == "Regression":
    st.header("Sales Regression")
    st.write("Choose a saved regression model and enter transaction details.")

    if not reg_bundle:
        st.error("Regression bundle could not be loaded. Check the logs and file path.")
        st.stop()

    reg_models = model_dict(reg_bundle)
    reg_features = reg_bundle.get("features", [])

    if not reg_models or not reg_features:
        st.error("Regression bundle must contain 'models' and 'features'.")
        st.stop()

    selected_model = st.selectbox("Regression Algorithm", list(reg_models.keys()))
    numeric_defaults = {
        "Quantity": (1.0, 1.0),
        "Unit_Price": (100.0, 10.0),
        "Discount_Percentage": (0.0, 1.0),
        "Customer_Age": (30.0, 1.0),
        "Delivery_Days": (3.0, 1.0),
        "Customer_Rating": (4.0, 0.5),
    }
    categorical_fields = [
        "Product_Category", "Customer_Segment", "Sales_Channel", "Payment_Method"
    ]

    with st.form("regression_form"):
        values = render_inputs(
            reg_bundle, "reg", numeric_defaults, categorical_fields
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            quantity = values.get("Quantity", get_default("Quantity"))
            unit_price = values.get("Unit_Price", get_default("Unit_Price"))
            discount = values.get("Discount_Percentage", get_default("Discount_Percentage"))

            if "Gross_Amount_Before_Discount" in reg_features:
                values["Gross_Amount_Before_Discount"] = quantity * unit_price
            if "Discount_Amount" in reg_features:
                values["Discount_Amount"] = quantity * unit_price * discount / 100.0

            input_df = prepare_features(reg_bundle, values)
            prediction, _, _ = saved_prediction(reg_bundle, selected_model, input_df)

            st.subheader("Prediction Result")
            st.metric("Predicted Value", f"{float(prediction):,.2f}")
            st.caption(
                "Features not entered manually use dataset defaults. "
                "The prediction is an estimate from the saved model."
            )
        except Exception as exc:
            st.error(f"Regression prediction failed: {type(exc).__name__}: {exc}")


# ============================================================
# CLASSIFICATION
# ============================================================
elif page == "Classification":
    st.header("Return Status Classification")
    st.write("Choose one of the saved classifiers to predict its output class.")

    if not cls_bundle:
        st.error("Classification bundle could not be loaded. Check the logs and file path.")
        st.stop()

    cls_models = model_dict(cls_bundle)
    cls_features = cls_bundle.get("features", [])

    if not cls_models or not cls_features:
        st.error("Classification bundle must contain 'models' and 'features'.")
        st.stop()

    st.info(
        f"{len(cls_models)} saved classifier(s) found. "
        "Only models actually present in the bundle are selectable."
    )
    selected_model = st.selectbox("Classification Algorithm", list(cls_models.keys()))

    numeric_defaults = {
        "Quantity": (1.0, 1.0),
        "Unit_Price": (100.0, 10.0),
        "Discount_Percentage": (0.0, 1.0),
        "Customer_Age": (30.0, 1.0),
        "Delivery_Days": (3.0, 1.0),
        "Customer_Rating": (4.0, 0.5),
    }
    categorical_fields = [
        "Product_Category", "Customer_Segment", "Sales_Channel", "Payment_Method"
    ]

    with st.form("classification_form"):
        values = render_inputs(
            cls_bundle, "cls", numeric_defaults, categorical_fields
        )
        submitted = st.form_submit_button("Predict Return Status")

    if submitted:
        try:
            input_df = prepare_features(cls_bundle, values)
            prediction, probabilities, classes = saved_prediction(
                cls_bundle, selected_model, input_df
            )

            st.subheader("Classification Result")
            st.success(f"Predicted class: {prediction}")

            if probabilities is not None and classes is not None:
                probability_df = pd.DataFrame({
                    "Class": list(classes),
                    "Probability (%)": np.asarray(probabilities) * 100.0,
                }).sort_values("Probability (%)", ascending=False)
                st.subheader("Class Probabilities")
                st.dataframe(probability_df, use_container_width=True)

            st.caption(
                "Unentered features use dataset defaults. "
                "Prediction quality depends on the saved model and its training data."
            )
        except Exception as exc:
            st.error(f"Classification prediction failed: {type(exc).__name__}: {exc}")


# ============================================================
# CLUSTERING
# ============================================================
elif page == "Clustering":
    st.header("Sales Clustering")
    st.write("Explore saved clusters and assign an input to a cluster when supported.")

    if not cluster_bundle:
        st.error("Clustering bundle could not be loaded. Check the logs and file path.")
        st.stop()

    cluster_models = model_dict(cluster_bundle)
    cluster_features = cluster_bundle.get("features", [])
    cluster_preprocessor = cluster_bundle.get("preprocessor")

    if cluster_models:
        selected_cluster_model_name = st.selectbox(
            "Clustering Algorithm", list(cluster_models.keys())
        )
        cluster_model = cluster_models[selected_cluster_model_name]
    else:
        selected_cluster_model_name = None
        cluster_model = cluster_bundle.get(
            "model",
            cluster_bundle.get("clustering_model"),
        )

    if cluster_model is None:
        st.error(
            "No clustering estimator found. Expected a 'models' dictionary, "
            "'model', or 'clustering_model' entry in the bundle."
        )
        st.write("Bundle keys:", list(cluster_bundle.keys()))
        st.stop()

    st.write(
        "Selected model: "
        + str(selected_cluster_model_name or "Saved clustering model")
    )

    labels = getattr(cluster_model, "labels_", None)
    if labels is not None:
        counts = pd.Series(labels, name="Cluster").value_counts().sort_index()
        st.subheader("Cluster Distribution")
        st.bar_chart(counts)
        st.dataframe(
            counts.rename("Number of Records").to_frame(),
            use_container_width=True,
        )
    else:
        st.info(
            "This model does not expose stored labels. Use the form below "
            "to assign a new record if the model supports predict()."
        )

    if cluster_features and cluster_preprocessor is not None:
        if hasattr(cluster_model, "predict"):
            st.subheader("Assign a Transaction to a Cluster")

            numeric_defaults = {
                "Quantity": (1.0, 1.0),
                "Unit_Price": (100.0, 10.0),
                "Discount_Percentage": (0.0, 1.0),
                "Customer_Age": (30.0, 1.0),
                "Customer_Rating": (4.0, 0.5),
                "Delivery_Days": (3.0, 1.0),
            }
            categorical_fields = [
                "Product_Category", "Customer_Segment", "Sales_Channel", "Payment_Method"
            ]

            with st.form("clustering_form"):
                values = render_inputs(
                    cluster_bundle, "cluster", numeric_defaults, categorical_fields
                )
                submitted = st.form_submit_button("Find Cluster")

            if submitted:
                try:
                    input_df = prepare_features(cluster_bundle, values)
                    if hasattr(cluster_model, "named_steps"):
                        cluster_label = cluster_model.predict(input_df)[0]
                    else:
                        transformed = cluster_preprocessor.transform(input_df)
                        cluster_label = cluster_model.predict(transformed)[0]

                    st.success(f"Assigned cluster: {cluster_label}")
                except Exception as exc:
                    st.error(
                        f"Clustering prediction failed: {type(exc).__name__}: {exc}"
                    )
        else:
            st.info(
                "This clustering algorithm does not support predict() "
                "(for example, DBSCAN). Cluster distribution is shown if labels exist."
            )
    else:
        st.warning(
            "The bundle does not contain both 'features' and 'preprocessor'. "
            "Only the information available in the saved model can be displayed."
        )


# ============================================================
# FOOTER
# ============================================================
st.sidebar.divider()
st.sidebar.caption("Retail Sales ML Dashboard")
