import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC

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
# CLASSIFICATION - 10 ALGORITHMS
# ============================================================
elif page == "Classification":
    st.header("Return Status Classification")
    st.write("Train and compare 10 classification algorithms on your dataset.")

    if not cls_bundle:
        st.error("Classification bundle could not be loaded. Check the model file and logs.")
        st.stop()

    saved_features = cls_bundle.get("features", [])
    target_candidates = ["Return_Flag", "Return_Status", "Returned", "Return"]
    target = next((col for col in target_candidates if col in df.columns), None)

    if target is None:
        st.error(
            "Could not find the return target column. Expected one of: "
            + ", ".join(target_candidates)
        )
        st.write("Dataset columns:", df.columns.tolist())
        st.stop()

    # Avoid obvious outcome-derived fields to reduce target leakage.
    excluded = {
        target, "Return_Reason", "Profit", "Sales_Amount",
        "Cost_Amount", "Order_Status"
    }

    # Prefer the features recorded in the saved bundle, but remove leakage fields.
    feature_cols = [
        col for col in saved_features
        if col in df.columns and col not in excluded
    ]

    if not feature_cols:
        feature_cols = [col for col in df.columns if col not in excluded]

    data = df[feature_cols + [target]].dropna(subset=[target]).copy()

    if data[target].nunique() < 2:
        st.error(f"The target column '{target}' must contain at least two classes.")
        st.stop()

    class_counts = data[target].value_counts()
    if class_counts.min() < 2:
        st.error(
            "At least one target class has fewer than 2 records. "
            "A stratified train/test split cannot be made reliably."
        )
        st.stop()

    st.info(
        f"Target: {target} | Features used: {len(feature_cols)} | "
        f"Classes: {data[target].nunique()}"
    )

    # Keep a separate test set for model comparison.
    X = data[feature_cols]
    y = data[target]

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # sparse=False supports older scikit-learn releases too.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        st.error("No usable input features were found.")
        st.stop()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError as exc:
        st.error(f"Could not split the classification data: {exc}")
        st.stop()

    classifier_factories = {
        "1. Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "2. K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
        "3. Gaussian Naive Bayes": lambda: GaussianNB(),
        "4. Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=6, random_state=42
        ),
        "5. Support Vector Machine": lambda: SVC(
            probability=True, random_state=42
        ),
        "6. Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "7. Gradient Boosting": lambda: GradientBoostingClassifier(
            random_state=42
        ),
        "8. Extra Trees": lambda: ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "9. AdaBoost": lambda: AdaBoostClassifier(random_state=42),
        "10. Hist Gradient Boosting": lambda: HistGradientBoostingClassifier(
            random_state=42
        ),
    }

    @st.cache_resource
    def train_ten_classifiers(data_frame, target_name, feature_names):
        X_all = data_frame[feature_names].copy()
        y_all = data_frame[target_name].copy()

        nums = X_all.select_dtypes(include=np.number).columns.tolist()
        cats = [col for col in X_all.columns if col not in nums]

        parts = []
        if nums:
            parts.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                nums,
            ))
        if cats:
            parts.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore", sparse=False
                    )),
                ]),
                cats,
            ))

        Xtr, Xte, ytr, yte = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        fitted_models = {}
        score_rows = []

        for name, factory in classifier_factories.items():
            try:
                prep = ColumnTransformer(parts)
                pipeline = Pipeline([
                    ("preprocessor", prep),
                    ("classifier", factory()),
                ])
                pipeline.fit(Xtr, ytr)
                predictions = pipeline.predict(Xte)
                score = accuracy_score(yte, predictions)
                fitted_models[name] = pipeline
                score_rows.append({
                    "Algorithm": name,
                    "Accuracy (%)": round(score * 100, 2),
                    "Status": "Trained",
                })
            except Exception as exc:
                score_rows.append({
                    "Algorithm": name,
                    "Accuracy (%)": np.nan,
                    "Status": f"Failed: {type(exc).__name__}: {exc}",
                })

        return fitted_models, pd.DataFrame(score_rows)

    try:
        trained_models, score_table = train_ten_classifiers(
            data, target, feature_cols
        )
    except Exception as exc:
        st.error(f"Classifier training failed: {type(exc).__name__}: {exc}")
        st.stop()

    st.subheader("Comparison of 10 Classification Algorithms")
    st.dataframe(score_table, use_container_width=True)

    successful_scores = score_table.dropna(subset=["Accuracy (%)"])
    if not successful_scores.empty:
        chart_data = successful_scores.set_index("Algorithm")["Accuracy (%)"]
        st.bar_chart(chart_data)

    if not trained_models:
        st.error("All classifiers failed to train. Check the dataset and logs.")
        st.stop()

    selected_model_name = st.selectbox(
        "Select Classifier for Prediction",
        list(trained_models.keys()),
    )

    st.subheader("Enter Transaction Details")
    values = {}
    numeric_inputs = {
        "Quantity": (1.0, 1.0),
        "Unit_Price": (100.0, 10.0),
        "Discount_Percentage": (0.0, 1.0),
        "Customer_Age": (30.0, 1.0),
        "Delivery_Days": (3.0, 1.0),
        "Customer_Rating": (4.0, 0.5),
    }
    categorical_inputs = [
        "Product_Category", "Customer_Segment",
        "Sales_Channel", "Payment_Method",
    ]

    with st.form("classification_prediction_form"):
        left, right = st.columns(2)

        for i, (feature, (default, step)) in enumerate(numeric_inputs.items()):
            if feature not in feature_cols or feature not in numeric_cols:
                continue
            if feature in df.columns:
                try:
                    default = float(get_default(feature))
                except (ValueError, TypeError):
                    pass
            with (left if i % 2 == 0 else right):
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=0.0,
                    value=float(default),
                    step=float(step),
                    key=f"new_cls_{feature}",
                )

        for i, feature in enumerate(categorical_inputs):
            if feature not in feature_cols or feature not in categorical_cols:
                continue
            options = df[feature].dropna().unique().tolist()
            if not options:
                continue
            default_value = get_default(feature)
            try:
                index = options.index(default_value)
            except ValueError:
                index = 0
            with (left if i % 2 == 0 else right):
                values[feature] = st.selectbox(
                    feature.replace("_", " "),
                    options,
                    index=index,
                    key=f"new_cls_cat_{feature}",
                )

        predict_clicked = st.form_submit_button("Predict Return Status")

    if predict_clicked:
        try:
            input_row = {
                feature: values.get(feature, get_default(feature))
                for feature in feature_cols
            }
            input_df = pd.DataFrame([input_row], columns=feature_cols)
            model = trained_models[selected_model_name]
            prediction = model.predict(input_df)[0]

            st.subheader("Classification Result")
            st.success(f"Predicted {target}: {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.named_steps["classifier"].classes_
                probability_df = pd.DataFrame({
                    "Class": classes,
                    "Probability (%)": probabilities * 100,
                }).sort_values("Probability (%)", ascending=False)
                st.subheader("Class Probabilities")
                st.dataframe(probability_df, use_container_width=True)

            st.caption(
                "Accuracy is measured on a held-out test split. Predictions are "
                "estimates; unentered fields use dataset defaults. Review class "
                "imbalance and recall/precision before relying on this model."
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
