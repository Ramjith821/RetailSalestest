
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

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
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Sales ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Machine Learning Dashboard")
st.caption("Regression • Classification • Clustering")

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

CLUSTER_PATH = os.path.join(
    BASE_DIR, "models", "clustering_bundle.joblib"
)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_bundle(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

reg_bundle = load_bundle(REG_PATH)
cls_bundle = load_bundle(CLS_PATH)
cluster_bundle = load_bundle(CLUSTER_PATH)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Regression",
        "Classification",
        "Clustering"
    ]
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_default(column):
    """Return a reasonable dataset default for a feature."""

    if column not in df.columns:
        return 0

    series = df[column].dropna()

    if series.empty:
        return 0

    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())

    return series.mode().iloc[0]


def get_bounds(bundle, column):
    bounds = bundle.get("iqr_bounds", {})

    if column not in bounds:
        return None

    value = bounds[column]

    if isinstance(value, dict):
        low = value.get("lower", value.get("lower_bound"))
        high = value.get("upper", value.get("upper_bound"))
        if low is not None and high is not None:
            return low, high

    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1]

    return None


def prepare_input(features, values):
    """Build one row in exactly the expected feature order."""

    row = {}

    for feature in features:
        row[feature] = values.get(feature, get_default(feature))

    return pd.DataFrame([row], columns=features)


def get_models(bundle):
    if not bundle:
        return {}

    models = bundle.get("models", {})
    return models if isinstance(models, dict) else {}


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Features", len(df.columns))
    c3.metric("Missing Values", f"{df.isna().sum().sum():,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Statistics")
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.subheader("Missing Values by Column")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing):
        st.bar_chart(missing)
    else:
        st.success("No missing values found.")

    st.subheader("Numerical Feature Correlations")

    numeric_df = df.select_dtypes(include=np.number)

    if len(numeric_df.columns) >= 2:
        st.dataframe(
            numeric_df.corr().round(2),
            use_container_width=True
        )
    else:
        st.info("Not enough numerical columns to calculate correlation.")


# --------------------------------------------------
# REGRESSION
# --------------------------------------------------

elif page == "Regression":

    st.header("📈 Sales Regression")
    st.write("Predict the target used by your saved regression models.")

    if not reg_bundle:
        st.error("Regression bundle not found. Check the models folder.")
        st.stop()

    reg_models = get_models(reg_bundle)
    reg_preprocessor = reg_bundle.get("preprocessor")
    reg_features = reg_bundle.get("features", [])

    if not reg_models or reg_preprocessor is None or not reg_features:
        st.error("The regression bundle is missing required components.")
        st.stop()

    model_name = st.selectbox(
        "Select Regression Algorithm",
        list(reg_models.keys())
    )

    values = {}

    numeric_inputs = {
        "Quantity": (1.0, 1.0),
        "Unit_Price": (100.0, 10.0),
        "Discount_Percentage": (0.0, 1.0),
        "Customer_Age": (30.0, 1.0),
        "Delivery_Days": (3.0, 1.0),
        "Customer_Rating": (4.0, 0.5)
    }

    categorical_inputs = [
        "Product_Category",
        "Customer_Segment",
        "Sales_Channel",
        "Payment_Method"
    ]

    with st.form("regression_form"):

        st.subheader("Transaction Details")
        left, right = st.columns(2)

        for i, (feature, (default, step)) in enumerate(
            numeric_inputs.items()
        ):
            if feature in reg_features:
                with (left if i % 2 == 0 else right):
                    values[feature] = st.number_input(
                        feature.replace("_", " "),
                        min_value=0.0,
                        value=float(default),
                        step=float(step)
                    )

        for i, feature in enumerate(categorical_inputs):
            if feature in reg_features:
                with (left if i % 2 == 0 else right):
                    if feature in df.columns:
                        options = df[feature].dropna().unique().tolist()
                    else:
                        options = []

                    if options:
                        default = get_default(feature)
                        index = (
                            options.index(default)
                            if default in options else 0
                        )

                        values[feature] = st.selectbox(
                            feature.replace("_", " "),
                            options,
                            index=index
                        )
                    else:
                        values[feature] = st.text_input(
                            feature.replace("_", " "),
                            value=""
                        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Fill other expected features using dataset defaults.
            for feature in reg_features:
                if feature not in values:
                    values[feature] = get_default(feature)

            # Derive common transaction features when available.
            quantity = values.get("Quantity", get_default("Quantity"))
            price = values.get("Unit_Price", get_default("Unit_Price"))
            discount = values.get(
                "Discount_Percentage",
                get_default("Discount_Percentage")
            )

            gross = quantity * price
            discount_amount = gross * discount / 100

            if "Gross_Amount_Before_Discount" in reg_features:
                values["Gross_Amount_Before_Discount"] = gross

            if "Discount_Amount" in reg_features:
                values["Discount_Amount"] = discount_amount

            # Apply saved training bounds where available.
            for feature in reg_features:
                bound = get_bounds(reg_bundle, feature)

                if bound and isinstance(values[feature], (int, float)):
                    values[feature] = np.clip(
                        values[feature], bound[0], bound[1]
                    )

            X = prepare_input(reg_features, values)
            X_transformed = reg_preprocessor.transform(X)

            model = reg_models[model_name]
            prediction = model.predict(X_transformed)[0]

            st.subheader("Prediction Result")
            st.metric("Predicted Value", f"{prediction:,.2f}")

            st.caption(
                "This is a model estimate. Features not entered manually "
                "were filled using dataset defaults."
            )

        except Exception as e:
            st.error(f"Regression prediction failed: {e}")


# --------------------------------------------------
# CLASSIFICATION - 10 ALGORITHMS
# --------------------------------------------------

elif page == "Classification":

    st.header("🧠 Return Status Classification")
    st.write(
        "Compare 10 classification algorithms and predict the return "
        "status of a transaction."
    )

    # Find the target column.
    target_candidates = [
        "Return_Flag",
        "Return_Status",
        "Returned",
        "Return"
    ]

    target = next(
        (col for col in target_candidates if col in df.columns),
        None
    )

    if target is None:
        st.error(
            "Could not find the return target column. "
            "Check the exact target name in your CSV."
        )
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    st.info(f"Classification target: {target}")

    # Features that could reveal the outcome or be derived from it.
    leakage_candidates = {
        target,
        "Return_Reason",
        "Profit",
        "Sales_Amount",
        "Cost_Amount",
        "Order_Status"
    }

    feature_cols = [
        col for col in df.columns
        if col not in leakage_candidates
    ]

    data = df[feature_cols + [target]].dropna(subset=[target]).copy()

    # Ensure the target has at least two classes.
    if data[target].nunique() < 2:
        st.error("The target column must contain at least two classes.")
        st.stop()

    # Drop rows with extremely rare target classes only if needed
    # for a stratified split.
    X = data[feature_cols]
    y = data[target]

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        st.warning(
            "Some target classes have fewer than two records. "
            "The model comparison may be unreliable."
        )

    # Fit a preprocessing pipeline from the current training data.
    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(transformers)

    # Ten classification algorithms.
    classifier_factories = {
        "1. Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "2. KNN Classifier": lambda: KNeighborsClassifier(
            n_neighbors=5
        ),
        "3. Gaussian Naive Bayes": lambda: GaussianNB(),
        "4. Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=6, random_state=42
        ),
        "5. Support Vector Machine": lambda: SVC(
            probability=True, random_state=42
        ),
        "6. Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "7. Gradient Boosting": lambda: GradientBoostingClassifier(
            random_state=42
        ),
        "8. Extra Trees": lambda: ExtraTreesClassifier(
            n_estimators=100, random_state=42
        ),
        "9. AdaBoost": lambda: AdaBoostClassifier(
            random_state=42
        ),
        "10. Hist Gradient Boosting": lambda: HistGradientBoostingClassifier(
            random_state=42
        )
    }

    selected_classifier = st.selectbox(
        "Select Classification Algorithm",
        list(classifier_factories.keys())
    )

    @st.cache_resource
    def train_classifier_models(data_frame, target_name, feature_names):
        # Convert a stable copy of the data into training inputs.
        X_all = data_frame[feature_names].copy()
        y_all = data_frame[target_name].copy()

        num_cols = X_all.select_dtypes(
            include=np.number
        ).columns.tolist()

        cat_cols = [
            col for col in X_all.columns
            if col not in num_cols
        ]

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        parts = []

        if num_cols:
            parts.append(("num", num_pipe, num_cols))

        if cat_cols:
            parts.append(("cat", cat_pipe, cat_cols))

        prep = ColumnTransformer(parts)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=42,
            stratify=y_all
        )

        trained = {}
        scores = {}

        for name, factory in classifier_factories.items():
            try:
                # GaussianNB and HistGradientBoosting need dense input.
                model = factory()

                pipe = Pipeline([
                    ("preprocessor", prep),
                    ("classifier", model)
                ])

                pipe.fit(X_train, y_train)
                predictions = pipe.predict(X_test)

                trained[name] = pipe
                scores[name] = accuracy_score(y_test, predictions)

            except Exception as exc:
                scores[name] = None

        return trained, scores

    # Avoid crashes on classes too small for stratification.
    try:
        trained_models, model_scores = train_classifier_models(
            data,
            target,
            feature_cols
        )
    except Exception as e:
        st.error(
            "Could not train the classification models. "
            f"Check target class counts and dataset values. Details: {e}"
        )
        st.stop()

    valid_scores = {
        name: score for name, score in model_scores.items()
        if score is not None
    }

    if valid_scores:
        st.subheader("Algorithm Comparison")
        comparison = pd.DataFrame({
            "Algorithm": list(valid_scores.keys()),
            "Accuracy": list(valid_scores.values())
        }).sort_values("Accuracy", ascending=False)

        st.dataframe(comparison, use_container_width=True)
        st.bar_chart(comparison.set_index("Algorithm"))

    if selected_classifier not in trained_models:
        st.warning(
            "This algorithm could not be trained on the current dataset. "
            "Try another algorithm or check the data."
        )
        st.stop()

    st.subheader("Predict a Transaction")

    values = {}

    with st.form("classification_form"):
        left, right = st.columns(2)

        manual_numeric = {
            "Quantity": 1.0,
            "Unit_Price": 100.0,
            "Discount_Percentage": 0.0,
            "Customer_Age": 30.0,
            "Delivery_Days": 3.0,
            "Customer_Rating": 4.0
        }

        for i, (feature, default) in enumerate(manual_numeric.items()):
            if feature in feature_cols and feature in numeric_features:
                with (left if i % 2 == 0 else right):
                    values[feature] = st.number_input(
                        feature.replace("_", " "),
                        min_value=0.0,
                        value=float(default),
                        key=f"cls_{feature}"
                    )

        categorical_inputs = [
            "Product_Category",
            "Customer_Segment",
            "Sales_Channel",
            "Payment_Method"
        ]

        for i, feature in enumerate(categorical_inputs):
            if feature in feature_cols and feature in categorical_features:
                with (left if i % 2 == 0 else right):
                    options = df[feature].dropna().unique().tolist()

                    if options:
                        default = get_default(feature)
                        index = (
                            options.index(default)
                            if default in options else 0
                        )

                        values[feature] = st.selectbox(
                            feature.replace("_", " "),
                            options,
                            index=index,
                            key=f"cls_cat_{feature}"
                        )

        submitted = st.form_submit_button("Predict Return Status")

    if submitted:
        try:
            for feature in feature_cols:
                if feature not in values:
                    values[feature] = get_default(feature)

            X_input = pd.DataFrame(
                [{feature: values[feature] for feature in feature_cols}],
                columns=feature_cols
            )

            model = trained_models[selected_classifier]
            prediction = model.predict(X_input)[0]

            st.subheader("Prediction Result")
            st.success(f"Predicted {target}: {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_input)[0]
                classes = model.classes_

                st.write("Class probabilities")

                probability_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=False)

                st.dataframe(
                    probability_df,
                    use_container_width=True
                )

            st.caption(
                "The classifier is trained on the dataset at app startup. "
                "Accuracy is a held-out test score, not a guarantee for "
                "individual transactions."
            )

        except Exception as e:
            st.error(f"Classification prediction failed: {e}")


# --------------------------------------------------
# CLUSTERING
# --------------------------------------------------

elif page == "Clustering":

    st.header("🔍 Customer and Sales Clustering")
    st.write("Explore the saved clustering model and its results.")

    if not cluster_bundle:
        st.error("Clustering bundle not found. Check the models folder.")
        st.stop()

    cluster_models = get_models(cluster_bundle)

    st.subheader("Saved Clustering Model")

    if cluster_models:
        cluster_model_name = st.selectbox(
            "Select Clustering Algorithm",
            list(cluster_models.keys())
        )
        cluster_model = cluster_models[cluster_model_name]
    else:
        cluster_model_name = None
        cluster_model = cluster_bundle.get(
            "model",
            cluster_bundle.get("clustering_model")
        )

    cluster_features = cluster_bundle.get("features", [])
    cluster_preprocessor = cluster_bundle.get("preprocessor")

    if cluster_model is None:
        st.warning(
            "No model was found under 'models', 'model', or "
            "'clustering_model' in the saved bundle."
        )
        st.write("Bundle keys:", list(cluster_bundle.keys()))
        st.stop()

    st.write(f"Selected model: **{cluster_model_name or 'Saved model'}**")

    if hasattr(cluster_model, "n_clusters"):
        st.metric("Number of Clusters", cluster_model.n_clusters)

    if hasattr(cluster_model, "labels_"):
        labels = cluster_model.labels_
        counts = pd.Series(labels).value_counts().sort_index()

        st.subheader("Cluster Distribution")
        st.bar_chart(counts)

        st.dataframe(
            counts.rename("Number of Records").to_frame(),
            use_container_width=True
        )

    if cluster_features and cluster_preprocessor is not None:
        st.subheader("Assign a Transaction to a Cluster")

        values = {}

        with st.form("clustering_form"):
            left, right = st.columns(2)

            manual_numeric = {
                "Quantity": 1.0,
                "Unit_Price": 100.0,
                "Discount_Percentage": 0.0,
                "Customer_Age": 30.0,
                "Customer_Rating": 4.0
            }

            for i, (feature, default) in enumerate(manual_numeric.items()):
                if feature in cluster_features:
                    with (left if i % 2 == 0 else right):
                        values[feature] = st.number_input(
                            feature.replace("_", " "),
                            min_value=0.0,
                            value=float(default),
                            key=f"cluster_{feature}"
                        )

            for i, feature in enumerate([
                "Product_Category",
                "Customer_Segment",
                "Sales_Channel"
            ]):
                if feature in cluster_features:
                    with (left if i % 2 == 0 else right):
                        options = (
                            df[feature].dropna().unique().tolist()
                            if feature in df.columns else []
                        )

                        if options:
                            values[feature] = st.selectbox(
                                feature.replace("_", " "),
                                options,
                                key=f"cluster_cat_{feature}"
                            )

            submitted = st.form_submit_button("Find Cluster")

        if submitted:
            try:
                for feature in cluster_features:
                    if feature not in values:
                        values[feature] = get_default(feature)

                X_input = prepare_input(cluster_features, values)
                X_transformed = cluster_preprocessor.transform(X_input)

                cluster_label = cluster_model.predict(X_transformed)[0]

                st.success(f"Assigned Cluster: {cluster_label}")

            except Exception as e:
                st.error(f"Clustering prediction failed: {e}")

    else:
        st.info(
            "The saved clustering bundle does not expose both 'features' "
            "and 'preprocessor'. The cluster distribution above is shown "
            "when the saved model contains labels."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("Retail Sales ML Dashboard")
