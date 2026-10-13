```python
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

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Sales_transactions_2022_2025.csv"
MODEL_DIR = BASE_DIR / "models"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


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
        st.write("Enter transaction details to predict profit.")

        features = reg_bundle["features"]
        models = reg_bundle["models"]
        preprocessor = reg_bundle["preprocessor"]
        bounds = reg_bundle["iqr_bounds"]

        # Build inputs using representative dataset values
        input_values = {}

        with st.form("regression_form"):
            col1, col2 = st.columns(2)

            for i, feature in enumerate(features):
                container = col1 if i % 2 == 0 else col2

                with container:
                    if feature in df.columns:
                        series = df[feature].dropna()

                        if pd.api.types.is_numeric_dtype(df[feature]):
                            default = (
                                float(series.median())
                                if not series.empty else 0.0
                            )
                            input_values[feature] = st.number_input(
                                feature.replace("_", " "),
                                value=default,
                                key=f"reg_{feature}"
                            )
                        else:
                            options = sorted(
                                series.astype(str).unique().tolist()
                            )
                            if options:
                                input_values[feature] = st.selectbox(
                                    feature.replace("_", " "),
                                    options,
                                    key=f"reg_{feature}"
                                )
                            else:
                                input_values[feature] = st.text_input(
                                    feature.replace("_", " "),
                                    key=f"reg_{feature}"
                                )
                    else:
                        input_values[feature] = st.number_input(
                            feature.replace("_", " "),
                            value=0.0,
                            key=f"reg_{feature}"
                        )

            model_name = st.selectbox(
                "Choose Regression Model",
                list(models.keys()),
                index=(
                    list(models.keys()).index("Random Forest Regressor")
                    if "Random Forest Regressor" in models else 0
                )
            )

            submitted = st.form_submit_button("Predict Profit")

        if submitted:
            try:
                input_df = pd.DataFrame(
                    [[input_values[f] for f in features]],
                    columns=features
                )

                # Apply the same outlier clipping used during training
                for feature, limits in bounds.items():
                    if feature in input_df.columns:
                        input_df[feature] = input_df[feature].clip(
                            lower=limits[0],
                            upper=limits[1]
                        )

                transformed = preprocessor.transform(input_df)
                model = models[model_name]
                prediction = model.predict(transformed)[0]

                st.success(f"Predicted Profit: {prediction:,.2f}")
                st.caption(
                    f"Model used: {model_name}. "
                    "This is a model estimate, not a guaranteed outcome."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif page == "Classification":
        st.subheader("Return Status Prediction")
        st.info("Classification prediction form will be added next.")

    elif page == "Clustering":
        st.subheader("Customer Segmentation")
        st.info("Customer clustering form will be added next.")

except Exception as e:
    st.error(f"Could not load the dataset or models: {e}")
    st.info(
        "Check that the CSV and all three model files exist "
        "at the expected repository paths."
    )
```
