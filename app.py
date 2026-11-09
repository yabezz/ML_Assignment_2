import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing ML Classifier",
    page_icon="ML",
    layout="wide",
)

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# ---------------------------------------------------------
# MODEL FILES
# ---------------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# ---------------------------------------------------------
# DATASET FEATURES
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

TARGET_COLUMN = "y"

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------


@st.cache_resource
def load_model(model_path):
    """Load a previously trained model from disk."""
    return joblib.load(model_path)


def get_model_score(model, X):
    """
    Return a continuous score for AUC.

    If the model supports probability prediction, use
    probability of the positive class.

    Otherwise, use decision_function.
    """
    try:
        probabilities = model.predict_proba(X)
        return probabilities[:, 1]
    except Exception:
        try:
            return model.decision_function(X)
        except Exception:
            return None


def calculate_metrics(model, X, y_true):
    """Generate predictions and all required assignment metrics."""

    predictions = model.predict(X)

    score = get_model_score(model, X)

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true,
        predictions,
    )

    if score is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(
            y_true,
            score,
        )
    else:
        auc = np.nan

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
    }, predictions


def validate_uploaded_data(df):
    """Check whether the uploaded CSV has the required columns."""

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        return (
            False,
            "The uploaded CSV is missing these required feature columns: "
            + ", ".join(missing_features),
        )

    if TARGET_COLUMN not in df.columns:
        return (
            False,
            f"The uploaded CSV must contain the target column '{TARGET_COLUMN}'.",
        )

    return True, ""


def prepare_target(series):
    """
    Convert Bank Marketing target values into binary values.

    no  -> 0
    yes -> 1
    """

    target = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "no": 0,
        "yes": 1,
        "0": 0,
        "1": 1,
    }

    converted = target.map(mapping)

    return converted


def display_metric_cards(metrics):
    """Display the six required metrics."""

    columns = st.columns(6)

    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]

    for column, metric_name in zip(columns, metric_names):
        value = metrics[metric_name]

        if pd.isna(value):
            display_value = "N/A"
        else:
            display_value = f"{value:.4f}"

        column.metric(
            label=metric_name,
            value=display_value,
        )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("Bank Marketing Classification")

st.markdown(
    """
This application evaluates five machine learning classification models
using an uploaded test CSV.

**Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, and SVM

**Metrics:** Accuracy, AUC, Precision, Recall, F1 Score, and MCC
"""
)

st.divider()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("Model Selection")

    selected_model = st.selectbox(
        "Choose a trained model",
        list(MODEL_FILES.keys()),
    )

    st.divider()

    st.subheader("Required Features")

    st.write(
        """
        - age
        - job
        - marital
        - education
        - default
        - balance
        - housing
        - loan
        - contact
        - day
        - month
        - duration
        - campaign
        - pdays
        - previous
        - poutcome
        """
    )

    st.divider()

    st.caption(
        "Models are loaded from the local model/ directory. "
        "No model training is performed by this application."
    )

# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

st.header("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"],
    help="The CSV must contain all 16 input features and the target column y.",
)

# ---------------------------------------------------------
# NO FILE UPLOADED
# ---------------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload test_data.csv to begin model evaluation."
    )

    st.subheader("Expected CSV Structure")

    example_data = pd.DataFrame(
        {
            "age": [30],
            "job": ["technician"],
            "marital": ["married"],
            "education": ["secondary"],
            "default": ["no"],
            "balance": [1000],
            "housing": ["yes"],
            "loan": ["no"],
            "contact": ["cellular"],
            "day": [15],
            "month": ["may"],
            "duration": [200],
            "campaign": [1],
            "pdays": [-1],
            "previous": [0],
            "poutcome": ["unknown"],
            "y": ["no"],
        }
    )

    st.dataframe(
        example_data,
        use_container_width=True,
    )

    st.stop()

# ---------------------------------------------------------
# READ CSV
# ---------------------------------------------------------

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the uploaded CSV file: {error}")
    st.stop()

# ---------------------------------------------------------
# DATA VALIDATION
# ---------------------------------------------------------

is_valid, validation_message = validate_uploaded_data(df)

if not is_valid:
    st.error(validation_message)
    st.stop()

# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------

X = df[FEATURE_COLUMNS].copy()

y = prepare_target(df[TARGET_COLUMN])

if y.isna().any():
    invalid_values = (
        df.loc[y.isna(), TARGET_COLUMN]
        .astype(str)
        .unique()
        .tolist()
    )

    st.error(
        "Invalid target values found in column 'y'. "
        f"Expected 'yes' or 'no'. Invalid values: {invalid_values}"
    )

    st.stop()

y = y.astype(int)

# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------

st.header("2. Uploaded Test Dataset")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

info_col1.metric(
    "Rows",
    len(df),
)

info_col2.metric(
    "Input Features",
    len(FEATURE_COLUMNS),
)

info_col3.metric(
    "Target Classes",
    y.nunique(),
)

info_col4.metric(
    "Selected Model",
    selected_model,
)

with st.expander("Preview Uploaded Data"):

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )

# ---------------------------------------------------------
# LOAD SELECTED MODEL
# ---------------------------------------------------------

model_filename = MODEL_FILES[selected_model]
model_path = MODEL_DIR / model_filename

if not model_path.exists():

    st.error(
        f"Model file not found:\n\n{model_path}"
    )

    st.info(
        "Make sure the model training script has been executed "
        "and that all five .joblib files are inside the model folder."
    )

    st.stop()

try:
    model = load_model(model_path)
except Exception as error:
    st.error(
        f"Unable to load the selected model: {error}"
    )
    st.stop()

# ---------------------------------------------------------
# EVALUATE SELECTED MODEL
# ---------------------------------------------------------

st.header("3. Model Evaluation")

try:
    metrics, predictions = calculate_metrics(
        model,
        X,
        y,
    )
except Exception as error:
    st.error(
        f"Model evaluation failed: {error}"
    )
    st.stop()

# ---------------------------------------------------------
# REQUIRED METRICS
# ---------------------------------------------------------

st.subheader(f"Performance of {selected_model}")

display_metric_cards(metrics)

# ---------------------------------------------------------
# CONFUSION MATRIX + CLASSIFICATION REPORT
# ---------------------------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y,
    predictions,
)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

image = ax.imshow(cm)

ax.set_title(
    f"Confusion Matrix - {selected_model}"
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["No", "Yes"])
ax.set_yticklabels(["No", "Yes"])

# Add numbers inside matrix
for row in range(cm.shape[0]):
    for column in range(cm.shape[1]):
        ax.text(
            column,
            row,
            cm[row, column],
            ha="center",
            va="center",
        )

fig.colorbar(image, ax=ax)

st.pyplot(
    fig,
    use_container_width=False,
)

plt.close(fig)

# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y,
    predictions,
    target_names=["No", "Yes"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True,
)

# ---------------------------------------------------------
# PREDICTION DISTRIBUTION
# ---------------------------------------------------------

st.subheader("Prediction Distribution")

prediction_counts = pd.Series(
    predictions
).map(
    {
        0: "No",
        1: "Yes",
    }
).value_counts()

st.bar_chart(
    prediction_counts
)

# ---------------------------------------------------------
# ALL FIVE MODEL COMPARISON
# ---------------------------------------------------------

st.header("4. Comparison of All Five Models")

st.write(
    "The table below evaluates all saved models on the same uploaded test data."
)

all_results = []

for model_name, filename in MODEL_FILES.items():

    current_path = MODEL_DIR / filename

    if not current_path.exists():
        st.warning(
            f"Model file missing: {filename}"
        )
        continue

    try:

        current_model = load_model(current_path)

        current_metrics, _ = calculate_metrics(
            current_model,
            X,
            y,
        )

        row = {
            "ML Model": model_name,
            "Accuracy": current_metrics["Accuracy"],
            "AUC": current_metrics["AUC"],
            "Precision": current_metrics["Precision"],
            "Recall": current_metrics["Recall"],
            "F1": current_metrics["F1"],
            "MCC": current_metrics["MCC"],
        }

        all_results.append(row)

    except Exception as error:

        st.warning(
            f"Could not evaluate {model_name}: {error}"
        )

# ---------------------------------------------------------
# DISPLAY COMPARISON
# ---------------------------------------------------------

if all_results:

    comparison_df = pd.DataFrame(
        all_results
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    # -----------------------------------------------------
    # BEST MODEL BY F1
    # -----------------------------------------------------

    best_f1_row = comparison_df.loc[
        comparison_df["F1"].idxmax()
    ]

    st.success(
        f"Best model by F1 Score: "
        f"{best_f1_row['ML Model']} "
        f"({best_f1_row['F1']:.4f})"
    )

    # -----------------------------------------------------
    # BEST MODEL BY ACCURACY
    # -----------------------------------------------------

    best_accuracy_row = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    st.info(
        f"Best model by Accuracy: "
        f"{best_accuracy_row['ML Model']} "
        f"({best_accuracy_row['Accuracy']:.4f})"
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Assignment-2 | Bank Marketing Classification"
)
