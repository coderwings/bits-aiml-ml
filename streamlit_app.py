import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Performance Review", layout="wide")
st.title("ML Classification Model Performance - Early Stage Diabetes Risk Prediction")

st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV data", type="csv")

st.sidebar.header("Download Test Data")
try:
    with open("diabetes_data_upload.csv", "rb") as file:
        st.sidebar.download_button(
            label="Download Sample Test CSV",
            data=file,
            file_name="diabetes_data_upload.csv",
            mime="text/csv",
            help="Download this file to use as a template for testing the models."
        )
except FileNotFoundError:
    st.sidebar.warning("Sample CSV file not found in repository root.")

if uploaded_file:
    # Load Data
    data = pd.read_csv(uploaded_file)
    st.write("### Test Data Preview")
    st.dataframe(data.head())

    # --- b. Model selection dropdown ---
    st.sidebar.header("Model Selection")
    model_option = st.sidebar.selectbox(
        'Which model would you like to use?',
        ('Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost')
    )
    
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    

    # 1. Identify Features and Target
    # NOTE: Change 'target' to match the actual column name in your diabetes CSV
    if 'class' in data.columns: # Assuming 'class' is the target column in the uploaded CSV
        X_test = data.drop('class', axis=1)
        y_test = data['class']
        if y_test.dtype == 'object':
            mapping = {'positive': 1, 'negative': 0, 'yes': 1, 'no': 0} # Lowercased keys to match lowercased column values
            y_test = y_test.map(lambda x: mapping.get(x.lower(), x)) # Map lowercased values
            # Convert to numeric to handle cases where mapping might leave strings or mixed types
            y_test = pd.to_numeric(y_test, errors='coerce')
    elif 'target' in data.columns:
        X_test = data.drop('target', axis=1)
        y_test = data['target']
        if y_test.dtype == 'object':
            mapping = {'positive': 1, 'negative': 0, 'yes': 1, 'no': 0} # Lowercased keys
            y_test = y_test.map(lambda x: mapping.get(x.lower(), x)) # Map lowercased values
            # Convert to numeric to handle cases where mapping might leave strings or mixed types
            y_test = pd.to_numeric(y_test, errors='coerce')
    else:
        print("The CSV must contain either a 'class' or 'target' column. Please check your column names.")
        exit() # Halts execution like st.stop()

    try:
        # 2. Load pre-trained model and scaler
        model_filename = f"model/{model_option.lower().replace(' ', '_')}.pkl"
        model = joblib.load(model_filename)
        scaler = joblib.load('model/scaler.pkl')
        
        # Get the feature names the scaler/model was trained on
        expected_features = scaler.feature_names_in_

        # Apply one-hot encoding to X_test (from diabetes_data_upload.csv)
        # Ensure drop_first=True to match training data preprocessing in IyHo2v3ntsRR
        X_test_processed_dummies = pd.get_dummies(X_test, drop_first=True)

        # Reinitialize X_test_aligned to ensure it starts fresh for current X_test
        X_test_aligned = pd.DataFrame(0, index=X_test_processed_dummies.index, columns=expected_features)

        # Populate X_test_aligned with columns that exist in both
        for col in expected_features:
            if col in X_test_processed_dummies.columns:
                X_test_aligned[col] = X_test_processed_dummies[col]

        X_test_aligned = X_test_aligned[expected_features]


        # 3. Preprocess data (scaling)
        # KNN and Logistic Regression need scaled data
        if model_option in ['Logistic Regression', 'KNN']:
            X_test_final = scaler.transform(X_test_aligned)
        else:
            X_test_final = X_test_aligned

        # 4. Generate Predictions
        y_pred = model.predict(X_test_final)
        y_prob = model.predict_proba(X_test_final)[:, 1] if hasattr(model, "predict_proba") else y_pred

        st.divider()
        st.subheader(f"Results for {model_option}")

        # --- c. Display evaluation metrics (Mandatory 6 metrics) ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        # --- d. Confusion matrix and Classification Report ---
        st.subheader("Visual Analysis")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        with viz_col2:
            st.text("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

    except FileNotFoundError:
        st.error(f"Could not find model file: {model_filename}. Ensure all .pkl files are in the 'model/' folder on GitHub.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Waiting for test CSV upload...")