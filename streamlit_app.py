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

# Set Page Title
st.set_page_config(page_title="ML Performance Review", layout="wide")
st.title("ML Classification Model Performance - Early Stage Diabetes Risk Prediction")

# --- a. Dataset upload option ---
st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV data", type="csv")

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

    # 1. Identify Features and Target
    # NOTE: Change 'target' to match the actual column name in your diabetes CSV
    if 'target' in data.columns:
        X_test = data.drop('target', axis=1)
        y_test = data['target']
        if y_test.dtype == 'object':
            mapping = {'Positive': 1, 'Negative': 0, 'positive': 1, 'negative': 0, 'Yes': 1, 'No': 0}
            y_test = y_test.map(lambda x: mapping.get(x, x))
            # Convert to numeric to handle cases where mapping might leave strings or mixed types
            y_test = pd.to_numeric(y_test, errors='coerce')
    else:
        st.error("The CSV must contain a 'target' column. Please check your column names.")
        st.stop()

    try:
        # 2. Load pre-trained model and scaler
        model_filename = f"model/{model_option.lower().replace(' ', '_')}.pkl"
        model = joblib.load(model_filename)
        scaler = joblib.load('model/scaler.pkl')
        
        # --- FIX: Ensure feature alignment ---
        # Get the feature names the scaler/model was trained on
        try:
            expected_features = scaler.feature_names_in_
            
            # Check if we need to perform One-Hot Encoding on the uploaded data to match training
            # This is common if the CSV has 'Male'/'Female' but model expects 'gender_Male'
            if not all(col in X_test.columns for col in expected_features):
                st.info("Aligning categorical features...")
                X_test = pd.get_dummies(X_test)
                
            # Add missing columns with zeros (for categories not present in this specific test set)
            for col in expected_features:
                if col not in X_test.columns:
                    X_test[col] = 0
            
            # Reorder columns to exactly match training order
            X_test = X_test[expected_features]
        except AttributeError:
            st.warning("Could not automatically verify feature names. Ensure CSV columns match training data exactly.")

        # 3. Preprocess data
        # KNN and Logistic Regression need scaled data
        if model_option in ['Logistic Regression', 'KNN']:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test

        # 4. Generate Predictions
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else y_pred

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