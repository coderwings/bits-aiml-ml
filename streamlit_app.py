import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Classification Model performance review - Early Stage Diabetes Risk Prediction")

# a. Dataset upload option
uploaded_file = st.file_uploader("Upload your test CSV data", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # b. Model selection dropdown
    model_option = st.selectbox(
        'Which model would you like to use?',
        ('Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost')
    )

    # Load pre-trained models
    model = joblib.load(f'model/{model_option.lower().replace(' ', '_')}}.pkl')
    
    st.write(f"Results for {model_option}")
    
    # c. Display evaluation metrics
    st.write(classification_report(y_test, y_preds))

    # d. Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, ax=ax)
    st.pyplot(fig)