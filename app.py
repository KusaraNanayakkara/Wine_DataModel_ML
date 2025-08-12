# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load dataset
df = pd.read_csv('data/winequality-red.csv')

# Sidebar navigation
st.sidebar.title("Wine Quality Prediction App")
page = st.sidebar.radio("Go to", ["Data Exploration", "Visualisation", "Prediction", "Model Info"])

# Data Exploration Page
if page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    if st.checkbox("Show Summary Stats"):
        st.write(df.describe())

# Visualisation Page
elif page == "Visualisation":
    st.title("📈 Visualisations")
    fig, ax = plt.subplots()
    sns.countplot(x='quality', data=df, ax=ax)
    st.pyplot(fig)

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Prediction Page
elif page == "Prediction":
    st.title("🍷 Wine Quality Prediction")
    st.write("Enter wine features:")

    features = []
    for col in df.columns[:-1]:
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        features.append(val)

    if st.button("Predict"):
        arr = np.array(features).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]
        prob = model.predict_proba(arr_scaled)[0][pred]
        st.write("Prediction:", "Good Quality" if pred == 1 else "Bad Quality")
        st.write("Confidence:", f"{prob*100:.2f}%")

# Model Info Page
elif page == "Model Info":
    st.title("ℹ️ Model Information")
    st.write("Trained on Wine Quality dataset from Kaggle.")
    st.write("Best model selected based on accuracy.")
