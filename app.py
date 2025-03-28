import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

df = pd.read_csv('non-verbal tourist data.csv')
categorical_cols = label_encoders.keys()
numerical_cols = [col for col in df.columns if col not in categorical_cols and col != 'Type of Client']

st.set_page_config(page_title="Client Type Prediction", page_icon="🔍", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: #e3f2fd;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            color: #0d47a1;
        }
        .sub-title {
            text-align: center;
            font-size: 20px;
            color: #1565c0;
        }
        .stButton>button {
            background-color: #1e88e5;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            width: 100%;
            padding: 10px;
            border: none;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            width: 100%;
            border-radius: 5px;
            border: 1px solid #90caf9;
            padding: 8px;
        }
        .result-box {
            background-color: #43a047;
            color: white;
            padding: 15px;
            text-align: center;
            border-radius: 10px;
            font-size: 20px;
            margin-top: 20px;
            font-weight: bold;
        }
        .form-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>🔍 Client Type Prediction</h1>", unsafe_allow_html=True)
st.markdown("### **Enter Details Below:**")
input_data = {}

with st.form("prediction_form"):
    cols = st.columns(2)
    for idx, col in enumerate(df.columns):
        with cols[idx % 2]:
            if col in categorical_cols:
                options = list(label_encoders[col].classes_)
                input_data[col] = st.selectbox(f"Select {col}", options)
            elif col in numerical_cols:
                input_data[col] = st.number_input(f"Enter {col}", min_value=0, format='%d')
    
    submitted = st.form_submit_button("🔮 Predict Client Type")
st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform([input_data[col]])[0]
    
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    st.markdown(f"<div class='result-box'>🎯 Predicted Client Type: <strong>{prediction}</strong></div>", unsafe_allow_html=True)










