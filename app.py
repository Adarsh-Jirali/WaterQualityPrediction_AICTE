import pandas as pd
import numpy as np
import streamlit as st
import joblib
import pickle
import base64

def bg_local(img_path):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)

bg_local("image5.jpg")

st.title("Water Quality Predictor")
st.write("This app predicts water quality based on year and station id.")


# Load the model
model=joblib.load("pollution_model.pkl")
model_cols=joblib.load("model_columns.pkl")

# Input fields for year and station id
year_input=st.number_input("Enter Year",min_value=2000,max_value=2100,value=2022)
station_id=st.number_input("Enter Station id",value=1)

#encode and prodict
if st.button("Predict"):
    if not station_id:
        st.warning("Please enter a station id")
    else:
        #input
        input_df=pd.DataFrame({"year":[year_input],"id":[station_id]})
        input_encoded=pd.get_dummies(input_df,columns=['id'])

        #align columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col]=0
        input_encoded=input_encoded[model_cols]

        #predict
        predicted_pollutants= model.predict(input_encoded)[0]
        pollutants=['O2','NO3','NO2','SO4','PO4','CL']

        st.subheader(f"Predicted pollutants levels for the station '{station_id}' in  {year_input}")
        predicted_values={}
        for p,val in zip(pollutants,predicted_pollutants):
            st.write(f"{p}: {val:.2f}")