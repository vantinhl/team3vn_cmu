import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Classifying type of Diabetes')
st.markdown('Real patient data to manipulate and to predict diabetes (Positive or Negative) using demographic and lab variables.')

st.header("Value")
col1, col2 = st.columns(2)

with col1:
    st.text("Select value")
    pregnancies = st.slider('Pregnancies', 0.0, 17.0, 3.0)
    glucose = st.slider('Glucose', 0.0, 199.0, 117.0)
    bloodPressure = st.slider('BloodPressure', 0.0, 122.0, 72.0)
    skinThickness = st.slider('SkinThickness', 0.0, 99.0, 23.0)

with col2:
    st.text("Select value")
    insulin = st.slider('Insulin', 0.0, 846.0, 30.5)
    bmi = st.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pf = st.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.3725 )
    age = st.slider('Age', 21.0, 81.0, 29.0)

st.text('')
if st.button("Predict type of Diabetes"):
    result = predict(
        np.array([[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetes_pf, age]]))
    st.text(result[0])
    if result[0] == 1:
        st.text("Positive for diabetes")
    else:
        st.text("Negative for diabetes")
   


st.text('')
st.text('')
st.markdown('`Create by` team3vn_cmu')
