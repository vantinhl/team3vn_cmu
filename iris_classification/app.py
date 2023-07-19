import requests
import streamlit as st
from src.train import Classifier
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Header
st.markdown(
    "<h1 style='text-align: center; font-size: 28px; background-color: red; color: #FFFFFF'; margin: 20px;padding: 20px;>"
    "&nbsp; Welcome to Team3VN-CMU House Price Prediction &nbsp;"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center; font-size: 24px; background-color: #f2f2f2;'>Predict House Prices</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center; font-size: 20px; background-color: #f8f8f8; color: blue;'>"
    "Member: Dieu - Man - Sanh - Thuan - Tinh - Trinh"
    "</h2>",
    unsafe_allow_html=True
)


# Function to predict house prices locally
def predict_price_local(data):
    dt = list(map(float, data))

    req = {
        "data": [
            dt
        ]
    }

    cls = Classifier()
    return cls.load_and_test(req)


# Function to predict house prices using AWS
def predict_price_aws(data):
    API_URL = "https://ti53furxkb.execute-api.us-east-1.amazonaws.com/test/predict"

    dt = list(map(float, data))

    req = {
        "data": [
            dt
        ]
    }

    r = requests.post(API_URL, json=req)
    return r.json()


# Function to read dataset for training
def read_training_dataset():
    # Replace with the path to your training dataset file
    training_file_path = "src/dataset.csv"
    training_data = pd.read_csv(training_file_path)
    return training_data


# Function to read dataset for testing
def read_testing_dataset():
    # Replace with the path to your testing dataset file
    testing_file_path = "src/dataset_test.csv"
    testing_data = pd.read_csv(testing_file_path)
    return testing_data




# Slider Inputs
slider_labels = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]
slider_values = [0.0] * len(slider_labels)
slider_min = [0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0]
slider_max = [100.0, 100.0, 100.0, 1, 1.0, 10.0, 100.0, 10.0, 24, 1000.0, 30.0, 100.0, 50.0]

for i, label in enumerate(slider_labels):
    slider_values[i] = st.slider(label, slider_min[i], slider_max[i], slider_min[i])

# Store initial values
initial_values = slider_values.copy()

left, right = st.columns(2)
with left:
    if st.button("Predict Local"):
        ret = predict_price_local(slider_values)
        price_prediction = ret['prediction'][0]
        st.write(f"Predicted Price: ${price_prediction}")

with right:
    if st.button("Predict AWS"):
        ret = predict_price_aws(slider_values)
        price_prediction = ret['prediction'][0]
        st.write(f"Predicted Price: ${price_prediction}")

# Update Flowchart with final result
x = np.arange(len(slider_labels))

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', 'orange', 'purple', 'pink', 'skyblue', 'gold']
ax.bar(x, slider_values, align='center', alpha=0.5, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(slider_labels, rotation=45)
ax.set_ylabel("Value")
ax.set_title("House Price Prediction Features")
ax.grid(True)

for i, v in enumerate(slider_values):
    ax.text(i, v, f"{v:.2f}", color='black', ha='center', va='bottom')

st.pyplot(fig)

# Function to export data to a file
#def export_data(data):
#    # Replace with the desired file path to export the data
 #   export_file_path = "src/dataset_result.csv"
 #   data.to_csv(export_file_path, index=False)

# Export data
#updated_data = pd.DataFrame([slider_values], columns=slider_labels)
#export_data(updated_data)
