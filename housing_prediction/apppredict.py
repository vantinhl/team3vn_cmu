import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Function to load the dataset
@st.cache_data()
def load_data():
    url = 'Boston_Housing.csv'
    return pd.read_csv(url)
# Function to describe the attribute information
def describe_attributes():
    st.write("## Data Set Characteristics")
    st.write("- The Boston Housing dataset contains information about various features of houses in Boston.")
    st.write("- It includes attributes such as per capita crime rate, proportion of residential land zoned for lots over 25,000 sq.ft., average number of rooms per dwelling, etc.")
    st.write("- The target variable is the median value of owner-occupied homes in thousands of dollars.")
    st.write("- The dataset consists of 506 instances and 13 input features.")
    st.write('===================================================================')
    st.write("## Attribute Information")
    st.write("- CRIM: Per capita crime rate by town")
    st.write("- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.")
    st.write("- INDUS: Proportion of non-retail business acres per town")
    st.write("- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
    st.write("- NOX: Nitric oxides concentration (parts per 10 million)")
    st.write("- RM: Average number of rooms per dwelling")
    st.write("- AGE: Proportion of owner-occupied units built prior to 1940")
    st.write("- DIS: Weighted distances to five Boston employment centers")
    st.write("- RAD: Index of accessibility to radial highways")
    st.write("- TAX: Full-value property tax rate per $10,000")
    st.write("- PTRATIO: Pupil-teacher ratio by town")
    st.write("- B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town")
    st.write("- LSTAT: Percentage of lower status of the population")
    st.write("- MEDV: Median value of owner-occupied homes in $1000s")
    st.write('===================================================================')
# Function to explore the dataset
def explore_data(df):
    describe_attributes()
    st.write("### Dataset Summary")
    st.write(df.head())
    st.write("### Dataset Shape")
    st.write(df.shape)
    st.write("### Dataset Description")
    st.write(df.describe())

    # Data visualization
    st.write("### Data Visualization")
    st.write("#### Scatter Plot")

    fig, ax = plt.subplots()
    ax.scatter(df['RM'], df['MEDV'])
    ax.set_xlabel('RM: Average number of rooms per dwelling')
    ax.set_ylabel('Median value of owner-occupied homes in $1000s')
    st.pyplot(fig)

    st.write("#### Histogram")
    fig, ax = plt.subplots()
    ax.hist(df['MEDV'])
    ax.set_xlabel('Median value of owner-occupied homes in $1000s')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write("#### Correlation Heatmap")
    corr_matrix = df.corr().values
    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix, cmap='coolwarm')
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticklabels(df.columns)
    fig.colorbar(im)
    st.pyplot(fig)

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to train and evaluate the model
def train_model(df):
    st.write("### Model Training and Evaluation")

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("#### Model Performance")
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))
    save_model(model, "LinearRegression.pkl")
    return model

# Function to train and evaluate the model Randomforest
def train_modelR(df):
    st.write("### Model Randomforest Training and Evaluation")

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelR = RandomForestRegressor(n_estimators=100, random_state=42)
    modelR.fit(X_train, y_train)

    y_pred = modelR.predict(X_test)

    st.write("#### Model Randomforest Performance")
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))
    save_model(modelR, "RandomForest.pkl")
    return modelR

# Function to predict house prices using LinearRegression

def predict_price(model, input_data):
    # Ensure input_data has the same number of features as the training dataset
    if input_data.shape[1] != model.coef_.shape[0]:
        raise ValueError("Number of features in input data does not match the model")

    prediction = model.predict(input_data)
    return prediction

# Function to predict house prices using RandomForest
def predict_priceR(modelR, input_data):
    predictionR = modelR.predict(input_data)
    return predictionR

# Function to visualize the predicted prices
def visualize_prediction(df, predicted_prices):
    sorted_indices = np.argsort(df['RM'])
    sorted_predicted_prices = predicted_prices.flatten()[sorted_indices]

    fig, ax = plt.subplots()
    ax.scatter(df['RM'], df['PRICE'], label='Actual')
    ax.scatter(df['RM'].iloc[sorted_indices], sorted_predicted_prices, color='red', label='Predicted')
    ax.set_xlabel('RM')
    ax.set_ylabel('PRICE')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("House Price Prediction")
    df = load_data()
    #describe_attributes()
    explore_data(df)
    model = train_model(df)
    modelR = train_modelR(df)

    st.write("### House Price Prediction")
    st.write("Enter the following features to get the predicted price:")
    crim = st.number_input("CRIM - Per Capita Crime Rate:", value=0.0, step=0.01)
    zn = st.number_input("ZN - Proportion of Residential Land Zoned:", value=0.0, step=0.5)
    indus = st.number_input("INDUS - Proportion of Non-Retail Business Acres:", value=0.0, step=0.01)
    chas = st.selectbox("CHAS - Charles River Dummy Variable:", options=[0, 1])
    nox = st.number_input("NOX - Nitric Oxides Concentration (parts per 10 million):", value=0.0, step=0.01)
    rm = st.number_input("RM - Average Number of Rooms per Dwelling:", value=0.0, step=0.01)
    age = st.number_input("AGE - Proportion of Owner-Occupied Units Built Prior to 1940:", value=0.0, step=0.01)
    dis = st.number_input("DIS - Weighted Distances to Five Boston Employment Centers:", value=0.0, step=0.01)
    rad = st.number_input("RAD - Index of Accessibility to Radial Highways:", value=0.0, step=1.0)
    tax = st.number_input("TAX - Full-Value Property Tax Rate per $10,000:", value=0.0, step=1.0)
    ptratio = st.number_input("PTRATIO - Pupil-Teacher Ratio by Town:", value=0.0, step=0.01)
    b = st.number_input("B - Proportion of Blacks:", value=0.0, step=0.01)
    lstat = st.number_input("LSTAT - Percentage of Lower Status of the Population:", value=0.0, step=0.01)
    medv = st.number_input("Median value of owner-occupied homes in $1000's:", value=0.0, step=0.01)

    input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv]])

    if st.button("Predict Price"):
        prediction = predict_price(model, input_data)
        st.write("### Predicted House Price using LinearRegression:", prediction)
      #  visualize_prediction(df, prediction)
      #  st.write(prediction)

        prediction = predict_priceR(modelR, input_data)
        st.write("### Predicted House Price using RandomForest:", prediction)

   # if st.button("Predict RandomForest"):
    #    prediction = predict_priceR(modelR, input_data)
     #   st.write("## Predicted House Price using RandomForest:", prediction)
      #  visualize_prediction(df, prediction)
       # st.write(prediction)

if __name__ == "__main__":
    main()
