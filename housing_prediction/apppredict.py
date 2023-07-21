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

# Header with logo
logo_path = "team3vn_cmu.jpg"
# Center the logo on the page
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
with col1:
    st.write("")
with col2:
    st.write("")
with col3:
    st.write("")
with col4:
    st.image(logo_path, width=220, caption="")
with col5:
    st.write("")
with col6:
    st.write("")
with col7:
    st.write("")
with col8:
    st.write("")
with col9:
    st.write("")

# Header
st.markdown(
    "<h1 style='text-align: center; font-size: 30px; background-color: red; color: #FFFFFF'; margin: 20px;padding: 20px;>"
    "&nbsp; Welcome to Team3VN-CMU House Price Prediction &nbsp;"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center; font-size: 30px; background-color: #f2f2f2; line-height: 0.3;'>Predict House Prices</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center; font-size: 26px; background-color: #f8f8f8; color: blue; line-height: 0.3;'>"
    "Member: Dieu - Man - Sanh - Thuan - Tinh - Trinh"
    "</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; font-size: 26px; background-color: #f8f8f8; color: gray; line-height: 0.3;'>"
    "CMU 2023"
    "</h3>",
    unsafe_allow_html=True
)

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
    ax.scatter(df['RM'],    df['MEDV'])
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("#### Model Performance")
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))
    save_model(model, "LinearRegression.pkl")
    return model

# Function to train and evaluate the Random Forest model
def train_model_random_forest(df):
    st.write("### Model Random Forest Training and Evaluation")

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict(X_test)

    st.write("#### Model Random Forest Performance")
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))
    save_model(model_rf, "RandomForest.pkl")
    return model_rf

# Function to predict house prices using Linear Regression
def predict_price_linear_regression(model, input_data):
    # Ensure input_data has the same number of features as the training dataset
    if input_data.shape[1] != model.coef_.shape[0]:
        raise ValueError("Number of features in input data does not match the model")

    prediction = model.predict(input_data)
    return prediction

# Function to predict house prices using Random Forest
def predict_price_random_forest(model_rf, input_data):
    prediction_rf = model_rf.predict(input_data)
    return prediction_rf

# Function to visualize the predicted prices using a pie chart
def visualize_prediction_pie(prediction_lr, prediction_rf):
    labels = ['Linear Regression', 'Random Forest']
    sizes = [prediction_lr[0], prediction_rf[0]]
    explode = (0.1, 0)  # explode the first slice

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

def main():
    df = load_data()
    explore_data(df)
    model_lr = train_model(df)
    model_rf = train_model_random_forest(df)

    st.write("### House Price Prediction")

    st.write("**Enter the following features to get the predicted price:**")
    st.header("")
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.write("**CRIM** - Per Capita Crime Rate:")
        crim = st.number_input("", key="crim", value=0.0, step=0.01)
        st.write("**INDUS** - Proportion of Non-Retail Business Acres:")
        indus = st.number_input("", key="indus", value=0.0, step=0.01)
        st.write("**NOX** - Nitric Oxides Concentration (parts per 10 million):")
        nox = st.number_input("", key="nox", value=0.0, step=0.01)
        st.write("**AGE** - Proportion of Owner-Occupied Units Built Prior to 1940:")
        age = st.number_input("", key="age", value=0.0, step=0.01)
        st.write("**RAD** - Index of Accessibility to Radial Highways:")
        rad = st.number_input("", key="rad", value=0.0, step=1.0)
        st.write("**PTRATIO** - Pupil-Teacher Ratio by Town:")
        ptratio = st.number_input("", key="ptratio", value=0.0, step=0.01)
        st.write("**LSTAT** - Percentage of Lower Status of the Population:")
        lstat = st.number_input("", key="lstat", value=0.0, step=0.01)

    with input_col2:
        st.write("**ZN** - Proportion of Residential Land Zoned:")
        zn = st.number_input("", key="zn", value=0.0, step=0.5)
        st.write("**CHAS** - Charles River Dummy Variable:")
        chas = st.selectbox("", key="chas", options=[0, 1])
        st.write("**RM** - Average Number of Rooms per Dwelling:")
        rm = st.number_input("", key="rm", value=0.0, step=0.01)
        st.write("**DIS** - Weighted Distances to Five Boston Employment Centers:")
        dis = st.number_input("", key="dis", value=0.0, step=0.01)
        st.write("**TAX** - Full-Value Property Tax Rate per $10,000:")
        tax = st.number_input("", key="tax", value=0.0, step=1.0)
        st.write("**B** - Proportion of Blacks:")
        b = st.number_input("", key="b", value=0.0, step=0.01)
        st.write("**MEDV** - Median value of owner-occupied homes in $1000's:")
        medv = st.number_input("", key="medv", value=0.0, step=0.01)

    submitted = st.button('Predict Price')

    if submitted:
        input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv]])
        prediction_lr = predict_price_linear_regression(model_lr, input_data)
        st.write("### **Predicted House Price using Linear Regression:**", prediction_lr)

        prediction_rf = predict_price_random_forest(model_rf, input_data)
        st.write("### **Predicted House Price using Random Forest:**", prediction_rf)

        visualize_prediction_pie(prediction_lr, prediction_rf)

        # Plot the predicted vs actual prices
        # import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Actual Prices')
        ax.set_ylabel('Predicted Prices')
        ax.set_title('Predicted vs Actual Prices')
        st.subheader('Predicted vs Actual Prices')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
