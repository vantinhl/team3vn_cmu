def main():
    # st.title("House Price Prediction")
    df = load_data()
    # describe_attributes()
    explore_data(df)
    model_lr = train_model(df)
    model_rf = train_model_random_forest(df)

    st.write("### House Price Prediction")

    st.write("**Enter the following features to get the predicted price:**")
    
    with st.form(key='input_form'):
        col1, col2 = st.columns(2)

        with col1:
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

        with col2:
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

        submitted = st.form_submit_button('Predict Price')

    if submitted:
        input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv]])
        prediction_lr = predict_price_linear_regression(model_lr, input_data)
        st.write("### **Predicted House Price using Linear Regression:**", prediction_lr)

        prediction_rf = predict_price_random_forest(model_rf, input_data)
        st.write("### **Predicted House Price using Random Forest:**", prediction_rf)

        visualize_prediction_pie(prediction_lr, prediction_rf)

if __name__ == "__main__":
    main()
