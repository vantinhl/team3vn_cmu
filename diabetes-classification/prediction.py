import joblib


def predict(data):
<<<<<<< HEAD
    clf = joblib.load("rf_model.sav")
=======
    clf = joblib.load("./model/rf_model.sav")
>>>>>>> 3f5ebd334f486c0583671cbee93bc0a7ded9998b
    return clf.predict(data)
