import joblib


def predict(data):
    clf = joblib.load("./model/rf_model.sav")
    return clf.predict(data)
