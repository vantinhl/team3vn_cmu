import joblib


def predict(data):
    clf = joblib.load("diabetes-classification/model/rf_model.sav")
    return clf.predict(data)
