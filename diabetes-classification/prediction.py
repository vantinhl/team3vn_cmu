import joblib


def predict(data):
    clf = joblib.load("diabetes_model.sav")

    return clf.predict(data)
