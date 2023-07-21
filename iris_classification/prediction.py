import joblib


def predict(data):
    clf = joblib.load("iris_classification/rf_model.sav")
    return clf.predict(data)
