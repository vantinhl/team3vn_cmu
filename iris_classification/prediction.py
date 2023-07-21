import joblib



def predict(data):
    clf = joblib.load("data/rf_model.sav", 'rb')
    return clf.predict(data)
