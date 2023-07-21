import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st


# random seed
seed = 42

# Read original dataset
diabetes_df = pd.read_csv("./data/diabetes.csv")
diabetes_df.sample(frac=1, random_state=seed)

# selecting features and target data
X = diabetes_df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = diabetes_df[['Outcome']]

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# save the model to disk
<<<<<<< HEAD
joblib.dump(clf, "rf_model.sav")
=======
joblib.dump(clf, "./model/rf_model.sav")
>>>>>>> 3f5ebd334f486c0583671cbee93bc0a7ded9998b


print(diabetes_df.describe(include='all'))
