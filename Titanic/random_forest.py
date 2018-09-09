import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

def fill_missing_values_with_mean(df, column_name):
    imputer = Imputer(missing_values = "NaN",
                      strategy = "mean", axis = 0)
    values = df[column_name].values.reshape(-1, 1)
    imputer = imputer.fit(values)
    df[column_name] = imputer.fit_transform(values)

def process_categories(X):
    X["Sex"] = X["Sex"].replace("male", 1)
    X["Sex"] = X["Sex"].replace("female", 0)

    X["Embarked"] = X["Embarked"].replace("S", 0)
    X["Embarked"] = X["Embarked"].replace("C", 1)
    X["Embarked"] = X["Embarked"].replace("Q", 2)

pd.options.display.max_rows = 9999
pd.options.display.max_columns = 9999
pd.options.mode.chained_assignment = None

df = pd.read_csv("train.csv")
df["Embarked"].fillna("S", inplace = True)
fill_missing_values_with_mean(df, "Age")

df = df[["Survived", "Pclass", "Sex",
         "Age", "SibSp", "Parch",
         "Fare", "Embarked"]]

Y = df["Survived"].values
X = df[["Pclass", "Sex", "Age",
        "SibSp", "Parch", "Fare",
        "Embarked"]]
process_categories(X)
X = X.values

classifier = RandomForestClassifier()
model = classifier.fit(X, Y)

df_test = pd.read_csv("test.csv")
fill_missing_values_with_mean(df_test, "Age")
fill_missing_values_with_mean(df_test, "Fare")
X_test = df_test[["Pclass", "Sex", "Age",
        "SibSp", "Parch", "Fare",
        "Embarked"]]

process_categories(X_test)
X_test = X_test.values
prediction = model.predict(X_test).tolist()

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": prediction})

submission.to_csv("submission.csv", index = False)