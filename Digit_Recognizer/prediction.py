import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def read_data(file_name):
    df = pd.read_csv(file_name)
    (nrows, ncols) = df.shape
    num_features = ncols - 1
    X = df.iloc[:, 1:(num_features+1)].values
    Y = df.iloc[:, 0].values
    return train_test_split(X, Y,
                            test_size = 0.3, random_state = 0)

def train_model(X_train, X_test, Y_train, Y_test):
    classifier = RandomForestClassifier()
    model = classifier.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    print_result(prediction, Y_test)
    return model

def print_result(prediction, Y_test):
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, prediction))
    print("Accuracy score: ")
    print(accuracy_score(Y_test, prediction))
    print("Classification Report: ")
    print(classification_report(Y_test, prediction))

def predict_test_data(model, file_name):
    df = pd.read_csv(file_name)
    prediction = model.predict(df.values)
    output_prediction(prediction)

def output_prediction(prediction):
    f = open("submission.csv", "w")
    f.write("ImageId,Label\n")
    for i in range(len(prediction)):
        f.write(str(i + 1))
        f.write(",")
        f.write(str(prediction[i]))
        if i != len(prediction) - 1:
            f.write("\n")
    f.close()
    print("result has been written to 'submission.csv'")
        
if __name__ == "__main__":
    (X_train, X_test, Y_train, Y_test) = read_data("train.csv")
    model = train_model(X_train, X_test, Y_train, Y_test)
    predict_test_data(model, "test.csv")
