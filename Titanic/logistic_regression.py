import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import Imputer

def map_column(df, column_name, value_map):
    values = df[column_name].tolist()
    result = [ ]
    for v in values:
        result.append(value_map[v])
    df[column_name] = result

def fill_missing_values_with_mean(df, column_name):
    imputer = Imputer(missing_values = "NaN",
                      strategy = "mean", axis = 0)
    values = df[column_name].values.reshape(-1, 1)
    imputer = imputer.fit(values)
    df[column_name] = imputer.fit_transform(values)

selected_column_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
df = pd.read_csv("train.csv")
df = df[["Survived"] + selected_column_names]

map_column(df, "Sex", {"male": 0, "female": 1})
fill_missing_values_with_mean(df, "Age")

X = df[selected_column_names].values
Y = df["Survived"].values
model = linear_model.LogisticRegression()
model.fit(X, Y)

test_dataset = pd.read_csv("test.csv")
test_data = test_dataset[selected_column_names]
map_column(test_data, "Sex", {"male": 0, "female": 1})
fill_missing_values_with_mean(test_data, "Age")
fill_missing_values_with_mean(test_data, "Fare")

X_test = test_data.values
Y_pred = model.predict(X_test)

output_dataset = test_dataset[["PassengerId"]]
output_dataset["Survived"] = Y_pred
output_dataset.to_csv("submission.csv", index=False)