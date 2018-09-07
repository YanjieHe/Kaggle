import numpy as np
import pandas as pd
import seaborn as sea
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import Imputer

pd.options.display.max_rows = 9999
pd.options.display.max_columns = 9999

def split_columns_by_type(df):
    numeric_columns = [ ]
    category_columns = [ ]
    for column in df.columns:
        if df[column].dtype == np.dtype("int64"):
            numeric_columns.append(column)
        else:
            category_columns.append(column)
    return numeric_columns, category_columns

def plot_corr_mat(df):
    cols = df.corr().nlargest(len(numeric_columns), 'SalePrice')['SalePrice'].index
    corr_mat = np.corrcoef(df[cols].values.T)
    sea.heatmap(corr_mat, cbar=True, annot=False, square=True, yticklabels=cols.values, xticklabels=cols.values)

def drop_highly_correlated_features(df):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
    return to_drop

def remove_columns(df, to_drop):
    cols = df.columns.tolist()
    drop_set = set(to_drop)
    new_columns = [ ]
    for col in cols:
        if col in drop_set:
            pass
        else:
            new_columns.append(col)
    return df[new_columns]


def fill_missing_values_with_mean(df, column_name):
    imputer = Imputer(missing_values = "NaN",
                      strategy = "mean", axis = 0)
    values = df[column_name].values.reshape(-1, 1)
    imputer = imputer.fit(values)
    df[column_name] = imputer.fit_transform(values)
            
def fill_missing_values_with_most_frequent(df, column_name):
    imputer = Imputer(missing_values = "NaN",
                      strategy = "most_frequent", axis = 0)
    values = df[column_name].values.reshape(-1, 1)
    imputer = imputer.fit(values)
    df[column_name] = imputer.fit_transform(values)
            
df = pd.read_csv("train.csv")

missing_data_percentage = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)

df = df.drop(missing_data_percentage[missing_data_percentage > 0].index, 1)
numeric_columns, category_columns = split_columns_by_type(df)
corr_with_sale_price = df.corr()["SalePrice"]
selected_cols = corr_with_sale_price[corr_with_sale_price.abs() > 0.5]
to_drop = drop_highly_correlated_features(df_t)
df_t = df[selected_cols.index.tolist() + category_columns]
df_t = remove_columns(df_t, to_drop + ["Id"])

df_test = pd.read_csv("test.csv")
df_test = df_test[remove_columns(df_t, ["SalePrice"]).columns.tolist()]
df_test_missing = df_test[["MSZoning", "Utilities", "Functional", "GarageCars"   , "KitchenQual"  , "SaleType"     , "TotalBsmtSF"  , "GarageArea", "Exterior2nd", "Exterior1st"]]

for col in ["MSZoning", "Utilities", "Functional", "GarageCars"   , "KitchenQual"  , "SaleType"     , "TotalBsmtSF"  , "GarageArea", "Exterior2nd", "Exterior1st"]:
    if col in df_test.columns:
        if col in numeric_columns:
            fill_missing_values_with_mean(df_test, col)
        else:
            print(df_test[col].tolist()[0])
            df_test.fillna(df_test[col].tolist()[0])

X_test = pd.get_dummies(df_test)

Y = df_t["SalePrice"]
X = remove_columns(df_t, ["SalePrice"])
X = pd.get_dummies(X)
X = X[X_test.columns.tolist()]
model = sm.OLS(Y, X).fit()

predictions = model.predict(X_test)
Y_test = pd.read_csv("test.csv")
Y_test = Y_test[["Id"]]
Y_test["SalePrice"] = predictions