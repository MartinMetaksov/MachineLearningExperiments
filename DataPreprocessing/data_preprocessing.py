# %%
# Imports
# Mathematical tools
import numpy as np
# Plotting charts
import matplotlib.pyplot as plt
# Importing and managing datasets
import pandas as pd
# Data preprocessing
from sklearn.preprocessing import Imputer  # handle missing values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # handle categorical data
# Split dataset into test and train data
from sklearn.model_selection import train_test_split
# Feature scaling
from sklearn.preprocessing import StandardScaler

# %%
# Reading the data set
dataset = pd.read_csv('DataPreprocessing/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %%
# Handle missing data
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# %%
# Handle categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# potential problem - doing this means that e.g.
# Spain - 2 is greater than Germany - 1 and France - 0
# We can prevent this by using dummy encoding
# in other words - transform our codes into different columns

# %%
ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()

# %%
# we don't have the same problem for the y variable...
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# %%
# Split the dataset into a train and test set
# it's important to have a large enough data set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
# Feature scaling is needed when we have values with
# very large differences (e.g. column 1 has a value of 40
# while column 2 has a value of 9999999).
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
