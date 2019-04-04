# %%
# y = b_0 + b_1 * x_1
# y is the dependent variable;
# x_1 is the independent variable;
# b_1 is a coefficient
# b_0 is a constant
# b_0 is the point where our regression line crosses the y axis
# in other words it's the starting point
# b_1 represents the line steepness

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
dataset = pd.read_csv('Regression/Linear_regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)


# %%
lr = LinearRegression()
lr.fit(X_train, y_train)

# %%
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lr.predict(X_train), color='cornflowerblue')
plt.title('Salary vs. Experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# %%
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, lr.predict(X_train), color='cornflowerblue')
plt.title('Salary vs. Experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
