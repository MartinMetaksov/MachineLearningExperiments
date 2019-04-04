# %%
# y = b_0 + b_1 * x_1 + b_2 * x_1ˆ2

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# %%
dataset = pd.read_csv('04_Polynomial_regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# %%
lr = LinearRegression()
lr.fit(X, y)

# %%
pf = PolynomialFeatures(degree=2)
X_poly = pf.fit_transform(X)


# %%
X_poly

# %%
lr_2 = LinearRegression()
lr_2.fit(X_poly, y)

# %%
plt.scatter(X, y, color='red')
plt.plot(X, lr.predict(X), color='cornflowerblue')
plt.title('Truth or bluff (Linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %%
plt.scatter(X, y, color='red')
plt.plot(X, lr_2.predict(pf.fit_transform(X)), color='cornflowerblue')
plt.title('Truth or bluff (Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %%
pf_n = PolynomialFeatures(degree=5)
X_poly_n = pf_n.fit_transform(X)
lr_n = LinearRegression()
lr_n.fit(X_poly_n, y)
plt.scatter(X, y, color='red')
plt.plot(X, lr_n.predict(pf_n.fit_transform(X)), color='cornflowerblue')
plt.title('Truth or bluff (Polynomial regression, degree=5)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %%
y_example = lr_n.predict(pf_n.fit_transform(6.5))
y_example

# %%
y_example_simple = lr.predict(6.5)
y_example_simple
