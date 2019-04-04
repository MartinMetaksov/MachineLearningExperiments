# %%
# Assumptions of linear regression:
# 1) Linearity
# 2) Homoscedasticity
# 3) Multivariate normality
# 4) Independence of errors
# 5) Lack of multicollinearity

# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # handle categorical data
import statsmodels.formula.api as sm

# %%
dataset = pd.read_csv('Regression/Multiple_linear_regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %%
# Encoding the categorical column
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray()

# %%
# Avoiding the dummy variable trap
X = X[:, 1:]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
lr = LinearRegression()
lr.fit(X_train, y_train)

# %%
y_pred_simple = lr.predict(X_test)

# %%
# Backward elimination

# %%
# artificially add x_0 (for b_0)
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X

# %%
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
sl = 0.5
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

# %%
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()


# %%
X_opt = X[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

# %%
X_opt = X[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

# %%
X_opt = X[:, [0, 3]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

# %%
# R&D span is the dependent variable with most significance towards the predictions
