# %%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# %%
# Importing the dataset
dataset = pd.read_csv('05_SVR_regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# %%
# Feature Scaling for X
ss_X = StandardScaler()
ss_y = StandardScaler()
X = ss_X.fit_transform(X)
y = np.squeeze(ss_y.fit_transform(y.reshape(-1, 1)))

# %%
# Fitting SVR to the dataset
svr = SVR(kernel='rbf')
svr.fit(X, y)

# %%
# Visualising the SVR results
plt.scatter(X, y, color='red')
plt.plot(X, svr.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# %%
# Predicting a new result
y_pred = svr.predict(ss_X.transform(6.5).reshape(-1, 1))
y_pred = ss_y.inverse_transform(y_pred)
y_pred

# %%
