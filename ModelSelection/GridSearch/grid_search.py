# %%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# %%
dataset = pd.read_csv(
    'ModelSelection/GridSearch/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# %%
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# %%
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# %%
y_pred = classifier.predict(X_test)
y_pred

# %%
cm = confusion_matrix(y_test, y_pred)
cm

# %%
# Applying k-Fold cross validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

# %%
accuracies.mean()

# %%
accuracies.std()

# %%
# Grid search
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {
    'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}]
grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)

# %%
best_accuracy = grid_search.best_score_
best_accuracy

# %%
best_params = grid_search.best_params_
best_params
