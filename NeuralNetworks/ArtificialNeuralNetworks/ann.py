# %% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense


# %%
# a classification problem
dataset = pd.read_csv(
    'NeuralNetworks/ArtificialNeuralNetworks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# %%
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# %%
# Initialising the ANN
classifier = Sequential()

# %%
# Add Input and hidden layers
classifier.add(Dense(activation='relu', input_dim=11,
                     units=6, kernel_initializer='uniform'))

# %%
# Add second hidden layer
classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform'))

# %%
# Add output layer
classifier.add(Dense(activation='sigmoid', units=1,
                     kernel_initializer='uniform'))

# %%
# Compiling the ANN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# %%
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# %%
cm = confusion_matrix(y_test, y_pred)
cm

# %%
100 - 100*(cm[0][1] + cm[1][0]) / (sum(sum(cm)))
