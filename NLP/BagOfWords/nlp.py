# %% Importing the libraries
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# %%
nltk.download('stopwords')

# %%
dataset = pd.read_csv(
    'NLP/BagOfWords/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# %%
# Cleaning the text...
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# %%
corpus

# %%
# Creating the bag of words model (a sparse matrix)
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# %%
y_pred = classifier.predict(X_test)
y_pred

# %%
cm = confusion_matrix(y_test, y_pred)
cm
