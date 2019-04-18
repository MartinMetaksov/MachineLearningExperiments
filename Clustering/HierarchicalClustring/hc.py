# %%
# Importing the libraries
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# %%
dataset = pd.read_csv(
    'Clustering/HierarchicalClustring/Mall_Customers.csv')
dataset


# %%
X = dataset.iloc[:, [3, 4]].values
X


# %%
# Using dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian distances')
plt.show()


# %%
# Fitting hierarchical clustering
hc = AgglomerativeClustering(
    n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
y_hc

# %%
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=10,
            c='red', label='Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=10,
            c='green', label='Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=10,
            c='cornflowerblue', label='Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=10,
            c='cyan', label='Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=10,
            c='magenta', label='Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()


# %%
