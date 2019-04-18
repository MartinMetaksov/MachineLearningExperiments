# %%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from AssociationRuleLearning.Apriori.apyori import apriori

# %%
dataset = pd.read_csv(
    'AssociationRuleLearning/Apriori/Market_Basket_Optimisation.csv', header=None)
dataset

# %%
# prepare the input as a list of lists
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j])
                         for j in range(0, dataset.shape[1])])
transactions


# %%
# products purchased 3 times a day
# that is 3*7 times a week
# 7500 total transactions
# min_support = 3*7/7500 = 0.0028 ~= 0.003
rules = apriori(transactions, min_support=0.003,
                min_confidence=0.2, min_lift=3, min_length=2)

# %%
results = list(rules)
results


# %%
