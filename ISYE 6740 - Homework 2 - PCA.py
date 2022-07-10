# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:16:48 2020

@author: graha
"""

import time
import pandas as pd
import numpy as np
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv('food-consumption.csv')

# Clean the data
data = data.dropna().reset_index(drop=True)

'''
The data found in 'food-consumption.csv' contains 16 Eurpoean countries and their consumption of the following food items:

1. Real coffee
2. Instant coffee
3. Tea
4. Sweetener
5. Biscuits
6. Powder soup
7. Tin soup
8. Potatoes
9. Frozen fish
10. Frozen vegetables
11. Apples
12. Oranges
13. Tinned fruit
14. Jam
15. Garlic
16. Butter
17. Margerine
18. Olive oil
19. Yogurt
20. Crisp bread
'''

# Extract attributes from raw data
Anew = data.iloc[:,1:]
m,n = Anew.shape
foods = data.columns.values

# Create indicator matrix
countries = data.iloc[:,0]

# Normalize data
Anew = Anew/Anew.std(axis=0)
Anew = Anew.T

# # PCA
mu_std = np.mean(Anew,axis = 1)
xc = Anew - mu_std[:,None]

C = np.dot(xc,xc.T)/m

K = 2
S,W = ll.eigs(C,k = K)
# S, W = np.linalg.eigh(C)
# vec1 = W[-1].T
# vec2 = W[-2].T

dim1 = np.dot(W[:,0].T,xc)/np.sqrt(S[0])
dim2 = np.dot(W[:,1].T,xc)/np.sqrt(S[1])
dim1 = dim1.real # X-values
dim2 = dim2.real # Y-values

# dim1 = np.dot(vec1.T,xc)/np.sqrt(S[0])
# dim2 = np.dot(vec1.T,xc)/np.sqrt(S[1])


# 3)
plt.figure(figsize=(10, 10))
plt.axis([-0.5,0.5,-0.5,0.5])
plt.scatter(W[:,0], W[:,1], c=np.random.rand(n))
plt.title('Top 2 Principal Components of the Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i in range(n):
    plt.annotate(foods[i+1], (W[i,0], W[i,1]))

plt.savefig('Composition of Top 2 Principal Components.png', dpi=80)
plt.show()

# 4)

plt.figure(figsize=(10, 10))
plt.axis([-3,3,-3,3])
plt.scatter(dim1, dim2, c=np.random.rand(m))
plt.title('Representations of Countries in Top 2 Principal Components')
plt.xlabel('Projection onto Principal Component 1')
plt.ylabel('Projection onto Principal Component 2')
for i in range(m):
    plt.annotate(countries[i], (dim1[i], dim2[i]))

plt.savefig('Countries Projected onto Principle Component Space.png', dpi=80)
plt.show()



    