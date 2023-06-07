import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Train a logistic regression to check whether a flower is iris virginica or not
'''
Q) What is logistic regression?
A)A regression algoritm which does classification.

  Calculates probability of belonging to a particualr class

  If p>50%->1
  If p<50%->0
'''

"""
Q)How does lohistic regrssion work
A) It takes your features and labels[Training Data].
   
   Fits a linear moedl(weights and biases).
   
   And instead of giving you the result,it gives you the logistic of the result.
   
"""

iris = datasets.load_iris()
""" 
print(iris.keys(), '\n')
print(iris.target, '\n')
print(iris.DESCR) 
"""
# Shift+alt+a to comment multiple lines
x = iris["data"][:, 3:]
y = (iris.target == 2).astype(np.int)
print(iris.data.shape)
print()
print(x)
print()
print(y)

# train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x, y)
print()
print(clf.predict(([[1.6]])))
print(clf.predict(([[2.6]])))

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_new)
Y_prob = clf.predict_proba(X_new)
plt.plot(X_new, Y_prob[:, 1], "g-")
plt.title("virganica")
plt.show()
