# loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# loading datasets
iris = datasets.load_iris()

# describing datset
print(iris.DESCR)

features = iris.data  # features

labels = iris.target  # labels

# Predicting description and features
print(features[0], labels[0])

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

pred = clf.predict([[5.1, 3.5, 1.4, 0.2]])
print(pred)
