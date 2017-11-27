from sklearn import datasets, svm
import numpy as np
iris = datasets.load_iris()
np.unique(iris.target)
clf = svm.LinearSVC()
clf.fit(iris.data, iris.target)

print(clf.predict([[ 5.0,  3.6,  1.3,  0.25]]))
