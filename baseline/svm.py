from sklearn import datasets, svm
import numpy as np
features = np.load("features.npy")
labels = np.load("labels.npy")
clf = svm.LinearSVC()
clf.fit(features, labels)

print(clf.predict([features[0]]))
