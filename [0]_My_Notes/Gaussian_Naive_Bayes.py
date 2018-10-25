from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import numpy as np
x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

classifier = GaussianNB()
classifier.fit(x, y)
score = metrics.accuracy_score(y, classifier.predict(x))
print("Accuracy: %f" % score)
