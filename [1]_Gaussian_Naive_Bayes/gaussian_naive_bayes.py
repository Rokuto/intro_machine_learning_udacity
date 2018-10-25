import numpy as np
x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# Use Gaussian Naive Bayes to generate a preiction
from sklearn.naive_bayes import GaussianNB
# Create a classifier
clf = GaussianNB()
clf.fit(x, y)
# make a prediction
pred = clf.predict([[-0.8, -1], [1, 1]])

# Check for accuracy
from sklearn.metrics import accuracy_score
# accuracy_score(predicted_result, expected_result)
print(accuracy_score(pred, [1, 2]))
