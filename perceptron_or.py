# import the required packages
from pyimagesearch.nn.perceptron import Perceptron
import numpy as np

# construct the OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# define perceptron and train it
print("[INFO] Training perceptron...")
p = Perceptron(X.shape[1], alpha = 0.1)
p.fit(X, y, epochs = 20)

# evaluating perceptron
print("[INFO] Testing perceptron...")

# loop over the data points to predict
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to console with the input data point
	pred = p.predict(x)
	print("[INFO] data {}, ground-truth = {}, pred = {}".format(x, target[0], pred))