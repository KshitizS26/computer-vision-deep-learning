# import the required packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# make_blobs create blobs of normally distributed data points
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse
# to maintain reproducibility 
# np.random.seed(1)

def sigmoid_activation(x):
	# compute the sigmoid activation value for a given input
	# sigmoid activation function is a non-linear activation function that is used to threshold prediction
	return 1.0 / (1 + np.exp(-x))

def predict(X, W):
	# take the dot product between our features and weight matrix
	# X: set of input data
	# W: weights
	preds = sigmoid_activation(X.dot(W))

	# apply a step function to threshold the outputs to binary class labels
	preds[preds <= 0.5] = 0
	preds[preds > 0] = 1

	# return the predictions
	return preds

# construct the argument parse and parse the arguments
# --epochs: number of iterations of gradient descent
# --alpha: learning rate for the gradient descent
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type = float, default = 100, help = "# of epochs")
ap.add_argument("-a", "--alpha", type = float, default = 0.01, help = "learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1000 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 1.5, random_state = 1)

# reshape y to [n_samples x 1]
y = y.reshape((y.shape[0], 1))
print(y.shape)

# insert a column of 1's as the last entry in the feature 
# matrix -- bias trick treat the bias as a trainable parameter within
# the weight matrix
# shape of X: [n_samples x (n_features + 1)]
X = np.c_[X, np.ones((X.shape[0]))]
print(X.shape)

# partition the data into training and testing splits using 50%
# of the data from training and remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size = 0.5, random_state = 42)

# initialize weight matrix randomly and list of losses for tracking loss after each epoch
print("[INFO] Training...")
# shape of W: [(n_features + 1) x 1]
W = np.random.randn(X.shape[1], 1)
print(W.shape)
losses = []

# loop over the desired number of epochs (allowing the training procedure to see each of the training points a total of 100 times)
for epoch in np.arange(0, args["epochs"]):

	# take the dot product between our feature 'X' and the weight matrix 'W'
	# pass this value through sigmoid activation function, thereby giving 
	# prediction on the dataset
	# shape of dot_product: [n_samples x 1]
	dot_product = trainX.dot(W)	
	preds = sigmoid_activation(dot_product)

	# determine the 'error', the difference between predictions and the true values
	error = preds - trainY
	# compute the least square error
	# least square error is a simple loss function for binary classification
	loss = np.sum(error ** 2)
	# append to list of losses for visualization
	losses.append(loss)

	# the gradient descent update is the dot product transpose of features and the error of the predictions
	gradient = trainX.T.dot(error)
	# nudge the weight matrix in the negative direction of the gradient
	# hence the 'gradient descent' by taking a small step towards a set of more optimal parameters
	W = W + -args["alpha"] * gradient

	# check to see if an update should be displayed
	if epoch == 0 or (epoch + 1) % 5 == 0:
		print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] Evaluating...")
# predict testX with the learned parameters (weights)
preds = predict(testX, W)
# display the prediction result and evaluation metrics
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker = "o", s = 30)

# construct a figure that plots the loss over time
# this plot validates whether weight matrix is being updated in a manner that allows
# the classifier to learn from the training data
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

'''
def evaluate_gradient(loss, data, W):
	# loss: function used to compute the loss over current parameters W and input data
	# data: training data where each training sample is represented by an image or feature vector
	# W: weight matrix to optimize to find a W that minimizes loss
	
	# Wgradient is K-dimensional where K is the number of dimensions in feature vector
	# Wgradient contains the gradient entry for each dimension
	Wgradient = None
	return Wgradient

# loop until either of the condition is met:
# a specificied number of epochs has passed (our learning algorithm has seen each of the 
# training data points N times)
# loss has become sufficiently low
# training accuracy has become satisfactory high
# loss has not improved in M subsequent epochs
while True:
	Wgradient = evaluate_gradient(loss, data, W)

	# apply gradient descent
	# alpha: learning rate that controls the size of the step
	W = W + -alpha * Wgradient
'''