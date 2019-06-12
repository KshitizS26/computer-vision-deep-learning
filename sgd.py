# import the required packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
	# compute the sigmoid activation value for a given input
	sigmoid = 1.0 / (1 + np.exp(-x))
	return sigmoid

def predict(X, W):
	# take the dot product between our features and weight matrix
	preds = sigmoid_activation(X.dot(W))

	# apply a step function to threshold the outputs to binary class labels
	preds[preds <= 0.5] = 0
	preds[preds > 0] = 1

	# return the predictions
	return preds

def next_batch(X, y, batchSize):
	# loop over our dataset 'X' in mini-batches, yielding a tuple of the current batched data and labels
	# X: training dataset of feature vectors/raw image pixel intensities
	# y: class labels associated with each of the training data points
	# batchSize: size of each mini-batch to return

	for i in np.arange(0, X.shape[0], batchSize):
		yield (X[i:i+batchSize], y[i:i+batchSize])


# construct the argument parse and parse the arguments
# --batch-size: size of mini-batches
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type = float, default = 100, help = "# of epochs")
ap.add_argument("-a", "--alpha", type = float, default = 0.01, help = "learning rate")
ap.add_argument("-b", "--batch-size", type = int, default = 32, help = "size of SGD mini-batches")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1000 data points, 
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 1.5, random_state = 1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1's as the last entry in the feature matrix -- bias trick allows to treat
# the bias as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of the
# data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size = 0.5, random_state = 42)

# initialize weight matrix and list of losses

print("[INFO] Training...")
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
	# initialize the total loss for the epochs
	epochLoss = []

	# loop over our data in batches
	for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
		# take the dot product between our current batch of features 
		# and the weight matrix, then pass this value through activation function
		preds = sigmoid_activation(batchX.dot(W))

		# determine the error
		error = preds - batchY
		epochLoss.append(np.sum(error ** 2))

		# the gradient descent update is the dot product between our
		# current batch and the error on the batch
		gradient = batchX.T.dot(error)

		# nudge the weight matrix in the negative direction of the gradient
		# hence the gradient descent by taking a small step towards
		# a set of more optimal parameters
		W = W + -args["alpha"] * gradient

	# update loss history by taking the average loss across all batches
	loss = np.average(epochLoss)
	losses.append(loss)

	# check to see if an update should be displayed
	if epoch == 0 or (epoch + 1) % 5 == 0:
		print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch + 1), loss))

# evaluate model
print("[INFO] Evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker = "o", s = 30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

'''
while True:
	# yield batches by randomly sampling training data before evaluating the gradient
	# typing batch sizes: 32, 64, 128, 256
	batch = next_training_batch(data, 256)
	# evaluate gradient of the batch
	Wgradient = evaluate_gradient(loss, batch, W)
	# update weight matrix
	W = W + -alpha * Wgradient
'''