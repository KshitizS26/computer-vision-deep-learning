# import the required packages
import numpy as np

class Perceptron:
	def __init__(self, N, alpha = 0.1):
		# initialize the weight matrix and store the learning rate
		# N: number of columns in our input feature vectors
		# alpha: optional, learning rate for perceptron algorithm
		
		# create weight matrix W with random values samples from a normal distribution 
		# with zero mean and unit variance
		# the weight matrix has N + 1 entries, one for each of the N inputs in the
		# feature vector and the bias
		# divide W by the square-root of the number of inputs to scale W, leading to faster
		# convergence
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha

	def step(self, x):
		# apply the step function
		return 1 if x > 0 else 0

	def fit(self, X, y, epochs = 10):
		# train the perceptron
		# X: actual training data
		# y: target output class label
		# epochs: the number of epochs perceptron will train for

		# insert a column of 1's as the last entry in the feature matrix 
		# bias trick allows to treat the bias as a trainable parameter within the weight
		# matrix
		X = np.c_[X, np.ones((X.shape[0]))]

		# loop over the desired number of epochs
		for epoch in np.arange(0, epochs):
			# loop over each individual data point
			for (x, target) in zip(X, y):
				# take the dot product between the input features
				# and the weight matrix, then pass this value through
				# the step function to obtain the prediction
				dot_product = np.dot(x, self.W)
				p = self.step(dot_product)

				# only perform a weight update if our prediction 
				# does not match the target
				if p != target:
					# determine the error by computing the sign (either positive or negative)
					# via difference operation
					error = p - target
					# update the weight matrix by taking a step towards the correct classification,
					# scaling by learning rate
					# over a series of epochs, the perceptron is able to learn patterns in the
					# underlying data and shift the values of weight matrix to correctly classify x
					self.W = self.W + -self.alpha * error * x

	def predict(self, X, addBias = True):
		# predict the class labels for a given set of input data X
		# X: input data to predict
		# addBias: optional, flag to decide whether to add bias in the input data X

		# ensure input is a matrix
		X = np.atleast_2d(X)

		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1's as the last entry in the feature matrix (bias)
			X = np.c_[X, np.ones((X.shape[0]))]

		# take the dot product between the input features and the weigth matrix
		# pass the value through the step function
		prediction = self.step(np.dot(X, self.W))
		return prediction













