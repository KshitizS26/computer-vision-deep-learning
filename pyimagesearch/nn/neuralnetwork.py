# import required packages
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

class NeuralNetwork:
	def __init__(self, layers, alpha = 0.1):
		# initialize the list of weight matrices, then store the
		# network architecture and learning rate
		
		# layers: list of integers which represents the actual architecture
		# of the feedforward network
		# alpha: learning rate of neural network, applied during the
		# weight update phase

		self.W = []
		self.layers = layers
		self.alpha = alpha

		# start looping from index of the first layer but 
		# stop before we reach the last two layers
		for i in np.arange(0, len(layers) - 2):
			# randomly initialize a weight matrix from  a standard, normal distribution
			# connecting the number of nodes in each respective layer together,
			# adding an extra node for the bias

			# shape of w: [number of dimensions x number of output nodes]
			w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
			# scale w by dividing by the square root of the number of nodes
			# in the current layer, thereby normalizing the variance of each
			# neuron's output
			self.W.append(w/np.sqrt(layers[i]))

			# the last two layers are a special case where the input connections
			# need a bias term but the output does not

			# shape of w: [number of dimensions X number of output nodes]
			w = np.random.randn(layers[-2] + 1, layers[-1])
			self.W.append(w/np.sqrt(layers[-2]))

	def __repr__(self):
		# magic method
		# construct and return a string that represents the network architecture
		net_architecture = "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
		return net_architecture

	def sigmoid(self, x):
		# compute and return the sigmoid activation value for a
		# given input value
		sig = 1.0/(1 + np.exp(-x))
		return sig

	def sigmoid_deriv(self, x):
		# compute the derivative of the sigmoid function 
		# assuming that x has already been passed through the
		# sigmoid function
		sig_deriv = x * (1 - x)
		return sig_deriv

	def fit_partial(self, x, y):
		# construct list of output activations for each layer as data points
		# flow through the network; the first activation is a special case -- it is
		# just the input feature vector itself
		# x: individual data point from design matrix
		# y: corresponding class label

		# A is a list that stores the output activations for each layer as data point x
		# forward propagates through the network
		A = [np.atleast_2d(x)]

		# FEEDFORWARD:
		# loop over the layers in the network
		for layer in np.arange(0, len(self.W)):
			# feedforward the activation at the current layer by
			# taking the dot product between the activation and
			# the weight matrix -- this is called the "net input"
			# to the current layer
			net = A[layer].dot(self.W[layer])

			# computing the "net output" is simply applying non-linear
			# activation function to the net input
			out = self.sigmoid(net)
			out = np.round(out, 3)

			# add net output to list of activations
			A.append(out)

		# final entry in A is the output of the last layer in the network (i.e., prediction)
		# print(A)

		# BACKPROPAGATION:
		# the first phase of backpropagation is to compute the difference
		# between prediction (the final output activation in the activation list) 
		# and the true target value
		error = A[-1] - y
		#print("[INFO] Printing error...")
		#print(error)

		# apply the chain rule and build list of deltas 'D'; the first entry in the deltas
		# is simply the error of the output layer times the derivation of
		# activation function for the output value
		D = [error * self.sigmoid_deriv(A[-1])]
		#print("[INFO] Printing delta of output layer...")
		#print(D)

		# loop over the layers in reverse order (ignoring the last two since we already have takem into account)
		for layer in np.arange(len(A) - 2, 0, -1):
			# the delta for the current layer is equal to the delta 
			# of the previous layer dotted with weight matrix of the current layer, following by
			# multiplying the delta by the derivative of the non-linear activation
			# function for the activations of the current layer
			delta = D[-1].dot(self.W[layer].T)
			#print("[INFO] Printing delta of previous layer...")
			#print(delta)
			#print(self.sigmoid_deriv(A[layer]))
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)

		# during the backpropagation phase, loop was run in a reverse order thus to perform weight update
		# reversing the order of entries in D is required
		D = D[::-1]

		# WEIGHT UPDATE PHASE
		# loop over the layers from 0 to N
		for layer in np.arange(0, len(self.W)):
			# update weights by taking the dot product of the layer activations
			# with their respective deltas, then multiplying this value by some small
			# learning rate and adding to weight matrix -- this is where the
			# actual learning takes place
			self.W[layer] = self.W[layer] + -self.alpha * A[layer].T.dot(D[layer])

	def predict(self, X, addBias = True):
		# initialize the output prediction as the input features -- this
		# value will be (forward) propagated through the network to obtain
		# the final prediction
		# X: the data points to predict the class for
		# addBias: boolean flag to check if addig a bias column is required
		p = np.atleast_2d(X)

		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1's as the last entry in the featur matrix (bias)
			p = np.c_[p, np.ones((p.shape[0]))]

		# loop over layers in the network
		for layer in np.arange(0, len(self.W)):
			# computing the output prediction is as simple as taking
			# the dot product between the current activation value 'p'
			# and the weight matrix associated with the current layer,
			# then passing this value through a non-linear activation function
			p = self.sigmoid(np.dot(p, self.W[layer]))

		# return the predicted value
		return p

	def calculate_loss(self, X, targets):
		# make predictions for the input data points and then compute the loss
		# X: input data points
		# targets: ground-truth labels
		targets = np.atleast_2d(targets)
		# predict the class
		predictions = self.predict(X, addBias = False)
		# compute the squred sum error
		loss = 0.5 * np.sum((predictions - targets) ** 2)

		# return the loss
		return loss

	def fit(self, X, y, epochs = 100, displayUpdate = 100):
		# insert a column of 1's as the last entry in the feature
		# matrix -- bias trick allows to treat the bias 
		# as a trainable parameter within the weight matrix
		# X: training data
		# y: class labels for each entry in X
		# epochs: number of epochs to train the network
		# displayUpdate: controls how many N epochs to print training progress to terminal
		X = np.c_[X, np.ones((X.shape[0]))]

		# list of losses
		epoch_losses = []

		# loop over the desired number of epochs
		for epoch in np.arange(0, epochs):
			# loop over each individual data point and train 
			# network on it - forward and backward propagation, update weight matrix
			for (x, target) in zip(X, y):
				self.fit_partial(x, target)
			
			# check to see if training update display is required
			if epoch == 0 or (epoch + 1) % displayUpdate == 0:
				loss = self.calculate_loss(X, y)
				print("[INFO] epoch = {}, loss = {:.7f}".format(epoch + 1, loss))
				epoch_losses.append(loss)

		return epoch_losses
		
# define 2-2-1 neural network and train it
# instantiate a NeuralNetwork to have a 2-2-1 architecture --
# 2 input nodes, single hidden layer with 2 nodes, 1 output node
nn = NeuralNetwork(layers = [2,2,1], alpha = 0.5)
print(nn)

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

epoch_losses = nn.fit(X, y, epochs = 20000)

# after training network, loop over the XOR data points
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = nn.predict(x)[0][0]
	# apply a step function to binarize output class labels
	step = 1 if pred > 0.5 else 0
	print("[INFO] data = {}, ground-truth = {}, pred = {:.4f}, step = {}".format(x, target[0], pred, step))

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20100, 100), epoch_losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()