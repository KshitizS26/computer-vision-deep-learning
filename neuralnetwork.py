# import the required packages
from pyimagesearch.nn.neuralnetwork import NeuralNetwork
import matplotlib.pyplot as plt
		
# define 2-2-1 neural network and train it
# instantiate a NeuralNetwork to have a 2-2-1 architecture --
# 2 input nodes, single hidden layer with 2 nodes, 1 output node
nn = NeuralNetwork(layers = [2,2,1], alpha = 0.5)
print(nn)

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

epoch_losses = nn.fit(X, y, epochs = 1)

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