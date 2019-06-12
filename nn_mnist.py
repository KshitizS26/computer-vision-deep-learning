# import the required packages
from pyimagesearch.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# scikit-learn helper function to load datasets
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# load the MNIST dataset and apply min/max scaling to scale
# the pixel intensity values to the range [0, 1] (each image is
# represented by an 8 x 8 = 64-dim feature vector)
print("[INFO] Loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size = 0.25)

# convert the labels from integers to vectors - one-hot encoding
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
print("[INFO] Training network...")
# training a neural network with 64-32-16-10 architecture
# output layer has 10 nodes because there are 10 possible output classes for the digits 0-9
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
epoch_losses = nn.fit(trainX, trainY, epochs = 1)

# evaluate the network
print("[INFO] Evaluating network...")
# compute the output predictions for every data point in testX
predictions = nn.predict(testX)
# find the class label with the largest probability for each data point
# return the index of the label with the highest predicted probability
# index of the label with the highest predicted probability = class label
predictions = predictions.argmax(axis = 1)
print(classification_report(testY.argmax(axis = 1), predictions))

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 1100, 100), epoch_losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()