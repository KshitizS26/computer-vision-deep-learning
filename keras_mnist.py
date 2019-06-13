# import the required packages
import warnings
warnings.filterwarnings('ignore')
# LabelBinarizer one-hot encodes integer labels as vector labels
from sklearn.preprocessing import LabelBinarizer
# classification_report returns a nicely formatted report displaying
# the total accuracy of our model, along with a breakdown on the 
# classification accuracy for each digit
from sklearn.metrics import classification_report
# train_test_split creates training and test splits from the dataset
from sklearn.model_selection import train_test_split
# Sequential indicates that the network is feedforward and layers will be
# sequentially added on top of the other
from keras.models import Sequential
# Dense contain implementation of fully-connected layers
from keras.layers.core import Dense
# SGD contain implementation of stochastic gradient descent
from keras.optimizers import SGD
# helper function to automatically load datasets from the disk
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# --output switch contain the path to where plot of loss and accuracy over time
# will be saved to disk
ap.add_argument("-o", "--output", required = True, help = "Path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the MNIST dataset
# MNIST dataset will be downloaded and stored locally on the disk
# once the dataset is downloaded, it is cached to the machine and will not
# have to be downloaded again
print("[INFO] Loading MNIST (full) dataset...")

# data normalizaton: scale the raw pixel intensities to the range [0, 1.0], then 
# construct the training and test splits
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], 784)
testX = testX.reshape(testX.shape[0], 784)
trainY = trainY.reshape(trainY.shape[0], 1)
testY = testY.reshape(testY.shape[0], 1)
print(trainX.shape, testX.shape)
print(trainY.shape, testY.shape)
trainX = trainX.astype("float")
trainX = trainX / 255.0
testX = testX.astype("float")
testX = testX / 255.0

# convert the labels from integers to vectors
# the index in the vector for label is set to 1 and 0 otherwise - one-hot encoding
# for example: label 3 - [0,0,0,1,0,0,0,0,0,0]
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# define the 784-256-128-10 architecture (feedforward architecture) using Keras
# this architecture implies that the layers will be stacked on top of each other 
# with the output of the previous layer feeding into the next
# instantiate the architecture using Sequential class
model = Sequential()
# first fully connected layer: (input_layer - hidden_layer1)
# input_shape = (784,) or (60000,784)
# shape of the weight matrix: (784, 256)
# number of weights to learn: 256
# activation function applied: sigmoid
model.add(Dense(256, input_shape = (784,), activation = "sigmoid"))
# second fully connected layer: (hidden_layer1 - hidden_layer2)
# input_shape = (256,) or (60000, 256)
# shape of the weight matrix: (256, 128)
# number of weights to learn: 256
# activation function applied: sigmoid
model.add(Dense(128, activation = "sigmoid"))
# third fully connected layer: (hidden_layer2 - output_layer)
# input_shape = (128,) or (60000, 128)
# shape of the weight matrix: (128, 10)
# number of weights to learn: 10
# activation function applied: softmax
# softmax activation obtain normalied class probabilities for each prediction
model.add(Dense(10, activation = "softmax"))
model.summary()

# train the model using SGD
print("[INFO] Training network...")
# initialize the SGD optimizer with a learning rate = 0.01 (1e-2)
sgd = SGD(0.01)
# categorical_crossentropy loss demands that class labels are vectors not integers
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])
# in most cases such as tuning hyperparameters or deciding on a model architecture, validation set
# should not be same as testing set
# it should be a true validation set
# fit returns a dictionary H which will be used to plot loss/accuracy of the network overtime
H = model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 100, batch_size = 128)

# evaluate the network
print("[INFO] Evaluating network...")
# return the class label probabilities for every data point in testX
# shape of predictions: (17500, 10)
# each entry in a given row is a probability 
# argmax(axis = 1) return the index that determine the class with the largest probability
predictions = model.predict(testX, batch_size = 128)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = [str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# save plot to the disk based on the --output command line argument
plt.savefig(args["output"])


