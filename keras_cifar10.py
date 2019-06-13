# import the required packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
# helper function to automatically load dataset from the disk
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "Path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load the CIFAR-10 dataset from disk which is pre-segmented into training and testing split, 
# scale it into the range [0, 1], then reshape the design matrix
print("[INFO] Loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# define the 3072-1024-512-10 architecture using Keras
model = Sequential()
# first fully connected layer: (input_layer - hidden_layer1)
# input_shape = (3072,) or (50000,3072)
# shape of the weight matrix: (3072, 1024)
# number of weights to learn: 1024
# activation function applied: relu
model.add(Dense(1024, input_shape = (3072,), activation = "relu"))
# second fully connected layer: (hidden_layer1 - hidden_layer2)
# input_shape = (1024,) or (50000, 1024)
# shape of the weight matrix: (1024, 512)
# number of weights to learn: 512
# activation function applied: relu
model.add(Dense(512, activation = "relu"))
# third fully connected layer: (hidden_layer2 - output_layer)
# input_shape = (512,) or (50000, 512)
# shape of the weight matrix: (512, 10)
# number of weights to learn: 10
# activation function applied: softmax
# softmax activation obtain normalied class probabilities for each prediction
model.add(Dense(10, activation = "softmax"))

# train the model using SGD
print("[INFO] Training network...")
sgd = SGD(0.01)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])
H = model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 100, batch_size = 32)
model.summary()

# evaluate the network
print("[INFO] Evaluating network...")
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = labelNames))

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
plt.savefig(args["output"])




