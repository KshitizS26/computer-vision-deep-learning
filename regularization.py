# import required packages
# SGDClassifier implementation encapsulates loss function, number of epochs, learning rate, and regularization term
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
args = vars(ap.parse_args())


# grab the list of image paths
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.reshape((data.shape[0], 3072))

# incode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)


# partition the data into training and testing splits using 75% of 
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 5)

# loop over set of regularizers
for r in (None, "l1", "l2"):
	# train a SGD classifier using a softmax loss function and the specificed
	# regularization function for 10 epochs
	print("[INFO] Training model with '{}' penalty".format(r))
	model = SGDClassifier(loss = "log", penalty = r, max_iter = 100, learning_rate = "constant", eta0 = 0.01, random_state = 42)
	model.fit(trainX, trainY)

	# evaluate the classifier
	# loss = "log" is cross-entropy loss
	# eta0 is learning rate
	# r is regularization penalty (None, l1, l2) with a default lambda of 0.0001
	acc = model.score(testX, testY)
	print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r, acc * 100))


