'''
knn.py accepts path of the dataset, k-NN hyperparameters through command line argument,
loads and preprocesses images in the dataset, encodes label, fits a k-NN model on the dataset,
and evaluates the classifier performance

usage: python knn.py [-h] -d DATASET [-k NEIGHBORS] [-j JOBS] 
'''

# import the required package
# KNeighborsClassifier contains the implementation of the k-NN algorithm
# LabelEncoder converts labels represented as strings to integers 
# train_test_split creates the training and testing split
# classification_report helps evaluate the classifier performance through a table
# paths - grabs the file path to all images in the dataset
# argparse - accepting command line arguments
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# construct the argument parse and parse the arguments
# --jobs - optional, the number of concurrent jobs to run when computing the distance between
# an input data point and the training set
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to input dataset")
ap.add_argument("-k", "--neighbors", type = int, default = 1, help = "# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type = int, default = -1, help = "# of concurrent jobs to k-NN distance (-1 uses all available cores")
args = vars(ap.parse_args())

# grab the file path of images in our dataset
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix

sp = SimplePreprocessor(32, 32)

# supplying instantiated SimplePreprocessor in a list as an argument, 
# implying that sp will be applied to every image in the dataset
sdl = SimpleDatasetLoader(preprocessors = [sp])
# load() returns a tuple containing input images and their corresponding labels
(data, labels) = sdl.load(imagePaths, verbose = 500)

# flatten the image: 32 x 32 x 3 = 3072
# new shape: [3000, 3072]
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
# compute the number of bytes the array consumes and convert it to MB
print("[INFO] Features matrix: {:.1f} MB".format(data.nbytes / (1024 * 1000.0)))

# encode the labels as integers; one unique integer per class
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing

trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.25, random_state = 42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] Evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors = args["neighbors"], n_jobs = args["jobs"])
model.fit(trainX, trainY)
# classification_report returns a table containing evaluation metrics
# target_names - opti onal, names of the class labels
print(classification_report(testY, model.predict(testX), target_names = le.classes_))

