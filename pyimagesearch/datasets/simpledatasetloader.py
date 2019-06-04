'''
simpledatasetloader.py contains SimpleDatasetLoader class that accepts path of the
input images and applies image preprocessors upon them sequentially

image preprocessors should be specified as a list rather than a single value so that
they could be applied independently and sequentially to an image in an efficient manner

usage: binded with knn.py
'''

# import the required packages
# numpy - NumPy for numerical processing
# cv2 - OpenCV binding
# os - for extracting the names of subdirectories in image paths
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors = None):
		'''
		define class constructor with an optional argument of list
		of image preprocessors that can be sequentially applied to a given
		input image
		'''
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose = -1):
		'''
		load images from the dataset and applies preprocessing on them

		imagePaths: list specifying the file paths to the images in our dataset
		verbose: "verbosity level" print updates to a console, allows to monitor how many images the SimpleDatasetLoader has processed
		'''

		# initializing data (images) list
		data = []
		# initializing labels (class labels for images) list
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):

			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/datasets/animals/{class}/{image}.jpg

			# {class} contains the name of class label

			# loading the image into the memory
			image = cv2.imread(imagePath)
			# fetching the class label at -2 index from image path
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:

				# loop over the preprocessors and apply each preprocessor sequentially to the image
				for p in self.preprocessors:
					image = p.preprocess(image)

					# treat the processed image as a "feature vector"
					# by updating the data list following by the labels
			
			data.append(image)
			labels.append(label)

			# convert input images and labels list to NumPy array
			# data = np.array(data)
			# labels = np.array(labels)

			# show an update every 'verbose' images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] Proceed {}/{}".format(i + 1, len(imagePaths)))

		# print the shape of the input images and labels array
		print(np.array(data).shape)
		print(np.array(labels).shape)

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))

