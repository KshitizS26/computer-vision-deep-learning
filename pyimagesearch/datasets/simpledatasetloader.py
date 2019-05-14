# import the necessary packages
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors = None):
		'''
		store the image preprocessor
		'''
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an empty list
		
		# specify preprocessors as list to allow for sequential preprocessing
		# independently on images

		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose = -1):
		'''
		initialize the list of features and labels

		imagePaths: list specifying the file paths to the images in our dataset
		verbose: "verbosity level" print updates to a console, allows to monitor how many images the SimpleDatasetLoader has processed
		'''
		data = []
		labels = []

		# loop over the input image
		for (i, imagePath) in enumerate(imagePaths):

			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg

			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:

				# loop over the preprocessors and apply each to the image
				for p in self.preprocessors:
					image = p.preprocess(image)

				# treat the processed image as a "feature vector"
				# by updating the data list following by the labels
				data.append(image)
				labels.append(label)

		print(np.array(data).shape)
		print(np.array(labels).shape)

		# show an update every 'verbose' images
		if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
			print("[INFO] Proceed {}/{}".format(i + 1, len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))

