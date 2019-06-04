'''
simplepreprocessor.py contains SimplePreprocessor class that accepts an input 
image, resizes it to a fixed dimension, and then returns it.

usage: binded with knn.py
'''

# import the required packages
# cv2 - OpenCV binding
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter = cv2.INTER_AREA):
		'''
		define class constructor requiring two arguments and an optional argument

		height: the target height of our input image after resizing
		inter: optional parameter to select interpolation algorithm for resizing
		'''
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		'''
		resize the input image to a fixed size, ignoring the aspect ratio

		image: input image to preprocess
		'''
		resized_image = cv2.resize(image, (self.width, self.height), interpolation = self.inter)
		return resized_image


