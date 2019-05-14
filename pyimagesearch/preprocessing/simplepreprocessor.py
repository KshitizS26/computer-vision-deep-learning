# import the necessary packages
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter = cv2.INTER_AREA):
		'''
		store the target image width, height, and interpolation
		method used when resizing

		width: the target width of our input image after resizing
		height: the target height of our input image after resizing
		inter: optional parameter to select interpolation algorithm for resizing
		'''
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		'''
		resize the image to a fixed size, ignoring the aspect ratio

		image: input image to preprocess
		'''
		return cv2.resize(image, (self.width, self.height), interpolation = self.inter)


