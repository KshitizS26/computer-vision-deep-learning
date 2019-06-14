# import the required packages
# scikit-image library to implement custom convolution function
from skimage.exposure import rescale_intensity
# numpy for standard numerical array processing
import numpy as np
import argparse
# cv2 for computer vision functions
import cv2

def convolve(image, K):
	# grab the spatial dimension of the image and kernel
	# image: grayscale input image
	# K: kernel to convolve on the input
	(iH, iW) = image.shape[:2]
	(kH, kW) = K.shape[:2]

	# allocate memory for the output image, taking care to "pad"
	# the borders of the input image so the spatial size (i.e.,
	# width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype = "float")
	