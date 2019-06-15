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

	# replicate the pixels along the border (left, up, right, down) of the image
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

	# application of padding allows equal dimension of output image and
	# original input image
	output = np.zeros((iH, iW), dtype = "float")
	
	# loop over the input image, "sliding" the kernel across
	# each (x, y) - coordinate from left-to-right and top-to-bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI (region of interest) of the image by extracting the
			# center region of the current (x, y) - coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the actual convolution by taking the element-wise
			# multiplication between the ROI and the kernel, then summing
			# the matrix
			k = (roi * K).sum()

			# store the convolved value in the output (x, y) -
			# coordinate of the output image
			output[y - pad, x - pad] = k

	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range = (0, 255))
	output = (output * 255).astype("uint8")

	# return the outpage image
	return output

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth as image
smallBlur = np.ones((7,7), dtype = "float") * (1.0/(7 * 7))
largeBlur = np.ones((21,21), dtype = "float") * (1.0/(21 * 21))

# construct the sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype = "int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype = "int")

# construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype = "int")

# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype = "int")

# construct an emboss kernel
emboss = np.array((
	[-2, -1, 0],
	[-1, 1, 1],
	[0, 1, 2]), dtype = "int")

# construct the kernel bank, a list of kernels to apply
# using both custom 'convolve' function and OpenCV's filter2D
# function
"""kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur))"""

kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY),
	("embose", emboss))

# load the input image and convert it to grayscale
# convolutions can also be applied to RGB and other 
# multi-channel volume images
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels in the kernelBank
for (kernelName, K) in kernelBank:
	# apply the current kernel to the grayscale image using both custom
	# convolve function and OpenCV's filter2D function
	print("[INFO] Applying {} kernel".format(kernelName))
	convolveOutput = convolve(gray, K)
	# filter2D is OpenCV's implementation of convolve function
	opencvOutput = cv2.filter2D(gray, -1, K)

	# show the output images
	cv2.imshow("Original", gray)
	cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


