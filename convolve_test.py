import numpy as np
from skimage.exposure import rescale_intensity
import cv2

input_image = cv2.imread("dog.jpg")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

'''input_image = np.array((
	[22, 56, 80, 2, 255],
	[126, 10, 15, 20, 100],
	[64, 60, 0, 255, 10],
	[0, 1, 255, 20, 10],
	[10, 20, 10, 0, 200]))'''

(iH, iW) = input_image.shape[:2]
# print(iH, iW)

kernel = np.array((
	[0, 1, 2],
	[-2, 0, 1],
	[1, 2, 0]))

(kH, kW) = kernel.shape[:2]
# print(kH, kW)

pad = (kW - 1) // 2
# print(pad)

input_image = cv2.copyMakeBorder(input_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
# print(input_image.shape)

output = np.zeros((iH, iW), dtype = "float")
# print(output_image.shape)

# print(pad, iH + pad)
# print(pad, iW + pad)
K = []

for y in np.arange(pad, iH + pad):
	
	for x in np.arange(pad, iW + pad):

		roi = input_image[y - pad:y + pad + 1, x - pad:x + pad + 1]
		
		k = (roi * kernel).sum()

		K.append(k)

		output[y - pad, x - pad] = k

output = rescale_intensity(output, in_range = (0, 255))

output = (output * 255).astype("uint8")

print(output)









