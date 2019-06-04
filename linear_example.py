# import the required packages
# numpy - NumPy for numerical processing
# cv2 - OpenCV binding
import numpy as np
import cv2

# initialize the class labels for Animals dataset

labels = ["dog", "cat", "panda"]

# set the pseudorandom number generator to reproduce results
# this is a worked example, random seed of 1 will generate such
# W and b values that would lead to correct classification
np.random.seed(1)

# randomly initialize the weight matrix and bias vecor -- in a
# real training and classification task, these parameters would 
# be leared by the model via an optimization algorithm

# the weight matrix W initializes random values from a uniform 
# distribution and sampled over the range [0, 1]
# shape of W: [3 x 3072]
W = np.random.randn(3, 3072)

# the bias vector is randomly filled with values uniformly samples
# over the distribution [0, 1]
# shape of b: [3 x 1]
b  = np.random.randn(3)


# load example image
orig = cv2.imread("datasets/animals/dogs/dogs_00001.jpg")

# resize ignoring the aspect ratio to [32 x 32 x 3]
image = cv2.resize(orig, (32, 32))
# flatten to [3072 x 1] which is a feature vector representation
image = image.flatten()

# compute the output scores by taking the dot product between the
# weight matrix and image pixels, followed by adding in the bias
scores = W.dot(image) + b

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
	print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score on the image as prediction results
# cv2.putText accepts image, label (in this case, highest score of prediction), (x, y) coordinate of the label, 
# algorithm to write on the image, size of the text, RGB, and thickness
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# display our input image
cv2.imshow("Image", orig)
cv2.waitKey(0)







