# import the necessary packages
import numpy as np
import cv2

# initialize the class labels and set the seed of the pseudorandom
# number generator so we can reproduce our results
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# randomly initialize our weight matrix and bias vecor -- in a
# real training and classification task, these parameters would 
# be leared by our model, but for the sake of this example, 
# let us use random values

# the weight matrix W initializes random values from a uniform 
# distribution and sampled over the range [0, 1]
W = np.random.randn(3, 3072)

# the bias vector is randomly filled with values uniformly samples
# over the distribution [0, 1]
b  = np.random.randn(3)


# load our example image, resize it, and then flatten it into our
# "feature vector" representation
orig = cv2.imread("datasets/animals/dogs/dogs_00001.jpg")

# resizing ignoring the aspect ratio and flattening
image = cv2.resize(orig, (32, 32)).flatten()

# compute the output scores by taking the dot product between the
# weight matrix and image pixels, followed by adding in the bias
scores = W.dot(image) + b

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
	print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score on the image  as our
# prediction
# cv2.putText accepts image, label, (x, y) coordinate of the label, 
# algorithm to write on the image, size of the text, RGB, and thickness
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

# display our input image
cv2.imshow("Image", orig)
cv2.waitKey(0)






