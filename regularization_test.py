import numpy as np

W = np.array(
	[[2, 3, 4, 5], 
	[5, -2, 1, 6], 
	[6, 7, 2, -4]])

# L2 regularization

penalty = 0

for i in np.arange(0, W.shape[0]):
	for j in np.arange(0, W.shape[1]):
		penalty = penalty + (W[i][j] ** 2)

print(penalty)

# L1 regularization

penalty = 0

for i in np.arange(0, W.shape[0]):
	for j in np.arange(0, W.shape[1]):
		penalty = penalty + abs(W[i][j])

print(penalty)