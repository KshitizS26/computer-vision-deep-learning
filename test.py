import numpy as np
# constant weight initialization
W = np.zeros((64, 32))
print(W)
W = np.ones((64, 32))
print(W)
C = 5
W = np.ones((64, 32)) * C
print(W)

# uniform and normal distribution
W = np.random.uniform(low = -0.05, high = 0.05, size = (64, 32))
print(W)
W = np.random.normal(0.0, 0.5, size = (64, 32))
print(W)

# LeCun uniform and normal weight initialization
F_in = 64
F_out = 32
limit = np.sqrt(3/float(F_in))
W = np.random.uniform(low = -limit, high = limit, size = (F_in, F_out))
print(W)

F_in = 64
F_out = 32
limit = np.sqrt(1/float(F_in))
W = np.random.normal(0.0, limit, size = (F_in, F_out))

# Glorot/Xavier uniform and normal
F_in = 64
F_out = 32
limit = np.sqrt(2 / float(F_in + F_out))
W = np.random.normal(0.0, limit, size = (F_in, F_out))

F_in = 64
F_out = 32
limit = np.sqrt(6 / float(F_in + F_out))
W = np.random.uniform(low = -limit, high = limit, size = (F_in, F_out))

# He/Kaiming/MSRA uniform and normal weight initialization
F_in = 64
F_out = 32
limit = np.sqrt(6/float(F_in))
W = np.random.uniform(low = -limit, high = limit, size = (F_in, F_out))
print(W)

F_in = 64
F_out = 32
limit = np.sqrt(2/float(F_in))
W = np.random.normal(0.0, limit, size = (F_in, F_out))
print(W)


