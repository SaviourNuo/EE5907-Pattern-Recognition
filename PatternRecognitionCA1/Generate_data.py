# Code used to generate 2D data following a normal distribution
# visualize it as a scatter plot with color-coded classes, then save the dataset
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

N1 = 300 # Number of samples in class 0
N2 = 80 # Number of samples in class 1
K = 2 # Feature dimension
sigma = 2 # Data dispersion

# Generate corresponding data belonging to class 0
mean1 = (10, 14)
cov1 = np.array([[sigma, 0], [0, sigma]])
X1 = np.random.multivariate_normal(mean1, cov1, N1)
c1 = ['red'] * len(X1)

# Generate corresponding data belonging to class 1
mean2 = (14, 18)
cov2 = np.array([[sigma, 0], [0, sigma]])
X2 = np.random.multivariate_normal(mean2, cov2, N2)
c2 = ['blue'] * len(X2)

# Data concatenation
X = np.concatenate((X1, X2))
color = np.concatenate((c1, c2))

# Define the class labels for generated data points.
T = []
for n in range(0, len(X)):
    if n < len(X1):
        T.append(0)
    else:
        T.append(1)

# Plot X1 and X2
plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = color)
plt.show()

# Save the generated data
np.save('class1.npy', X1)
np.save('class2.npy', X2)
io.savemat('class1.mat', {'class1': X1})
io.savemat('class2.mat', {'class2': X2})