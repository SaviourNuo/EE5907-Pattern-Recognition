# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Set the seed of the NumPy random number generator to ensure reproducibility of results
np.random.seed(40)

# Load the pre-generated data and create corresponding labels.
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')
X = np.concatenate((X1, X2))
Y = np.concatenate((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))

num_centers = 3 # Number of RBF centers
random_indices = np.random.choice(len(X), num_centers, replace = False) # Randomly select corresponding indices from all X as RBF centers without replacement
rbf_centers = X[random_indices]
sigma = 1  # The Sigma parameter of the Gaussian RBF kernel

# Define Gaussian RBF function
def rbf_function(x, rbf_center, sig):
    return np.exp(-np.linalg.norm(x - rbf_center, axis = 1) ** 2 / (2 * sig ** 2))

# Calculate the RBF values for each data point to all RBF centers
H = np.zeros((len(X), num_centers)) # Create the hidden layer output matrix of the RBF network with shape (380, 3)
for i in range(num_centers):
    H[:, i] = rbf_function(X, rbf_centers[i], sigma)

# Calculate the weight matrix using the standard solution of Least Squares Estimation
# Use pseudoinverse matrix to increase robustness
W = np.linalg.pinv(H.T @ H) @ H.T @ Y

# Print RBF centers
print("RBF centers:")
for i, center in enumerate(rbf_centers):
    print(f"center {i + 1}: {center}")

# Data classification
Y_pred = H @ W
Y_pred_class = (Y_pred > 0.5).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(Y, Y_pred_class)
precision = precision_score(Y, Y_pred_class)
recall = recall_score(Y, Y_pred_class)

# Generate a grid of coordinates based on the two features X1 and X2
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
res = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Calculate the RBF network's predictions for the grid points
H_grid = np.zeros((len(grid_points), num_centers))
for i in range(num_centers):
    H_grid[:, i] = rbf_function(grid_points, rbf_centers[i], sigma)
Z = H_grid @ W
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, levels = np.linspace(Z.min(), Z.max(), 50), cmap = "coolwarm_r", alpha = 0.3)
plt.contour(xx, yy, Z, levels = [0.5], colors = 'k', linestyles = '-', linewidths = 2)
plt.scatter(X1[:, 0], X1[:, 1], color = 'red', label = "Class 1")
plt.scatter(X2[:, 0], X2[:, 1], color = 'blue', label = "Class 2")

for i, center in enumerate(rbf_centers):
    plt.scatter(center[0], center[1], color = 'black', marker = 'x', s = 100) # Plot RBF centers
    circle = plt.Circle((center[0], center[1]), sigma * 2, color = 'black', fill = False, linestyle = '--', alpha = 0.5)
    # Draw the influence range of each RBF center with a dashed circle, radius 2 * sigma
    plt.gca().add_patch(circle)
    plt.text(center[0] + 0.1, center[1] + 0.1, f"w = {W[i]:.4f}", fontsize = 12) # Annotate weight values

plt.legend()
plt.title(
    f"RBF Network Decision Boundary (3 Centers)\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
