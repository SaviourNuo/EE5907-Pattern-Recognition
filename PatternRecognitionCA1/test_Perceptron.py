# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

# Define an MLP class to implement a neural network for binary classification task
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden = nn.Linear(2,3) # Two input neurons/One hidden layer with three neurons
        self.out = nn.Linear(3,1) # One output neuron
        self.relu = nn.ReLU() # Use ReLU as the activation function after the hidden layer
        self.sigmoid = nn.Sigmoid() # Use Sigmoid as the activation function following the output layer

        nn.init.normal_(self.hidden.weight, mean = 0.0, std = 0.1) # Initialize weights following a normal distribution
        nn.init.normal_(self.out.weight, mean = 0.0, std = 0.1)
        nn.init.zeros_(self.hidden.bias) # Set initial biases to zero
        nn.init.zeros_(self.out.bias)

    def forward(self,x):
        x = self.relu(self.hidden(x)) # Forward propagation of the hidden layer
        x = self.sigmoid(self.out(x)) # Forward propagation of the output layer
        return x

# Instantiate the MLP
mlp = MLP()

# Load the pre-generated data and create corresponding labels.
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')
X_concat = np.concatenate((X1, X2))
Y_concat = np.concatenate((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))

# Generate a grid of coordinates based on the two features X1 and X2
x_min, x_max = X_concat[:, 0].min() - 1, X_concat[:, 0].max() + 1
y_min, y_max = X_concat[:, 1].min() - 1, X_concat[:, 1].max() + 1
res = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
grid_tensor = torch.tensor(grid_points)

# Use the untrained MLP model to infer the classification results of the grid coordinates
with torch.no_grad(): # Disable gradient calculation
    Z = mlp(grid_tensor).numpy()
    Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, levels = [0, 0.5, 1], cmap = "coolwarm_r", alpha = 0.3)
''' 
'Contourf' is used to draw contour plots, and 'levels' to define the contour level.
'Cmap' is used to represent the mapping of colors and 'alpha' represents transparency
'''

# Plot the feature points X1 and X2
plt.scatter(X1[:, 0], X1[:, 1], color = 'red', label = "Class 1")
plt.scatter(X2[:, 0], X2[:, 1], color = 'blue', label = "Class 2")

plt.legend()
plt.title("MLP Initial Decision Boundary (Untrained)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show(block = False)

# Use the untrained MLP model to infer the classification results of the input features
with torch.no_grad():
    X_concat_tensor = torch.tensor(X_concat, dtype = torch.float32)
    Y_pred = mlp(X_concat_tensor).numpy()
    Y_pred_class = (Y_pred > 0.5).astype(int) # Convert probability to categorical labels

# Calculate classification metrics
accuracy = accuracy_score(Y_concat, Y_pred_class)
precision = precision_score(Y_concat, Y_pred_class, zero_division = 1)
recall = recall_score(Y_concat, Y_pred_class, zero_division = 1)

# Calculate and plot the ROC curve
fpr, tpr, thresholds = roc_curve(Y_concat, Y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(
    f"Receiver Operating Characteristic\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
plt.legend(loc = "lower right")
plt.show()
