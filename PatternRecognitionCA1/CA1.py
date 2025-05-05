# Import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

# Define an MLP class to implement a neural network for binary classification task
class MLP:
    def __init__(self):
        # Manually define weights, biases, and intermediate variables
        self.z1 = None
        self.h1 = None
        self.z2 = None
        self.h2 = None
        # Constrain initial weights to avoid gradient vanishing
        self.W1 = torch.randn(2, 3) * 0.1 # Two input neurons/One hidden layer with three neurons
        self.b1 = torch.zeros(1, 3) # The bias of hidden layer
        self.W2 = torch.randn(3, 1) * 0.1 # One output neuron
        self.b2 = torch.zeros(1, 1) # The bias of output layer

    def forward(self, x):
        # Perform manual forward propagation from scratch
        self.z1 = x @ self.W1 + self.b1 # Calculate the input of hidden layer
        self.h1 = torch.relu(self.z1) # Activate hidden layer
        self.z2 = self.h1 @ self.W2 + self.b2 # Calculate the input of output layer
        self.h2 = torch.sigmoid(self.z2) # Activate output layer
        return self.z1, self.h1, self.z2, self.h2

# Derivative calculation function of the ReLU function
def relu_derivative(x):
    return (x > 0).float()

# Derivative calculation function of the Sigmoid function
def sigmoid_derivative(x):
    sig = torch.sigmoid(x)
    return sig * (1 - sig)

# Perform manual backpropagation from scratch
def back_propagation(model, x, y, lr):
    z1, h1, z2, h2 = model.forward(x)
    y_predict = h2
    loss_value = (y_predict - y) ** 2 # Use squared error as the loss function
    d_loss = 2 * (y_predict - y) # Calculate the gradient of the loss with respect to the output

    delta_out = d_loss * sigmoid_derivative(z2) # Calculate the output layer error
    dW2 = h1.T @ delta_out / len(x) # Calculate the gradient of the output layer weights
    db2 = torch.mean(delta_out, dim = 0)

    delta_hidden = delta_out @ model.W2.T * relu_derivative(z1) # Calculate the hidden layer error
    dW1 = x.T @ delta_hidden / len(x) # Calculate the gradient of the hidden layer weight
    db1 = torch.mean(delta_hidden, dim = 0)

    # Update the weights and biases
    model.W2 -= lr * dW2
    model.b2 -= lr * db2
    model.W1 -= lr * dW1
    model.b1 -= lr * db1

    return loss_value.mean().item()

# Load the pre-generated data and create corresponding labels
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')
X_concat = np.concatenate((X1, X2))
Y_concat = np.concatenate((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))
X_tensor = torch.tensor(X_concat, dtype = torch.float32)
Y_tensor = torch.tensor(Y_concat, dtype = torch.float32).view(-1, 1)

# Instantiate the MLP
mlp = MLP()

# Define hyperparameters
epochs = 100000
learning_rate = 0.01
loss_list = []

# Model training
for epoch in range(epochs):
    loss = back_propagation(mlp, X_tensor, Y_tensor, learning_rate)
    loss_list.append(loss)
    if epoch % 10000 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Generate a grid of coordinates based on the two features X1 and X2
x_min, x_max = X_concat[:, 0].min() - 1, X_concat[:, 0].max() + 1
y_min, y_max = X_concat[:, 1].min() - 1, X_concat[:, 1].max() + 1
res = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
grid_tensor = torch.tensor(grid_points)

# Use the trained MLP model to infer the classification results of the grid coordinates
_, _, _, grid_pred = mlp.forward(grid_tensor)
Z = grid_pred.detach().numpy().reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, levels = [0, 0.5, 1], cmap = "coolwarm_r", alpha=0.3)

# Plot the feature points X1 and X2
plt.scatter(X1[:, 0], X1[:, 1], color = 'red', label = "Class 1")
plt.scatter(X2[:, 0], X2[:, 1], color = 'blue', label = "Class 2")
plt.legend()
plt.title("MLP Decision Boundary (Trained)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Data classification
_, _, _, y_pred = mlp.forward(X_tensor)
Y_pred = y_pred.detach().numpy()
Y_pred_class = (Y_pred > 0.5).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(Y_concat, Y_pred_class)
precision = precision_score(Y_concat, Y_pred_class, zero_division = 1)
recall = recall_score(Y_concat, Y_pred_class, zero_division = 1)

# Calculate and plot the ROC curve
fpr, tpr, thresholds = roc_curve(Y_concat, Y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', lw=2, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw=2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f"ROC Curve\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
plt.legend(loc = "lower right")

# Plot the loss curve
plt.figure()
plt.plot(range(epochs), loss_list, color = 'darkorange', lw = 2, label = 'Loss curve')
plt.xlim([0.0, epochs * 1.1])
plt.ylim([min(loss_list) * 0.9, max(loss_list) * 1.1])
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.title('Loss Trajectory')
plt.legend(loc = "lower right")
plt.show()