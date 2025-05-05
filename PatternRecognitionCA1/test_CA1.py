# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

# Define an MLP class to implement a neural network for binary classification task
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 3) # Two input neurons/One hidden layer with three neurons
        self.out = nn.Linear(3, 1) # One output neuron
        self.relu = nn.ReLU() # Use ReLU as the activation function after the hidden layer
        self.sigmoid = nn.Sigmoid() # Use Sigmoid as the activation function following the output layer

        nn.init.normal_(self.hidden.weight, mean = 0.0, std = 0.1) # Initialize weights following a normal distribution
        nn.init.normal_(self.out.weight, mean = 0.0, std = 0.1)
        nn.init.zeros_(self.hidden.bias) # Set initial biases to zero
        nn.init.zeros_(self.out.bias)

    def forward(self, x): # Perform forward propagation and output intermediate variables.
        z1 = self.hidden(x)
        h1 = self.relu(z1)
        z2 = self.out(h1)
        h2 = self.sigmoid(z2)
        return z1, h1, z2, h2

# Derivative calculation function of the ReLU function
def relu_derivative(x):
    return (x > 0).float()

# Derivative calculation function of the Sigmoid function
def sigmoid_derivative(x):
    sig = torch.sigmoid(x)
    return sig * (1 - sig)

# Perform manual backpropagation from scratch
def back_propagation(model, x, y, lr):
    z1, h1, z2, h2 = model(x)
    y_predict = h2
    loss_value = (y_predict - y) ** 2 # Use squared error as the loss function
    d_loss = 2 * (y_predict - y) # Calculate the gradient of the loss with respect to the output

    delta_out = d_loss * sigmoid_derivative(z2) # Calculate the output layer error
    dw2 = delta_out.T @ h1 / len(x) # Calculate the gradient of the output layer weights
    db2 = torch.mean(delta_out, dim = 0)

    delta_hidden = delta_out @ model.out.weight * relu_derivative(z1) # Calculate the hidden layer error
    dw1 = delta_hidden.T @ x / len(x) # Calculate the gradient of the hidden layer weight
    db1 = torch.mean(delta_hidden, dim = 0)

    # Update the weights and biases
    model.out.weight.data -= lr * dw2
    model.out.bias.data -= lr * db2

    model.hidden.weight.data -= lr * dw1
    model.hidden.bias.data -= lr * db1

    return loss_value.mean().item()

# Load the pre-generated data and create corresponding labels.
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
    loss_list.append(loss) # Print the loss function value
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
_, _, _, grid_pred = mlp(grid_tensor)
Z = grid_pred.detach().numpy()
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap = "coolwarm_r", alpha = 0.3)

# Plot the feature points X1 and X2
plt.scatter(X1[:, 0], X1[:, 1], color = 'red', label = "Class 1")
plt.scatter(X2[:, 0], X2[:, 1], color = 'blue', label = "Class 2")

plt.legend()
plt.title("MLP Decision Boundary (Trained)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show(block = False)

# Data classification
_, _, _, y_pred = mlp(X_tensor)
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
plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(
    f"Receiver Operating Characteristic\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
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