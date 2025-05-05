import images_prep
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ======= Data Preparation =======
# Metric number: A0313771H
random.seed(71) # Random seed for reproducibility
selected_indices = random.sample(range(1, 69), 25) # Randomly select 25 people from PIE dataset
selected_indices.append(69) # Add selfie photos to the list

PIE_path = "./PIE" # Path to PIE dataset

training_images, training_labels, test_images, test_labels = images_prep.get_images(PIE_path, selected_indices) # Get the images and labels from the dataset

X_train = np.array(training_images).reshape(-1, 1, 32, 32) # Reshape the training images to (N, C, H, W)
X_test = np.array(test_images).reshape(-1, 1, 32, 32) # Reshape the test images to (N, C, H, W)
X_train = torch.tensor(X_train, dtype=torch.float32) # Convert the training images to PyTorch tensor
X_test = torch.tensor(X_test, dtype=torch.float32) # Convert the test images to PyTorch tensor

label_encoder = LabelEncoder() # Create a label encoder (original labels are between 1-69, we need to convert them to 0-25)
Y_train_encoded = label_encoder.fit_transform(training_labels) # Encode the training labels
Y_test_encoded = label_encoder.transform(test_labels) # Encode the test labels

Y_train = torch.tensor(Y_train_encoded, dtype=torch.long) # Convert the training labels to PyTorch tensor
Y_test = torch.tensor(Y_test_encoded, dtype=torch.long) # Convert the test labels to PyTorch tensor

train_dataset = TensorDataset(X_train, Y_train) # Create a training dataset
test_dataset = TensorDataset(X_test, Y_test) # Create a test dataset

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Create a training data loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Create a test data loader

# ======= Define a CNN According to the Requirements =======
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0) # Convolutional layer with 20 filters. Kernel size 5x5, stride 1, padding 0
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0) # Convolutional layer with 50 filters. Kernel size 5x5, stride 1, padding 0
        self.fc = nn.Linear(50 * 5 * 5, 500) # Fully connected layer with 500 units, 5 * 5 because of two convolutional layers and two max-pooling layers 
        self.op = nn.Linear(500, 26) # Output layer with 26 units (26 classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Max pooling layer with kernel size 2x2, stride 2
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Max pooling layer with kernel size 2x2, stride 2
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc(x))
        x = self.op(x) # Output layer without softmax activation because CrossEntropyLoss includes it
        return x
    
# ======= Define the Training Function =======
def train(model, train_loader, optimizer, criterion, device, verbose=False):
    model.train() # Set the model to training mode
    running_loss = 0.0 # Initialize the running loss
    total_correct = 0 # Initialize the total correct predictions
    total_samples = 0 # Initialize the total samples

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) # Move the inputs and labels to the device
        optimizer.zero_grad() # Zero the parameter gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Calculate the loss
        loss.backward() # Backward pass
        optimizer.step() # Optimize
        running_loss += loss.item() # Accumulate the loss
        _, preds = torch.max(outputs, dim=1) # Get the predicted labels
        total_correct += torch.sum(preds == labels).item() # Accumulate the correct predictions
        total_samples += labels.size(0) # Accumulate the total samples
    
    avg_loss = running_loss / len(train_loader) # Calculate the average loss
    accuracy = total_correct / total_samples # Calculate the accuracy

    if verbose:
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2%}")

    return avg_loss, accuracy

# ======= Define the Evaluation Function =======
def evaluate(model, test_loader, criterion, device, verbose=False):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item() 
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    accuracy = total_correct / total_samples

    if verbose:
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2%}")

    return avg_loss, accuracy, all_preds, all_labels

# ======= Train the Model =======
num_epochs = 20
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Define the training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
model = CNN().to(device) # Move the model to the device
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer 
criterion = nn.CrossEntropyLoss() # Cross-entropy loss

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_accuracy, eva_preds, eva_labels = evaluate(model, test_loader, criterion, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}")

# Calculate the evaluation metrics: accuracy score, precision score, recall score, f1 score
accuracy = accuracy_score(eva_labels, eva_preds)
precision = precision_score(eva_labels, eva_preds, average='macro')
recall = recall_score(eva_labels, eva_preds, average='macro')
f1 = f1_score(eva_labels, eva_preds, average='macro')

print(f"\n=== Final Evaluation on Test Set ===")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

# Save the model
model_path = "./cnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Plot the training and test losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')
plt.show(block=False)

# Plot the training and test accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')
plt.show()

