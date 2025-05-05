import images_prep
import pca_from_scratch
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "20" # Set the maximum number of CPU cores to be used by the joblib library

# Metric number: A0313771H
random.seed(71) # Random seed for reproducibility
selected_idx = random.sample(range(1, 69), 25) # Randomly select 25 people from PIE dataset
selected_idx.append(69) # Add selfie photos to the list

PIE_path = "./PIE" # Path to PIE dataset

training_images, training_labels, test_images, test_labels = images_prep.get_images(PIE_path, selected_idx) # Get the images and labels from the dataset

X_train = np.array([img.flatten() for img in training_images]) # Flatten the traing images
X_test = np.array([img.flatten() for img in test_images]) # Flatten the test images
Y_train = np.array(training_labels) # Convert the training labels to numpy array
Y_test = np.array(test_labels) # Convert the test labels to numpy array

X_mean_80, projection_matrix_80, X_pca_80 = pca_from_scratch.mannual_pca(X_train, 80) # Perform PCA on the training images with 80 components
X_mean_200, projection_matrix_200, X_pca_200 = pca_from_scratch.mannual_pca(X_train, 200) # Perform PCA on the training images with 200 components

X_test_pca_80 = (X_test - X_mean_80) @ projection_matrix_80 # Project the test images to the new space (centralization using the mean of training images)
X_test_pca_200 = (X_test -  X_mean_200) @ projection_matrix_200

knn_80 = KNeighborsClassifier(n_neighbors=1) # Create a KNN classifier with 1 neighbor
knn_80.fit(X_pca_80, Y_train) # Fit the classifier with the training data
Y_pred_80 = knn_80.predict(X_test_pca_80) # Predict the test data

pie_mask = (Y_test != 69) # Boolean array to find photos that are not selfies
acc_pie_80 = accuracy_score(Y_test[pie_mask], Y_pred_80[pie_mask]) # Calculate the accuracy of the model on non-selfie photos

selfie_mask = (Y_test == 69) # Boolean array to find the selfie photos
acc_selfie_80 = accuracy_score(Y_test[selfie_mask], Y_pred_80[selfie_mask]) # Calculate the accuracy of the model on selfie photos

print(f"[PCA-80] Accuracy on CMU PIE test images: {acc_pie_80:.2%}")
print(f"[PCA-80] Accuracy on Selfie test images: {acc_selfie_80:.2%}")

knn_200 = KNeighborsClassifier(n_neighbors=1)
knn_200.fit(X_pca_200, Y_train)
Y_pred_200 = knn_200.predict(X_test_pca_200)

acc_pie_200 = accuracy_score(Y_test[pie_mask], Y_pred_200[pie_mask])
acc_selfie_200 = accuracy_score(Y_test[selfie_mask], Y_pred_200[selfie_mask])

print(f"[PCA-200] Accuracy on CMU PIE test images: {acc_pie_200:.2%}")
print(f"[PCA-200] Accuracy on Selfie test images: {acc_selfie_200:.2%}")