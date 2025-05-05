import images_prep
import lda_from_scratch
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

projection_matrix_9, X_lda_9 = lda_from_scratch.mannual_lda(X_train, Y_train, 9) # Perform LDA on the training images with 9 components
projection_matrix_15, X_lda_15 = lda_from_scratch.mannual_lda(X_train, Y_train, 15) # Perform LDA on the training images with 15 components

X_test_lda_9 = X_test @ projection_matrix_9
X_test_lda_15 = X_test @ projection_matrix_15

knn_9 = KNeighborsClassifier(n_neighbors=1) # Create a KNN classifier with 1 neighbor
knn_9.fit(X_lda_9, Y_train) # Fit the classifier with the training data
Y_pred_9 = knn_9.predict(X_test_lda_9) # Predict the test data

pie_mask = (Y_test != 69) # Boolean array to find photos that are not selfies
acc_pie_9 = accuracy_score(Y_test[pie_mask], Y_pred_9[pie_mask]) # Calculate the accuracy of the model on non-selfie photos

selfie_mask = (Y_test == 69) # Boolean array to find the selfie photos
acc_selfie_9 = accuracy_score(Y_test[selfie_mask], Y_pred_9[selfie_mask]) # Calculate the accuracy of the model on selfie photos

print(f"[LDA-9] Accuracy on CMU PIE test images: {acc_pie_9:.2%}")
print(f"[LDA-9] Accuracy on Selfie test images: {acc_selfie_9:.2%}")

knn_15 = KNeighborsClassifier(n_neighbors=1)
knn_15.fit(X_lda_15, Y_train)
Y_pred_15 = knn_15.predict(X_test_lda_15)

acc_pie_15 = accuracy_score(Y_test[pie_mask], Y_pred_15[pie_mask])
acc_selfie_15 = accuracy_score(Y_test[selfie_mask], Y_pred_15[selfie_mask])

print(f"[LDA-15] Accuracy on CMU PIE test images: {acc_pie_15:.2%}")
print(f"[LDA-15] Accuracy on Selfie test images: {acc_selfie_15:.2%}")