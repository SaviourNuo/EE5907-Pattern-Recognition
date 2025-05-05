import images_prep
import pca_from_scratch
import numpy as np
import random
from sklearn.mixture import GaussianMixture
import os
import matplotlib.pyplot as plt

os.environ["LOKY_MAX_CPU_COUNT"] = "20" # Set the maximum number of CPU cores to be used by the joblib library

# Metric number: A0313771H
random.seed(71)  # For reproducibility
selected_idx = random.sample(range(1, 69), 25) # Randomly select 25 people from PIE dataset
selected_idx.append(69) # Add selfie photos to the list

PIE_path = "./PIE" # Path to PIE dataset
training_images, training_labels, test_images, test_labels = images_prep.get_images(PIE_path, selected_idx)

X_train = np.array([img.flatten() for img in training_images]) # Flatten the training images
X_test = np.array([img.flatten() for img in test_images]) # Flatten the test images

colors = ['red', 'green', 'blue']

# ======= GMM on raw vectorized images =======
gmm_raw = GaussianMixture(n_components=3, covariance_type='full', random_state=71) # Create a GMM classifier with 3 components
gmm_raw.fit(X_train) # Fit the classifier with the training data
Y_pred_raw = gmm_raw.predict(X_train) # Predict the training data

_, _, X_raw_pca_2d = pca_from_scratch.mannual_pca(X_train, 2) # Perform PCA on the training images with 2 principal components

plt.figure(figsize=(8, 6))
for cluster_id, color in enumerate(colors):
    idx = (Y_pred_raw == cluster_id)
    plt.scatter(X_raw_pca_2d[idx, 0], X_raw_pca_2d[idx, 1], c=color, label=str(cluster_id),
                s=20, alpha=0.6, edgecolors='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM Clustering (Raw Images) - Visualized in PCA 2D")
plt.legend()

# ======= GMM on PCA p=80 images =======
_, _, X_pca_80 = pca_from_scratch.mannual_pca(X_train, 80)
gmm_80 = GaussianMixture(n_components=3, covariance_type='full', random_state=71)
gmm_80.fit(X_pca_80)
Y_pred_80 = gmm_80.predict(X_pca_80)

_, _, X_pca_80_2d = pca_from_scratch.mannual_pca(X_pca_80, 2)

plt.figure(figsize=(8, 6))
for cluster_id, color in enumerate(colors):
    idx = (Y_pred_80 == cluster_id)
    plt.scatter(X_pca_80_2d[idx, 0], X_pca_80_2d[idx, 1], c=color, label=str(cluster_id),
                s=20, alpha=0.6, edgecolors='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM Clustering (PCA p=80) - Visualized in PCA 2D")
plt.legend()

# ======= GMM on PCA p=200 images =======
_, _, X_pca_200 = pca_from_scratch.mannual_pca(X_train, 200)
gmm_200 = GaussianMixture(n_components=3, covariance_type='full', random_state=71)
gmm_200.fit(X_pca_200)
Y_pred_200 = gmm_200.predict(X_pca_200)

_, _, X_pca_200_2d = pca_from_scratch.mannual_pca(X_pca_200, 2)

plt.figure(figsize=(8, 6))
for cluster_id, color in enumerate(colors):
    idx = (Y_pred_200 == cluster_id)
    plt.scatter(X_pca_200_2d[idx, 0], X_pca_200_2d[idx, 1], c=color, label=str(cluster_id),
                s=20, alpha=0.6, edgecolors='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM Clustering (PCA p=200) - Visualized in PCA 2D")
plt.legend()
plt.show()
