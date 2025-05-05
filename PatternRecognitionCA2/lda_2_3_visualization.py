import images_prep
import lda_from_scratch
import numpy as np
import matplotlib.pyplot as plt
import random

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

projection_matrix_2, X_lda_2 = lda_from_scratch.mannual_lda(X_train, Y_train, 2) # Perform LDA on the training images with 2 components
projection_matrix_3, X_lda_3 = lda_from_scratch.mannual_lda(X_train, Y_train, 3) # Perform LDA on the training images with 3 components

# ======= 2D LDA Visualization =======
plt.figure(figsize=(8, 6)) # Set the figure size
classes = np.unique(Y_train) # Get the unique classes from the training labels, discard the duplicates

for cls in classes:
    idx = (Y_train == cls) # Create a boolean array of the same length as Y_train, True if the class is cls, False otherwise
    if cls == 69: # Selfie class is plotted as red stars
        plt.scatter(X_lda_2[idx, 0], X_lda_2[idx, 1],
                    color='red', marker='*', s=120, edgecolors='black',
                    label=f"Selfie Class ({cls})")
    else: # Other classes are plotted as circles
        plt.scatter(X_lda_2[idx, 0], X_lda_2[idx, 1],
                    alpha=0.6, s=40, label=f"Class {cls}")
        
plt.title("LDA 2D Projection of Training Data", fontsize=14)
plt.xlabel("LD Vector 1", fontsize=12)
plt.ylabel("LD Vector 2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(fontsize=8, markerscale=0.8, loc='best')
plt.tight_layout()
plt.show(block=False)

# ======= 3D LDA Visualization =======
plt.figure(figsize=(8, 6)) # Set the figure size
ax = plt.axes(projection='3d') # Create a 3D plot

for cls in classes:
    idx = (Y_train == cls) # Create a boolean array of the same length as Y_train, True if the class is cls, False otherwise
    if cls == 69: # Selfie class is plotted as red stars
        ax.scatter(X_lda_3[idx, 0], X_lda_3[idx, 1], X_lda_3[idx, 2],
                    color='red', marker='*', s=120, edgecolors='black',
                    label=f"Selfie Class ({cls})")
    else: # Other classes are plotted as circles
        ax.scatter(X_lda_3[idx, 0], X_lda_3[idx, 1], X_lda_3[idx, 2],
                    alpha=0.6, s=40, label=f"Class {cls}")

ax.set_title("LDA 3D Projection of Training Data", fontsize=14)
ax.set_xlabel("LD Vector 1", fontsize=12)
ax.set_ylabel("LD Vector 2", fontsize=12)
ax.set_zlabel("LD Vector 3", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(fontsize=8, markerscale=0.8, loc='best')
plt.tight_layout()
plt.show(block=False)

# ======= Eigenfaces Visualization =======
def plot_eigenfaces(projection_matrix, count, title_prefix="Fisherfaces"):
    plt.figure(figsize=(3 * count, 4)) # Set the figure size
    for i in range(count):
        eigenface = projection_matrix[:, i].reshape(32, 32) # Reshape the fisherfaces to 32x32, evry column is an eigenface (1024, i)
        plt.subplot(1, count, i + 1)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f"{title_prefix}\nLD{i + 1}", fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)

plot_eigenfaces(projection_matrix_2, 2, title_prefix="Top 2 Fisherfaces (p=2)")
plot_eigenfaces(projection_matrix_3, 3, title_prefix="Top 3 Fisherfaces (p=3)")
plt.show() # Display all the plots