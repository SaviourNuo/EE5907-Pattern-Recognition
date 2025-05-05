import numpy as np

# ======= Perform PCA from Scratch =======
def mannual_pca(X, p):
    
    X_mean = np.mean(X, axis=0) # Calculate the mean of the pixels at the same position of all images
    X_centered = X - X_mean # Subtract the mean from the images (Centralization)
    cov_matrix = np.cov(X_centered, rowvar=False, bias = True) # Calculate the covariance matrix (rowvar=False for each column to represent a variable) (bias = True for biased estimation)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # Calculate the eigenvalues and eigenvectors (eigh for symmetric matrix)
    sort_eigen_idx = np.argsort(eigenvalues)[::-1] # Sort the eigenvalues in descending order
    eigenvalues = eigenvalues[sort_eigen_idx] # Sort the eigenvalues
    eigenvectors = eigenvectors[:, sort_eigen_idx] # Sort the eigenvectors
    projection_matrix = eigenvectors[:, :p] # Select the first p eigenvectors as the projection matrix
    X_pca = X_centered @ projection_matrix # Project the images to the new space

    return X_mean, projection_matrix, X_pca # Return the mean (for reconstruction), projection matrix, and the projected images
