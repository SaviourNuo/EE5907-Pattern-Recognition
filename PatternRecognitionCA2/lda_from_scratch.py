import numpy as np

# ======= Perform LDA from Scratch =======
def mannual_lda(X, Y, p):
    classes = np.unique(Y) # Get the unique classes from the labels
    class_mean = np.array([np.mean(X[Y == cls], axis=0) for cls in classes]) # Calculate the mean of each class
    X_mean = np.mean(X, axis=0) # Calculate the overall mean

    S_i_list = [np.cov(X[Y == cls], rowvar=False, bias = True) for cls in classes] # Calculate the class-specific covariance matrix (bias = True for biased estimation)
    S_i_hat_list = []
    for cls, S_i in zip(classes, S_i_list): # Iterate through each class and its corresponding scatter matrix (zip to Pack two lists into pairs by position)
        N_i = len(X[Y == cls]) # Get the number of samples in the class
        S_i_hat = N_i * S_i # Calculate the class-specific scatter matrix
        S_i_hat_list.append(S_i_hat) # Append the scatter matrix to the list

    n_i_list = [len(X[Y == cls]) for cls in classes] # Get the number of samples in each class
    N = sum(n_i_list) # Get the total number of samples
    S_w = np.zeros_like(S_i_list[0]) # Initialize the within-class covariance matrix
    for n_i, S_i in zip(n_i_list, S_i_list):
        P_i = n_i / N # Calculate the class-specific probability
        S_w += P_i * S_i # Calculate the within-class covariance matrix

    S_w_hat = np.sum(S_i_hat_list, axis=0) # Calculate the total within-class scatter matrix

    S_b = np.zeros((X.shape[1], X.shape[1])) # Initialize the between-class covariance matrix
    S_b_hat = np.zeros_like(S_b) # Initialize the between-class scatter matrix

    for n_i, mean_i in zip(n_i_list, class_mean):
        mean_diff = (mean_i - X_mean).reshape(-1, 1)
        S_b_hat += n_i * (mean_diff @ mean_diff.T)
        S_b += (n_i / N) * (mean_diff @ mean_diff.T)
    
    # S_T = S_w + S_b # Calculate the total covariance matrix
    # S_T_hat = S_w_hat + S_b_hat # Calculate the total scatter matrix

    # eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.pinv(S_w_hat) @ S_b_hat) # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(S_w_hat) @ S_b_hat) # Calculate the eigenvalues and eigenvectors
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    sort_eigen_idx = np.argsort(eigenvalues)[::-1] # Sort the eigenvalues in descending order
    eigenvalues = eigenvalues[sort_eigen_idx] # Sort the eigenvalues
    eigenvectors = eigenvectors[:, sort_eigen_idx] # Sort the eigenvectors
    projection_matrix = eigenvectors[:, :p] # Select the first p eigenvectors as the projection matrix
    X_lda = X @ projection_matrix # Project the images to the new space

    return projection_matrix, X_lda # Return the projection matrix and the projected images[]

