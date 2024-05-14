import numpy as np
from scipy.linalg import orthogonal_procrustes
import time

def linear_cka(X, Y):
    """
    Compute the Linear Centered Kernel Alignment (CKA) between two matrices X and Y.

    Parameters
    ----------
    X: numpy.ndarray, shape (n_samples, n_features)
    Y: numpy.ndarray, shape (n_samples, n_features)

    Returns
    -------
    cka: float, the CKA value between X and Y
    """
    def center_gram_matrix(K):
        """Center the Gram matrix."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_X = X @ X.T
    K_Y = Y @ Y.T

    K_X_centered = center_gram_matrix(K_X)
    K_Y_centered = center_gram_matrix(K_Y)

    numerator = np.trace(K_X_centered @ K_Y_centered)
    denominator = np.sqrt(np.trace(K_X_centered @ K_X_centered) * np.trace(K_Y_centered @ K_Y_centered))

    return numerator / denominator

def orthogonal_procrustes_distance(X, Y):
    """
    Compute the Orthogonal Procrustes distance between two matrices X and Y.

    Parameters
    ----------
    X: numpy.ndarray, shape (n_samples, n_features)
    Y: numpy.ndarray, shape (n_samples, n_features)

    Returns
    -------
    distance: float, the Orthogonal Procrustes distance between X and Y
    """
    R, _ = orthogonal_procrustes(X, Y)
    X_transformed = X @ R
    distance = np.linalg.norm(X_transformed - Y, "fro")

    return distance

# Example usage:
X = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_clip_features.npy")
Y = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_clip_features.npy")

print("Start of the CKA calculation")
start_time = time.time()
cka_value = linear_cka(X, Y)
end_time = time.time()
print("Linear CKA value:", cka_value)
print("Time taken for Linear CKA:", end_time - start_time, "seconds")

print("Start of the Orthogonal Procrustes distance calculation")
start_time = time.time()
op_distance = orthogonal_procrustes_distance(X, Y)
end_time = time.time()
print("Orthogonal Procrustes distance:", op_distance)
print("Time taken for Orthogonal Procrustes distance:", end_time - start_time, "seconds")
