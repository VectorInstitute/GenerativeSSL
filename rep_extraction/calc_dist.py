import numpy as np
from scipy.linalg import orthogonal_procrustes
import time

def orthogonal_procrustes_distance(X, Y):
    R, _ = orthogonal_procrustes(X, Y)
    X_transformed = X @ R
    distance = np.linalg.norm(X_transformed - Y, "fro")

    return distance


def center_and_normalize(X):
    # Center the matrix
    X_centered = X - np.mean(X, axis=1, keepdims=True)

    # Normalize the matrix
    norm = np.linalg.norm(X_centered, ord='fro')
    X_normalized = X_centered / norm

    return X_normalized

def lin_cka_dist(A, B):
    # Center and normalize the representations
    A = center_and_normalize(A)
    B = center_and_normalize(B)

    # Compute the numerator
    sim_AB = np.linalg.norm(A @ B.T, ord='fro') ** 2

    # Compute the denominator
    norm_A = np.linalg.norm(A @ A.T, ord='fro')
    norm_B = np.linalg.norm(B @ B.T, ord='fro')

    # Compute the Linear CKA distance
    return 1 - sim_AB / (norm_A * norm_B)


# Load the representations
X = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_clip_features.npy").T
Y = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_simclr_features.npy").T
Z = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_simclr2_features.npy").T

print("Start of the CKA calculation")
start_time = time.time()
print('Linear CKA, between CLIP and SimCLR 1: {}'.format(lin_cka_dist(X, Y)))
end_time = time.time()
print("Time taken for first Linear CKA:", end_time - start_time, "seconds")

start_time = time.time()
print('Linear CKA, between CLIP and SimCLR 2: {}'.format(lin_cka_dist(X, Z)))
end_time = time.time()
print("Time taken for second Linear CKA:", end_time - start_time, "seconds")

print("Start of the Orthogonal Procrustes distance calculation")
start_time = time.time()
op_distance = orthogonal_procrustes_distance(X, Y)
end_time = time.time()
print("Orthogonal Procrustes distance between CLIP and SimCLR 1:", op_distance)
print("Time taken for first Orthogonal Procrustes distance:", end_time - start_time, "seconds")


print("Start of the Orthogonal Procrustes distance calculation")
start_time = time.time()
op_distance = orthogonal_procrustes_distance(X, Z)
end_time = time.time()
print("Orthogonal Procrustes distance between CLIP and SimCLR 2:", op_distance)
print("Time taken for second Orthogonal Procrustes distance:", end_time - start_time, "seconds")
