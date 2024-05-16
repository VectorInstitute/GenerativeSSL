import numpy as np

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

    print(A.max())
    print(B.max())

    # Compute the numerator
    C = A @ B.T
    print(C.shape)
    sim_AB = np.linalg.norm(A @ B.T, ord='fro') ** 2
    print("sim", sim_AB)

    # Compute the denominator
    norm_A = np.linalg.norm(A @ A.T, ord='fro')
    norm_B = np.linalg.norm(B @ B.T, ord='fro')
    print(norm_A, norm_B)

    # Compute the Linear CKA distance
    return 1 - sim_AB / (norm_A * norm_B)

def opd(A, B):
    # Center and normalize the representations
    A = center_and_normalize(A)
    B = center_and_normalize(B)

    # Calculate norms
    frobenius_norm_A = np.linalg.norm(A, 'fro')**2
    frobenius_norm_B = np.linalg.norm(B, 'fro')**2
    nuclear_norm_ATB = np.linalg.norm(np.dot(A.T, B), ord='nuc')
    # Calculate the OPD
    distance = frobenius_norm_A + frobenius_norm_B - 2 * nuclear_norm_ATB
    return distance

# Load the representations
X = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_clip_features.npy") # shape (50000, 512)
Y = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_simclr_features.npy") # shape (50000, 2048)
Z = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_simclr2_features.npy") # shape (50000, 2048)

print('CKA between CLIP and SimCLR 1: {}'.format(lin_cka_dist(X.T, Y.T)))
print('CKA between CLIP and SimCLR 2: {}'.format(lin_cka_dist(X.T, Z.T)))
print('CKA between SimCLR 1 and SimCLR 2: {}'.format(lin_cka_dist(Y.T, Z.T)))

# print("_"*50)

print("OPD between CLIP and SimCLR 1:", opd(X, Y))
print("OPD between CLIP and SimCLR 2:", opd(X, Z))
print("OPD between SimCLR 1 and SimCLR 2:", opd(Y, Z))
