import numpy as np

def center_and_normalize(X):
    X = X - np.mean(X, axis=1, keepdims=True)
    coef = 100.0
    X /= coef
    # Normalize the matrix
    norm = np.linalg.norm(X, ord="fro")
    return X / norm

def cka(A, B):
    # Center and normalize the representations
    A = center_and_normalize(A)
    B = center_and_normalize(B)

    # Compute the numerator
    AB_T = A @ B.T
    N_AB_T = np.linalg.norm(AB_T, ord='fro') ** 2

    # Compute the denominator
    AA_T = A @ A.T
    BB_T = B @ B.T
    N_AA_T = np.linalg.norm(AA_T, ord='fro')
    N_BB_T = np.linalg.norm(BB_T, ord='fro')

    # Compute the Linear CKA distance
    return 1 - N_AB_T / (N_AA_T * N_BB_T)

def opd(A, B):
    # Center and normalize the representations
    A = center_and_normalize(A.T)
    B = center_and_normalize(B.T)

    # Calculate norms
    frobenius_norm_A = np.linalg.norm(A, 'fro')**2
    frobenius_norm_B = np.linalg.norm(B, 'fro')**2
    nuclear_norm_ATB = np.linalg.norm(np.dot(A.T, B), ord='nuc')
    # Calculate the OPD
    return frobenius_norm_A + frobenius_norm_B - 2 * nuclear_norm_ATB

# Load the representations
X = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_clip_features.npy") # shape (50000, 512)
Y = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_simclr_features.npy") # shape (50000, 2048)
Z = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_simclr2_features.npy") # shape (50000, 2048)

print('CKA between CLIP and SimCLR 1: {}'.format(cka(X.T, Y.T)))
print('CKA between CLIP and SimCLR 2: {}'.format(cka(X.T, Z.T)))
print('CKA between SimCLR 1 and SimCLR 2: {}'.format(cka(Y.T, Z.T)))

print("_"*50)

print("OPD between CLIP and SimCLR 1:", opd(X.T, Y.T))
print("OPD between CLIP and SimCLR 2:", opd(X.T, Z.T))
print("OPD between SimCLR 1 and SimCLR 2:", opd(Y.T, Z.T))
