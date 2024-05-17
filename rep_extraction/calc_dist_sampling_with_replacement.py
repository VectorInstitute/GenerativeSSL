import numpy as np
import pickle
import argparse

def center_and_normalize(X):
    X = X - np.mean(X, axis=1, keepdims=True)
    coef = 100.0
    X /= coef
    norm = np.linalg.norm(X, ord="fro")
    return X / norm

def cka(A, B):
    A = center_and_normalize(A)
    B = center_and_normalize(B)
    N_AB_T = np.linalg.norm(A @ B.T, ord='fro') ** 2
    N_AA_T = np.linalg.norm(A @ A.T, ord='fro')
    N_BB_T = np.linalg.norm(B @ B.T, ord='fro')
    return 1 - N_AB_T / (N_AA_T * N_BB_T)

def opd(A, B):
    A = center_and_normalize(A.T)
    B = center_and_normalize(B.T)
    frobenius_norm_A = np.linalg.norm(A, 'fro')**2
    frobenius_norm_B = np.linalg.norm(B, 'fro')**2
    nuclear_norm_ATB = np.linalg.norm(np.dot(A.T, B), ord='nuc')
    return frobenius_norm_A + frobenius_norm_B - 2 * nuclear_norm_ATB

def bootstrap_cka_opd(X, Y, Z, runs):
    cka_results = {"X_Y": [], "X_Z": [], "Y_Z": []}
    opd_results = {"X_Y": [], "X_Z": [], "Y_Z": []}

    for r in range(runs):
        print(f"Run {r+1}/{runs}")
        indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
        X_sample = X[indices]
        Y_sample = Y[indices]
        Z_sample = Z[indices]

        cka_results["X_Y"].append(cka(X_sample.T, Y_sample.T))
        cka_results["X_Z"].append(cka(X_sample.T, Z_sample.T))
        cka_results["Y_Z"].append(cka(Y_sample.T, Z_sample.T))

        opd_results["X_Y"].append(opd(X_sample.T, Y_sample.T))
        opd_results["X_Z"].append(opd(X_sample.T, Z_sample.T))
        opd_results["Y_Z"].append(opd(Y_sample.T, Z_sample.T))

    return cka_results, opd_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CKA and OPD calculations")
    parser.add_argument("--runs", type=int, default=20, help="Number of bootstrap runs")

    args = parser.parse_args()
    runs = args.runs

    # Load the representations
    X = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_clip_features.npy")
    Y = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_simclr_features.npy")
    Z = np.load("/projects/imagenet_synthetic/extracted_representations/imagenet_val_simclr2_features.npy")

    # Perform bootstrapping
    cka_results, opd_results = bootstrap_cka_opd(X, Y, Z, runs)

    # Save the results as pickle files
    cka_filename = f"/h/vkhazaie/GenerativeSSL/rep_extraction/cka_results_{runs}.pkl"
    opd_filename = f"/h/vkhazaie/GenerativeSSL/rep_extraction/opd_results_{runs}.pkl"

    with open(cka_filename, "wb") as f:
        pickle.dump(cka_results, f)

    with open(opd_filename, "wb") as f:
        pickle.dump(opd_results, f)
