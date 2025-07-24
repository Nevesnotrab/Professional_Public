import numpy as np
import matplotlib.pyplot as plt

"""Excersise 1"""
"""Implement a function to check if a set of vectors is linearly independent."""

def CheckLinearIndependence(A, tol=1.0e-10):
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > tol)
    num_vectors = A.shape[1]
    return rank == num_vectors

"""Exercise 2"""
"""Create a visualization showing how matrix multiplication transforms the unit square."""
def PlotMatrixMultiplicationOfUnitSquare(B):
    UnitSquare = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    TransformedSquare = np.matmul(UnitSquare, B)
    plt.plot(UnitSquare[:,0],UnitSquare[:,1])
    plt.show()

"""Excercise 3"""
"""Implement the Gram-Schmidt process for orthogonalization."""
def GramSchmidt(v1, v2):
    v1_mag = np.linalg.norm(v1)
    e1 = v1/v1_mag
    u2 = v2 - np.dot(v2,e1)*e1
    u2_mag = np.linalg.norm(u2)
    e2 = u2/u2_mag
    return e1, e2

"""Excercise 4"""
"""Build a simple PCA implemenetation using only eigendecomposition."""
def SimplePCA(A):
    A = A - np.mean(A, axis = 0)
    #Step 1 - Compute the covariance matrix
    n = np.shape(A)[0]
    AT = np.transpose(A)
    C = (1/(n-1))*(AT @ A)

    #Step 2 - Eigendecomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(C)

    #Step 2a - Order the eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]

    Q = eigenvectors #Redundant, but stated for readability
    Q = Q[:,idx]
    #capital_lambda = np.diag(eigenvalues) #Unused. Commented out.
    #Q_inv = np.linalg.inv(Q) #Unused. Commented out.

    #Step 3 - Project the data
    XPCA = A @ Q
    return XPCA

"""Excercise 5"""
"""Use SVD to implement a basic image compression algorithm."""
def BasicImageCompression(A, k):
    #Compute SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    #Truncate values after k
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    #Reconstruct
    A_k = U_k @ np.diag(s_k) @ Vt_k
    return A_k