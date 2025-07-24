import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def vector_add(v1, v2):
    #Add two vectors element-wise
    num_elements = len(v1)
    added_vector = np.zeros(num_elements)
    for i in range(0, num_elements):
        added_vector[i] = v1[i] + v2[i]
    return added_vector

def vector_dot(v1, v2):
    #Compute dot product of two vectors
    num_elements = len(v1)
    dot_sum = 0
    for i in range(0, num_elements):
        dot_sum += v1[i] * v2[i]
    return dot_sum

def vector_norm(v, p=2):
    #Compute p-norm of a vector
    num_elements = len(v)
    norm_sum = 0
    for i in range(0, num_elements):
        norm_sum += abs(v[i])**p
    p_norm = norm_sum**(1/p)
    return p_norm

def vector_angle(v1, v2):
    #Compute angle between two vectors
    v1_norm = vector_norm(v1)
    v2_norm = vector_norm(v2)
    dot = vector_dot(v1, v2)
    angle = np.arccos(dot/(v1_norm * v2_norm))
    return angle

# Test your implementations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Addition: {vector_add(v1, v2)}")
print(f"Dot product: {vector_dot(v1, v2)}")
print(f"L2 norm of v1: {vector_norm(v1)}")
print(f"Angle between vectors: {vector_angle(v1, v2)} radians")

#------------------------------------------------------------

def matrix_multiply(A, B):
    #Multiply two matrices A and B
    dim_a = np.shape(A)
    dim_b = np.shape(B)
    m = dim_a[0]
    n = dim_a[1]
    n_check = dim_b[0]
    p = dim_b[1]

    #Check dimensions
    if(n != n_check):
        print("Dimension mismatch. Matrix multiplication not performed.")
        return None

    #Execute multiplication
    multiplied_matrix = np.zeros((m, p))
    for i in range(0, m):
        for j in range(0, p):
            row = A[i,:]
            col = B[:,j]
            multiplied_matrix[i,j] = vector_dot(row, col)
    return multiplied_matrix

def matrix_transpose(A):
    #Transpose a matrix
    dim_matrix = np.shape(A)
    m = dim_matrix[0]
    n = dim_matrix[1]
    transposed_matrix = np.zeros((n,m))
    for i in range(0, m):
        for j in range(0, n):
            transposed_matrix[j,i] = A[i,j]
    return transposed_matrix

def matrix_inverse_2x2(A):
    #Compute inverse of 2x2 matrix (if exists)
    if np.shape(A) != (2,2):
        print("Dimension mismatch. Must be 2x2 matrix.")
        return None
    
    det = A[0,0]*A[1,1]-A[1,0]*A[0,1]

    if (det == 0):
        print("0 determinant. Cannot invert matrix.")
    
    reciprocal_det = 1/det
    inverted_matrix = np.zeros((2,2))
    inverted_matrix[0,0] =  reciprocal_det * A[1,1]
    inverted_matrix[0,1] = -reciprocal_det * A[0,1]
    inverted_matrix[1,0] = -reciprocal_det * A[1,0]
    inverted_matrix[1,1] =  reciprocal_det * A[0,0]
    return inverted_matrix

def matrix_rank(A, tol=1e-10):
    #Estimate rank using SVD
    U, S, Vh = np.linalg.svd(A)
    return np.sum(S > tol)

# Test with example matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A * B = \n{matrix_multiply(A, B)}")
print(f"A^T = \n{matrix_transpose(A)}")
print(f"A^-1 = \n{matrix_inverse_2x2(A)}")
print(f"Rank of A: {matrix_rank(A)}")

#------------------------------------------------------------

def visualize_eigenvectors(A):
    """Visualize eigenvectors of 2x2 matrix"""
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(A)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Original and transformed unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    transformed_circle = A @ unit_circle
    
    ax1.plot(unit_circle[0], unit_circle[1], 'b-', label='Unit circle')
    ax1.plot(transformed_circle[0], transformed_circle[1], 'r-', label='Transformed')
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Linear Transformation')
    
    # Plot 2: Eigenvectors
    ax2.quiver(0, 0, eigenvecs[0, 0], eigenvecs[1, 0], 
               scale=1, scale_units='xy', angles='xy', color='red', 
               label=f'v1 (λ={eigenvals[0]:.2f})')
    ax2.quiver(0, 0, eigenvecs[0, 1], eigenvecs[1, 1], 
               scale=1, scale_units='xy', angles='xy', color='blue',
               label=f'v2 (λ={eigenvals[1]:.2f})')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Eigenvectors')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return eigenvals, eigenvecs

# Test with example matrix
A = np.array([[3, 1], [0, 2]])
eigenvals, eigenvecs = visualize_eigenvectors(A)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")

#------------------------------------------------------------

def low_rank_approximation(A, k):
    """Create rank-k approximation using SVD"""
    # Compute SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only top k singular values
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct approximation
    A_k = U_k @ np.diag(s_k) @ Vt_k
    
    return A_k, U_k, s_k, Vt_k

def visualize_svd_approximation(A, max_rank=None):
    """Visualize how SVD approximation improves with rank"""
    if max_rank is None:
        max_rank = min(A.shape)
    
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original matrix
    axes[0, 0].imshow(A, cmap='viridis')
    axes[0, 0].set_title('Original Matrix')
    
    # Singular values
    axes[0, 1].plot(s, 'bo-')
    axes[0, 1].set_title('Singular Values')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Value')
    
    # Approximations with different ranks
    ranks = [1, 2, max_rank//2, max_rank]
    for i, k in enumerate(ranks[:4]):
        if k <= len(s):
            A_k, _, _, _ = low_rank_approximation(A, k)
            row, col = (0, 2) if i == 0 else (1, i-1)
            axes[row, col].imshow(A_k, cmap='viridis')
            axes[row, col].set_title(f'Rank-{k} Approximation')
            
            # Compute reconstruction error
            error = np.linalg.norm(A - A_k, 'fro')
            axes[row, col].text(0.02, 0.98, f'Error: {error:.2f}', 
                               transform=axes[row, col].transAxes, 
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Example: Create a low-rank matrix and test approximation
np.random.seed(42)
rank = 3
m, n = 50, 40
A_true = np.random.randn(m, rank) @ np.random.randn(rank, n)
A_noisy = A_true + 0.1 * np.random.randn(m, n)

visualize_svd_approximation(A_noisy, max_rank=10)