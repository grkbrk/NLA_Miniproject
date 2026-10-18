import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import sparse

#1) Load image, make it grayscale and 300 x 300
n1 = 300
n2 = 300
n = n1 * n2

image = Image.open("img-1.png").convert("L") #returns a grayscale converted copy of the image
image = image.resize((n2, n1))
X = np.asarray(image, dtype=float)/255.0 #PIL image to num array rescaled to [0,1]

#the vectorized image, vec(X)
x_true = X.reshape(-1, order="F") #-1=flatten, F=fortran order, stacks columns

#2) Regularization matrix L (discrete gradient)
def difference_matrix(m): #First-diff matrix of size (m-1) x m with rows [-1, 1]
    main_diag = -np.ones(m - 1) #-1 entries on main diag
    upper_diag = np.ones(m - 1) #1 entries on diag above it
    #sparse matrix from diagonals. Offset 0 = main, offset 1 = upper diag
    #shape -> rect. "csr" = compressed sparse row format
    return sparse.diags([main_diag, upper_diag], [0, 1], shape=(m - 1, m), format="csr") 

def difference_matrix_2d(n1, n2): #2D discrete gradient matrix L_G
    L1 = difference_matrix(n1) #vertical betw rows
    L2 = difference_matrix(n2) #between columns
    L_vertical = sparse.kron(sparse.eye(n2), L1) #I_n2 ⊗ L_1, size = (n1-1)n2*n
    L_horizontal = sparse.kron(L2, sparse.eye(n1)) #L_2 ⊗ I_n1, size = (n2-1)n1*n
    #stack the two on each other so L_G gets 2n-n1-n2 rows = 179400, 2 nonzeros in each row
    L = sparse.vstack([L_vertical, L_horizontal], format="csr")
    return(L)

L = difference_matrix_2d(n1, n2) #L_G = L, size = 179400 x 90000, 2 nonzeros in each row


#3) Blur matrix A (Gaussian, thresholded, sparse)
def gaussian_1d(m, beta, eps): #1D Gaussian blur matrix: entry (i,j) = exp(-beta*(i-j)^2).
    idx = np.arange(m) #The vector [0, 1, ..., m-1], pixel positions
    T = np.exp(-beta * (idx[:, None] - idx[None, :]) ** 2) #-beta * column * row
    T[T < eps] = 0.0 #drop small entries
    return sparse.csr_matrix(T) #stores only nonzero

def blur_matrix(n1, n2, beta, eps): #2D blur = Kronecker product of two 1D blurs
    T1 = gaussian_1d(n1, beta, eps) #T_n1
    T2 = gaussian_1d(n2, beta, eps) #T_n2
    A = sparse.kron(T2, T1, format="csr") #T_n2 ⊗ T_n1
    A.data[A.data < eps] = 0.0 #|a_ij|< eps -> a_ij = 0
    A.eliminate_zeros() #removes stored zeros

    #Normalize so every column sums to 1
    col_sums = np.asarray(A.sum(axis=0)).ravel() #sum of each col, dj = sum of a_ij
    A = A @ sparse.diags(1.0 / col_sums) #normlization, A <- AD^(-1), D=diag matrix
    return A.tocsr()

beta = 0.05
eps = 1e-3
sigma = 0.01
A = blur_matrix(n1, n2, beta, eps)

#4) Blurred and noisy data: b = Ax + e
np.random.seed(123)
e = sigma * np.random.randn(n)
b = A @ x_true + e #@ is the sparse matrix-vector product
B = b.reshape((n1, n2), order="F") #inverse of vectorization, put back into image again
X_rec = X.reshape((n1, n2), order="F") # reconstructed image

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap="gray")
plt.title("Original image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(B, cmap="gray")
plt.title("Blurred + noisy image")
plt.axis("off")
plt.show()

#5) Sparsity and memory
def print_info(M, name):
    total_entries = M.shape[0] * M.shape[1] #mn, num of entries in full dense matrix
    percent_nonzero = 100 * M.nnz / total_entries #M.nnz = only nonzeros
    dense_GB = total_entries*8 / 1024**3 #dense mem = 8mn bytes converted to gigabytes (1024^3 bytes per GB)
    sparse_MB = (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1024**2 #act size of the 3 arrays in bytes (data, indicies,indpt)

    print(f"\n--- {name} ---")
    print(f"shape:          {M.shape}")
    print(f"nonzeros:       {M.nnz}")
    print(f"% nonzero:      {percent_nonzero:.4f}%")
    print(f"dense memory:   {dense_GB:.2f} GB")
    print(f"sparse memory:  {sparse_MB:.2f} MB")
print_info(A, "A")
print_info(L, "L")

#6) Sparsity patterns
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.spy(A[:1000, :1000], markersize=1)
plt.title("Sparsity pattern of A (top-left 1000x1000)")

plt.subplot(1, 2, 2)
plt.spy(L[:1000, :1000], markersize=1)
plt.title("Sparsity pattern of L (top-left 1000x1000)")
#plt.show()
plt.savefig("sparsity_pattern.png")

#7) Effect of beta and epsilon on sparsity and cost
'''
print("beta    eps     % nonzero   matvec time (ms)")
for beta_i in [0.05, 0.1, 0.2]:
    for eps_i in [1e-2, 1e-3, 1e-4]:
        A_i = blur_matrix(n1, n2, beta_i, eps_i)
        start = time.perf_counter()
        A_i @ x_true
        elapsed_ms = 1000 * (time.perf_counter() - start)
        percent = 100 * A_i.nnz / n**2
        print(f"{beta_i:<7} {eps_i:<7.0e} {percent:<11.4f} {elapsed_ms:.1f}")
        #larger beta -> narrower blur and a sparser
        #smaller epsilon -> denser A
        #A sparse matrix-vector product costs O(nnz)
        #'''
