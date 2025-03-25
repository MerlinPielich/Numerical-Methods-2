import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from skimage.data import shepp_logan_phantom

# Choose the image:
gif_name = 'SL_simulated.gif'
# gif_name = 'SL_measured.gif'

# Time step:
dt = 1.e-5
nt = 40

Im = Image.open(gif_name)

fm = np.array([])
fn = np.array(Im)/255.0
nx, ny = fn.shape


# Image setup: Use Shepp-Logan phantom as the ground truth image
fm = shepp_logan_phantom()
fm = fm / np.max(fm)  # Normalize the image to [0, 1]
nx, ny = fm.shape

# Noise setup
sigma = 0.1
np.random.seed(0)
fn = fm + sigma * np.random.randn(nx, ny)  # Add noise

# Fidelity parameter values
lambdas = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

# Grid spacing
h = 1.0 / nx

# Prepare the solution vector (flattened version of the noisy image)
phi = fn.flatten()
from scipy.sparse import lil_matrix
def construct_matrix_A(Nx, Ny, h, lambda_val, c):
    """
    Constructs the matrix A for the Finite Volume Method discretization of
    the equation 0 = div(c grad(phi)) + lambda (phi_0 - phi)
    with Neumann boundary conditions.

    Args:
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        h (float): Grid spacing.
        lambda_val (float): Fidelity parameter lambda.
        c (array-like): Diffusion coefficient.  Can be a constant or a 2D array.
                         If a constant, it's assumed to be uniform.  If a 2D array,
                         it should have shape (Nx, Ny).

    Returns:
        scipy.sparse.lil_matrix: The matrix A in LIL format.
    """

    N = Nx * Ny  # Total number of grid points
    A = lil_matrix((N, N))  # Initialize A as a sparse matrix (LIL format for efficient modification)

    # Helper function to map 2D indices (i, j) to a 1D index k
    def k_index(i, j):
        return j * Nx + i

    # Determine if c is uniform or a 2D array
    if np.isscalar(c):
        c_uniform = True
        c_val = c  # Store the uniform c value
    else:
        c_uniform = False
        c = np.asarray(c)  # Ensure c is a NumPy array
        if c.shape != (Nx, Ny):
            raise ValueError("The shape of c must be (Nx, Ny)")

    for i in range(Nx):
        for j in range(Ny):
            k = k_index(i, j)  # 1D index for the current grid point (i, j)

            # Diffusion coefficients at cell faces (use c_val if c is uniform)
            if c_uniform:
                c_ip12_j = c_val  # c_{i+1/2, j}
                c_im12_j = c_val  # c_{i-1/2, j}
                c_i_jp12 = c_val  # c_{i, j+1/2}
                c_i_jm12 = c_val  # c_{i, j-1/2}
            else:
                # Use the provided c values.  For simplicity, we'll just use c[i,j]
                # as an approximation for the cell face values.  More accurate
                # approximations (e.g., averaging neighboring c values) are possible.
                c_ip12_j = c[i, j]
                c_im12_j = c[i, j]
                c_i_jp12 = c[i, j]
                c_i_jm12 = c[i, j]

            # Diagonal element (A_kk)
            if i == 0:  # Left boundary
                A[k, k] = (2 * c_im12_j + c_i_jp12 + c_i_jm12) / h**2 + lambda_val
            elif i == Nx - 1:  # Right boundary
                A[k, k] = (c_ip12_j + c_im12_j + c_i_jp12 + c_i_jm12) / h**2 + lambda_val
            elif j == 0:  # Bottom boundary
                A[k, k] = (c_ip12_j + c_im12_j + 2 * c_i_jm12) / h**2 + lambda_val
            elif j == Ny - 1:  # Top boundary
                A[k, k] = (c_ip12_j + c_im12_j + 2 * c_i_jp12) / h**2 + lambda_val
            else:  # Interior points
                A[k, k] = (c_ip12_j + c_im12_j + c_i_jp12 + c_i_jm12) / h**2 + lambda_val

            # Off-diagonal elements
            if i < Nx - 1:  # Right neighbor
                A[k, k_index(i + 1, j)] = -c_ip12_j / h**2
            if i > 0:  # Left neighbor
                A[k, k_index(i - 1, j)] = -c_im12_j / h**2
            if j < Ny - 1:  # Top neighbor
                A[k, k_index(i, j + 1)] = -c_i_jp12 / h**2
            if j > 0:  # Bottom neighbor
                A[k, k_index(i, j - 1)] = -c_i_jm12 / h**2

    return A

# Example usage:
Nx = 10
Ny = 10
h = 1.0
lambda_val = 0.1
c = 1.0  # Uniform diffusion coefficient

A = construct_matrix_A(Nx, Ny, h, lambda_val, c)

print(A.toarray()) # Convert to a dense array for printing (for small matrices)



Im = Image.open(gif_name)

fm = np.array([])
fn = np.array(Im)/255.0
nx, ny = fn.shape

if gif_name == 'SL_simulated.gif':
    fm = fn
    sigma = 0.1
    np.random.seed(0)
    fn = fm + sigma * np.random.randn(nx, ny)
    # Plot the model image
    plt.figure()
    plt.title('Noise-free image')
    plt.imshow(fm, extent=[0, 1, 0, 1], cmap = 'gray')
    plt.axis('square')
    plt.axis('off')


# Picard iteration
method = 1
plt.figure(figsize=(8,6))
plt.suptitle('Picard iteration')
sigma = np.zeros(9)
lambdas_vol_meth =  np.linspace(0, 8, num = 8)
for i,ele in enumerate(lambdas_vol_meth):
    fidelity = 10**(i)
    Nx,Ny = fn.shape
    h = 1.0
    lambda_val = ele
    c = 1.0  
    # Add your code here. At this moment the filtered image is just a copy of the original image
    A = construct_matrix_A(Nx, Ny, h, lambda_val, c)
    b = lambda_val * fn.flatten()  # Right-hand side vector
    fs = spsolve(A, b).reshape(Nx, Ny)  # Solve the linear systm
    
    
    plt.subplot(3, 3, i+1)
    plt.title(f'Picard, $\lambda$ = {ele}')
    plt.imshow(fs, extent=[0, 1, 0, 1], cmap = 'gray')
    plt.axis('square')
    plt.axis('off')

    if fm.size > 0:
        sigma[i] = np.linalg.norm(fs - fm, ord='fro') / np.sqrt(nx*ny)

if fm.size > 0:
    plt.figure()
    plt.title('Standard deviation versus fidelity')
    plt.plot(range(9), sigma)
    plt.xlabel('Logarithm Fidelity')
    plt.ylabel('$\sigma$')

