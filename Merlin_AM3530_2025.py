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


# def cij(i,j,phi = fn,h=1.0):
#     """
#     Calculates c_{i,j} based on the provided formula.

#     Args:
#         phi (numpy.ndarray): A 2D numpy array representing the phi values.
#         i (int): The i index.
#         j (int): The j index.
#         hx (float): The grid spacing in the x direction.
#         hy (float): The grid spacing in the y direction.

#     Returns:
#         float: The calculated c_{i,j} value. Returns np.nan if a division by zero occurs or index is out of bounds.
#     """
#     try:
#         # Calculate the terms inside the denominator
#         term1 = ((phi[i+1, j] - phi[i-1, j]) / (2 * h))**2
#         term2 = ((phi[i, j+1] - phi[i, j-1]) / (2 * h))**2

#         # Calculate c_ij
#         cij = 25 / (25 + term1 + term2)
#         return cij

#     except IndexError:
#         print(f"Warning: Index out of bounds at i={i}, j={j}. Returning NaN.")
#         return np.nan  # Handle edge cases where indices are out of bounds
#     except ZeroDivisionError:
#         print("Error: Division by zero encountered. Returning NaN.")
#         return np.nan     

# def construct_matrix_A(phi, h, lambda_val, c):
#     """
#     Constructs the matrix A for the Finite Volume Method discretization of
#     the equation 0 = div(c grad(phi)) + lambda (phi_0 - phi)
#     with Neumann boundary conditions.

#     Args:
#         fs (array-like): the input image
#         h (float): Grid spacing.
#         lambda_val (float): Fidelity parameter lambda.
#         c (array-like): Diffusion coefficient.  Can be a constant or a 2D array.
#                          If a constant, it's assumed to be uniform.  If a 2D array,
#                          it should have shape (Nx, Ny).

#     Returns:
#         scipy.sparse.lil_matrix: The matrix A in LIL format.
#     """
#     Nx,Ny = phi.shape
#     N = Nx * Ny  # Total number of grid points
#     A = lil_matrix((N, N))  # Initialize A as a sparse matrix (LIL format for efficient modification)

#     # Helper function to map 2D indices (i, j) to a 1D index k
#     def k_index(i, j):
#         return j * Nx + i

#     # # Determine if c is uniform or a 2D array
#     # if np.isscalar(c):
#     #     c_uniform = True
#     #     c_val = c  # Store the uniform c value
#     # else:
#     #     c_uniform = False
#     #     c = np.asarray(c)  # Ensure c is a NumPy array
#     #     if c.shape != (Nx, Ny):
#     #         raise ValueError("The shape of c must be (Nx, Ny)")
        
    

#     for i in range(Nx):
#         for j in range(Ny):
#             k = k_index(i, j)  # 1D index for the current grid point (i, j)

#             # # Diffusion coefficients at cell faces (use c_val if c is uniform)
#             # if c_uniform:
#             #     c_ip12_j = c_val  # c_{i+1/2, j}
#             #     c_im12_j = c_val  # c_{i-1/2, j}
#             #     c_i_jp12 = c_val  # c_{i, j+1/2}
#             #     c_i_jm12 = c_val  # c_{i, j-1/2}
#             # else:
#             # right
#             c_ip12_j = (cij(i,j) +cij(i+1,j))/2
#             # left
#             c_im12_j = (cij(i,j) +cij(i-1,j))/2
#             # top
#             c_i_jp12 = (cij(i,j) + cij(i,j+1))/2
#             # bottom
#             c_i_jm12 = (cij(i,j) + cij(i,j-1))/2

#             # Diagonal element (A_kk)
#             if i == 0:  # Left boundary
#                 A[k, k] = (2 * c_im12_j + c_i_jp12 + c_i_jm12) / h**2 + lambda_val
#             elif i == Nx - 1:  # Right boundary
#                 A[k, k] = (c_ip12_j + c_im12_j + c_i_jp12 + c_i_jm12) / h**2 + lambda_val
#             elif j == 0:  # Bottom boundary
#                 A[k, k] = (c_ip12_j + c_im12_j + 2 * c_i_jm12) / h**2 + lambda_val
#             elif j == Ny - 1:  # Top boundary
#                 A[k, k] = (c_ip12_j + c_im12_j + 2 * c_i_jp12) / h**2 + lambda_val
#             else:  # Interior points
#                 A[k, k] = (c_ip12_j + c_im12_j + c_i_jp12 + c_i_jm12) / h**2 + lambda_val

#             # Off-diagonal elements
#             if i < Nx - 1:  # Right neighbor
#                 A[k, k_index(i + 1, j)] = -c_ip12_j / h**2
#             if i > 0:  # Left neighbor
#                 A[k, k_index(i - 1, j)] = -c_im12_j / h**2
#             if j < Ny - 1:  # Top neighbor
#                 A[k, k_index(i, j + 1)] = -c_i_jp12 / h**2
#             if j > 0:  # Bottom neighbor
#                 A[k, k_index(i, j - 1)] = -c_i_jm12 / h**2

#     return A

# # Example usage:
# Nx = 10
# Ny = 10
# h = 1.0
# lambda_val = 0.1
# c = 1.0  # Uniform diffusion coefficient

# # A = construct_matrix_A(Nx, Ny, h, lambda_val, c)

# # print(A.toarray()) # Convert to a dense array for printing (for small matrices)



# Im = Image.open(gif_name)

# fm = np.array([])
# fn = np.array(Im)/255.0
# nx, ny = fn.shape

# if gif_name == 'SL_simulated.gif':
#     fm = fn
#     sigma = 0.1
#     np.random.seed(0)
#     fn = fm + sigma * np.random.randn(nx, ny)
#     # Plot the model image
#     plt.figure()
#     plt.title('Noise-free image')
#     plt.imshow(fm, extent=[0, 1, 0, 1], cmap = 'gray')
#     plt.axis('square')
#     plt.axis('off')


import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

def anisotropic_diffusion_picard(phi0, lambda_val, K, h, epsilon=1e-5, max_iter=100):
    """
    Applies anisotropic diffusion filtering using Picard iteration with FVM and Neumann BCs.

    Args:
        phi0: The initial image (numpy array).
        lambda_val: The fidelity parameter (lambda).
        K: The edge-stopping parameter (K).
        h: Grid spacing.
        epsilon: Convergence tolerance.
        max_iter: Maximum number of Picard iterations.

    Returns:
        phi: The filtered image (numpy array).
    """

    nx, ny = phi0.shape
    N = nx * ny
    phi = phi0.copy()  # Initialize phi with the original image
    u_prev = phi.copy()

    for k in range(max_iter):
        # 1. Calculate c(u^k)
        # 2. Construct the Matrix A(u^k)
        A = construct_matrix_A(fn, lambda_val, h)

        # 3. Construct the right-hand side vector f
        f = -lambda_val * h**2 * phi0.flatten()  # Corrected sign and h^2

        # 4. Solve the Linear System: -A(u^k) u^{k+1} = f
        u = splinalg.spsolve(A, f).reshape(nx, ny)
        phi = u.copy()

        # 5. Check for Convergence
        error = np.linalg.norm(phi - u_prev) / np.sqrt(N)
        print(f"Iteration {k+1}: Error = {error}")

        if error < epsilon:
            print(f"Picard iteration converged after {k+1} iterations.")
            break

        u_prev = phi.copy() # Update previous solution

    else:
        print("Picard iteration did not converge within the maximum number of iterations.")

    return phi


# def cij( K, h, phi = fn):
#     """Calculates the diffusion coefficient c based on the gradient magnitude."""
#     nx, ny = phi.shape
#     c = np.zeros_like(phi)

#     # Calculate gradient using central differences, handling boundaries
#     phi_x = np.zeros_like(phi)
#     phi_y = np.zeros_like(phi)

#     phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * h)
#     phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * h)

#     # Boundary conditions (Neumann - zero flux)
#     phi_x[:, 0] = (phi[:, 1] - phi[:, 0]) / h  # Left boundary
#     phi_x[:, -1] = (phi[:, -1] - phi[:, -2]) / h # Right boundary
#     phi_y[0, :] = (phi[1, :] - phi[0, :]) / h  # Top boundary
#     phi_y[-1, :] = (phi[-1, :] - phi[-2, :]) / h # Bottom boundary

#     G = np.sqrt(phi_x**2 + phi_y**2)
#     c = 1 / (1 + (G / K)**2)
#     return c

import numpy as np

def cij(i, j,phi=fn, K=5.0, h=1.0):
    """
    Calculates c_{i,j} ,based on the formulas in 'Hand in j'.

    Args:
        phi (numpy.ndarray): A 2D numpy array representing the phi values (image).
        i (int): The row index of the pixel.
        j (int): The column index of the pixel.
        K (float): The edge-stopping parameter.
        h (float): The grid spacing.

    Returns:
        float: The calculated c_{i,j} value.  Returns np.nan if an error occurs.
    """
    try:
        # Calculate the gradient components using central differences
        phi_x = (phi[i+1, j] - phi[i-1, j]) / (2 * h)
        phi_y = (phi[i, j+1] - phi[i, j-1]) / (2 * h)

        # Calculate the gradient magnitude
        G_ij = np.sqrt(phi_x**2 + phi_y**2)

        # Calculate c_ij
        cij_calc = 1 / (1 + (G_ij / K)**2)
        return cij_calc

    except IndexError:
        print(f"Warning: Index out of bounds at i={i}, j={j}. Returning NaN.")
        return np.nan  # Handle edge cases
    except ZeroDivisionError:
        print("Error: Division by zero encountered. Returning NaN.")
        return np.nan # Handle potential division by zero

    
def construct_matrix_A(image, lambda_val:float , h=1.0):
    """Constructs the sparse matrix A using FVM discretization with Neumann BCs."""
    nx, ny = image.shape
    N = nx * ny
    A = sparse.lil_matrix((N, N))

    for i in range(nx-2):
        for j in range(ny-2):
            k = j * nx + i  # 1D index

            # Diffusion coefficients at cell faces (averaging)
            if i < nx - 1:
                c_east = (cij(i, j) + cij(i + 1, j)) / 2
            else:
                c_east = cij(i, j)  # Neumann BC: use value at the boundary

            if i > 0:
                c_west = (cij(i, j) + cij(i - 1, j)) / 2
            else:
                c_west = cij(i, j)  # Neumann BC: use value at the boundary

            if j < ny - 1:
                c_north = (cij(i, j) + cij(i, j + 1)) / 2
            else:
                c_north = cij(i, j)  # Neumann BC: use value at the boundary

            if j > 0:
                c_south = (cij(i, j) + cij(i, j - 1)) / 2
            else:
                c_south = cij(i, j)  # Neumann BC: use value at the boundary

            # Diagonal element (FVM discretization with Neumann BCs)
            A[k, k] = (c_east + c_west + c_north + c_south) / h**2 + lambda_val

            # Off-diagonal elements (neighbors)
            if i < nx - 1:  # East neighbor
                A[k, (j * nx) + (i + 1)] = -c_east / h**2
            if i > 0:  # West neighbor
                A[k, (j * nx) + (i - 1)] = -c_west / h**2
            if j < ny - 1:  # North neighbor
                A[k, ((j + 1) * nx) + i] = -c_north / h**2
            if j > 0:  # South neighbor
                A[k, ((j - 1) * nx) + i] = -c_south / h**2

    return A.tocsc()  # Convert to CSC format for efficient sparse linear algebra


# Example Usage:
# Picard iteration
method = 1
plt.figure(figsize=(8,6))
plt.suptitle('Picard iteration')
sigma = np.zeros(9)
lambdas_vol_meth =  np.linspace(1, 8, num = 8)
for i,ele in enumerate(lambdas):
    fidelity = 10**(i)
    Nx,Ny = fn.shape
    h = 1.0
    lambda_val = ele
    c = 1.0  
    # Add your code here. At this moment the filtered image is just a copy of the original image
    A = construct_matrix_A(fn, h, lambda_val)
    b = lambda_val * fn.flatten()  # Right-hand side vector
    fs = spsolve(A, b).reshape(Nx, Ny)  # Solve the linear systm
    
    
    plt.subplot(3, 3, i+1)
    plt.title(f'Picard, $\lambda$ = {ele}')
    plt.imshow(fs, extent=[0, 1, 0, 1], cmap = 'gray')
    plt.axis('square')
    plt.axis('off')

    if fm.size > 0:
        sigma[i] = np.linalg.norm(fs - fm, ord='fro') / np.sqrt(nx*ny)

plt.show()

if fm.size > 0:
    plt.figure()
    plt.title('Standard deviation versus fidelity')
    plt.plot(range(9), sigma)
    plt.xlabel('Logarithm Fidelity')
    plt.ylabel('$\sigma$')



plt.show()