import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt  # Import matplotlib
from PIL import Image

fm = Image.open('SL_simulated.gif')
def picard_iteration(phi0, lambda_val, K, h, epsilon=1e-5, max_iter=100, c_const = False):
    """
    Applies anisotropic diffusion filtering using Picard iteration.

    Args:
        phi0: The initial image (numpy array).
        lambda_val: The fidelity parameter (lambda).
        K: The edge-stopping parameter (K).
        h: Grid spacing.
        epsilon: Convergence tolerance.
        max_iter: Maximum number of Picard iterations.
	c_const: Boolean to determine if c is constant or not

    Returns:
        phi: The filtered image (numpy array).
    """

    nx, ny = phi0.shape
    N = nx * ny
    phi = phi0.copy()  # Initialize phi with the original image
    u_prev = phi.copy()

    for k in range(max_iter):
        u_prev = phi.copy()  # Store the previous solution
        # 1. Calculate c(phi)
        if c_const:
            c = np.ones_like(phi)
        else:
            c = calculate_diffusion_coefficient(phi, K, h)

        # 2. Construct the Matrix A(phi)
        A = construct_matrix_A(phi, c, lambda_val, h, nx, ny)

        # 3. Construct the right-hand side vector f
        f = -lambda_val * (h**2) * phi0.flatten()

        # 4. Solve the Linear System: -A(phi) u = f
        phi = splinalg.spsolve(-A, f).reshape(nx, ny)

        # 5. Check for Convergence
        error = np.linalg.norm(phi - u_prev) / np.sqrt(N)
        print(f"Iteration {k+1}: Error = {error}")

        if error < epsilon:
            print(f"Picard iteration converged after {k+1} iterations.")
            break

        u_prev = phi.copy()  # Update previous solution

    else:
        print("Picard iteration did not converge within the maximum number of iterations.")

    cur_sigma  = np.linalg.norm(fm - phi, ord='fro') / np.sqrt(N)
    return phi, cur_sigma


def calculate_diffusion_coefficient(phi, K, h):
    """Calculates the diffusion coefficient c based on the gradient magnitude."""
    nx, ny = phi.shape
    c = np.zeros_like(phi)

    # Calculate gradient using central differences, handling boundaries
    phi_x = np.zeros_like(phi)
    phi_y = np.zeros_like(phi)

    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * h)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * h)

    # Boundary conditions (Neumann - zero flux)
    phi_x[:, 0] = (phi[:, 1] - phi[:, 0]) / h  # Left boundary
    phi_x[:, -1] = (phi[:, -1] - phi[:, -2]) / h  # Right boundary
    phi_y[0, :] = (phi[1, :] - phi[0, :]) / h  # Top boundary
    phi_y[-1, :] = (phi[-1, :] - phi[-2, :]) / h  # Bottom boundary

    G = np.sqrt(phi_x**2 + phi_y**2)
    c = 1 / (1 + (G / K)**2)
    # print(c)
    return c


def construct_matrix_A(phi, c, lambda_val, h, nx, ny):
    """Constructs the sparse matrix A using FVM discretization."""
    N = nx * ny
    A = sparse.lil_matrix((N, N))

    # Create a meshgrid of indices
    row_ind, col_ind = np.indices((nx, ny))

    for i in range(nx):
        for j in range(ny):
            k = j * nx + i  # 1D index

            # Diffusion coefficients at cell faces (using averaging from hand-in J)
            c_east = (c[i,j] + c[(i+1)%nx, j]) / 2
            c_west = (c[i,j] + c[(i-1)%nx, j]) / 2
            c_north = (c[i,j] + c[i, (j+1)%ny]) / 2
            c_south = (c[i,j] + c[i, (j-1)%ny]) / 2

            # Diagonal element
            A[k, k] = (c_east + c_west + c_north + c_south) / h**2 + lambda_val

            # Off-diagonal elements (neighbors)
            if i < nx - 1:  # East neighbor
                A[k, k + 1] = -c_east / h**2
            else: #Periodic Boundary Condition
                A[k, k - (nx-1)] = -c_east / h**2

            if i > 0:  # West neighbor
                A[k, k - 1] = -c_west / h**2
            else: #Periodic Boundary Condition
                A[k, k + (nx-1)] = -c_west / h**2

            if j < ny - 1:  # North neighbor
                A[k, k + nx] = -c_north / h**2
            else: #Periodic Boundary Condition
                A[k, k - (nx*(ny-1))] = -c_north / h**2

            if j > 0:  # South neighbor
                A[k, k - nx] = -c_south / h**2
            else: #Periodic Boundary Condition
                A[k, k + (nx*(ny-1))] = -c_south / h**2

    return A.tocsc()  # Convert to CSC format for efficient sparse linear algebra


# Example Usage:
if __name__ == '__main__':

    # Load the images
    Im = Image.open('SL_measured.gif')
    fn = np.array(Im)/255.0
    nx, ny = fn.shape
    # dither = Image.effect_noise((nx,ny), 0.1)
    # fn = np.array(Im)/255 + np.array(dither)
    fm = np.array([])

    fm = fn  # Original image (noiseless)
    sigma = 0.1
    np.random.seed(0)
    # fn = fm + sigma * np.random.randn(nx, ny) # Noisy image

    phi0 = fn.copy() # np.clip(fn.copy(), 0, 1)  # Initial image (noisy)

    # Set parameters
    lambda_val = 10000
    K = 5
    h = 1 / (nx - 1)  # Grid spacing
    # Define missing variables
    epsilon = 1e-10
    max_iter = 100


    # Apply anisotropic diffusion filtering
    phi_filtered,sig = picard_iteration(phi0, lambda_val, K, h, epsilon = epsilon, max_iter = max_iter, c_const = False)
    phi_filtered_const,sigc = picard_iteration(phi0, lambda_val, K, h,  epsilon = epsilon, max_iter = max_iter, c_const = True)

    # Display the original and filtered images (requires matplotlib)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(phi0, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(phi_filtered, cmap='gray')
    plt.title('Filtered Image (c, edge preserved) ')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(phi_filtered_const, cmap='gray')
    plt.title('Filtered Image (c = 1)')
    plt.axis('off')

    plt.show()