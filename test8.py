import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from PIL import Image
import matplotlib.pyplot as plt
# caching tool
from joblib import Memory
from joblib import Parallel, delayed

# Initialize joblib Memory cache
memory = Memory(location='g:\\GitHub\\Numerical-Methods-2\\joblib_cache', verbose=0)

def implicit_euler_anisotropic_diffusion(phi0, lambda_val, K, h, dt, n_steps, c_const=False):
    """
    Applies anisotropic diffusion filtering using Implicit Euler method.

    Args:
        phi0: The initial image (numpy array).
        lambda_val: The fidelity parameter (lambda).
        K: The edge-stopping parameter (K).
        h: Grid spacing.
        dt: Time step size.
        n_steps: Number of time steps.
        c_const: Boolean to determine if c is constant or not

    Returns:
        phi: The filtered image (numpy array).
        summed_per_time: List of average pixel values at each time step.
        error_per_time: List of standard deviations at each time step.
    """

    nx, ny = phi0.shape
    N = nx * ny
    phi = phi0.copy()
    summed_per_time = []
    error_per_time = []

    for n in range(n_steps):
        # 1. Calculate c(phi)
        if c_const:
            c = np.ones_like(phi)
        else:
            c = None

        # 2. Construct the Matrix A(phi)
        A,b = construct_matrix_A(phi, phi0 , c , lambda_val, h, nx, ny,dt, K)

        # 3. Construct the right-hand side vector f
        # f = phi.flatten() + dt * lambda_val * phi0.flatten()  # Correct RHS

        # 4. Apply Implicit Euler update
        # Solve (I - dt*A) phi_new = phi + dt*f

        # A_imp = A 
        phi = splinalg.spsolve(A, b).reshape(nx, ny)

        summed_per_time.append(np.sum(phi) / np.size(phi))
        error_per_time.append(np.std(phi))

    return phi, summed_per_time, error_per_time

# Replace the previous caching decorator with joblib's memory.cache
@memory.cache
def construct_matrix_A(phi, phi0, c_in, lambda_val, h, nx, ny, dt, K):
    N = nx * ny
    b = np.zeros(N)
    A = sparse.lil_matrix((N, N))

    # Helper function to compute diffusion coefficient at a point (i,j)
    def compute_c(i, j):
        if c_in is not None:
            return c_in[i, j]
        # Compute horizontal gradient
        if j == 0:
            gx = (phi[i, j + 1] - phi[i, j]) / h
        elif j == ny - 1:
            gx = (phi[i, j] - phi[i, j - 1]) / h
        else:
            gx = (phi[i, j + 1] - phi[i, j - 1]) / (2 * h)
        # Compute vertical gradient
        if i == 0:
            gy = (phi[i + 1, j] - phi[i, j]) / h
        elif i == nx - 1:
            gy = (phi[i, j] - phi[i - 1, j]) / h
        else:
            gy = (phi[i + 1, j] - phi[i - 1, j]) / (2 * h)
        G = np.sqrt(gx**2 + gy**2)
        return 1 / (1 + (G / K) ** 2)

    # Loop over each grid cell
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j  # 1D index (row-major order)
            c_center = compute_c(i, j)
            # Compute neighbor coefficients by averaging with current pixel
            if i < nx - 1:
                c_south = (c_center + compute_c(i + 1, j)) / 2
            else:
                c_south = c_center
            if i > 0:
                c_north = (c_center + compute_c(i - 1, j)) / 2
            else:
                c_north = c_center
            if j < ny - 1:
                c_east = (c_center + compute_c(i, j + 1)) / 2
            else:
                c_east = c_center
            if j > 0:
                c_west = (c_center + compute_c(i, j - 1)) / 2
            else:
                c_west = c_center

            # Diagonal element
            A[k, k] = (1 + dt * lambda_val) + dt * (c_east + c_west + c_north + c_south) / h**2
            # Off-diagonal elements
            if i < nx - 1:  # South neighbor
                A[k, k + ny] = -dt * c_south / h**2
            if i > 0:  # North neighbor
                A[k, k - ny] = -dt * c_north / h**2
            if j < ny - 1:  # East neighbor
                A[k, k + 1] = -dt * c_east / h**2
            if j > 0:  # West neighbor
                A[k, k - 1] = -dt * c_west / h**2

            b[k] = phi[i, j] + dt * lambda_val * phi0[i, j]  # Correct RHS

    if A is None or b is None:
        raise ValueError("Matrix A or vector b is not properly constructed.")
    return A.tocsc(), b  # Convert to CSC format


def calculate_total_energy(phi, h):
    """Calculates the total energy of the image."""
    return np.sum(phi) * h**2


# Example Usage:
if __name__ == '__main__':

    # Load the images
    Im = Image.open('SL_simulated.gif')
    fm = np.array([])
    fn = np.array(Im) / 255.0
    nx, ny = fn.shape

    fm = fn  # Original image (noiseless)
    sigma = 0.1
    np.random.seed(0)
    fn = fm + sigma * np.random.randn(nx, ny)  # Noisy image

    phi0 = fn.copy()  # Initial image (noisy)

    # Set parameters
    lambda_val = 2245.5709610369454
    K = 5
    h = 1 / (nx - 1)  # Grid spacing

    # Choose a suitable dt
    dt =  1.0941925152053495e-06  # Larger dt for stationary solution
    n_steps = 500  # Adjust n_steps accordingly

    # Apply anisotropic diffusion filtering
    phi_filtered, summed_per_time, error_per_time = implicit_euler_anisotropic_diffusion(
        phi0, lambda_val, K, h, dt, n_steps, c_const=False)
    phi_filtered_const, summed_per_time_const, error_per_time_const = implicit_euler_anisotropic_diffusion(
        phi0, lambda_val, K, h, dt, n_steps, c_const=True)

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
    fs = phi_filtered.copy()
    sum_x = summed_per_time.copy()
    err_x = error_per_time.copy()

    sum_x_const = summed_per_time_const.copy()
    err_x_const = error_per_time_const.copy()

    # Plot the sum of the average pixel values:
    t = np.arange(0, (len(sum_x)) * dt, dt)
    plt.figure()
    plt.plot(t, sum_x, '-x', label='Variable c')
    plt.plot(t, sum_x_const, '-o', label='Constant c=1')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.title('Sum of pixel values, Implicit Euler')
    plt.xlabel('Time')
    plt.xlim(left=0)
    plt.ylabel('Σ x/N')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Implicit Euler (Variable c)')
    plt.imshow(fs, extent=[0, 1, 0, 1], cmap='gray')
    plt.axis('square')
    plt.axis('off')
    plt.show()

    if fm.size > 0:
        plt.figure()
        t = np.arange(0, (len(err_x)) * dt, dt)
        plt.plot(t, err_x, '-x', label='Variable c')
        plt.plot(t, err_x_const, '-o', label='Constant c=1')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.title('Standard deviation versus time, Implicit Euler')
        plt.xlabel('Time')
        plt.xlim(left=0)
        plt.ylabel('σ')
        plt.legend()
        plt.show()

