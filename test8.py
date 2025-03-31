import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from PIL import Image
import matplotlib.pyplot as plt


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
            c = calculate_diffusion_coefficient(phi, K, h)

        # 2. Construct the Matrix A(phi)
        A,b = construct_matrix_A(phi, phi0 , c , lambda_val, h, nx, ny,dt)

        # 3. Construct the right-hand side vector f
        # f = phi.flatten() + dt * lambda_val * phi0.flatten()  # Correct RHS

        # 4. Apply Implicit Euler update
        # Solve (I - dt*A) phi_new = phi + dt*f

        # A_imp = A 
        phi = splinalg.spsolve(A, b).reshape(nx, ny)

        summed_per_time.append(np.sum(phi) / np.size(phi))
        error_per_time.append(np.std(phi))

    return phi, summed_per_time, error_per_time


def calculate_diffusion_coefficient(phi, K, h):
    """Calculates the diffusion coefficient c based on the gradient magnitude."""
    nx, ny = phi.shape
    c = np.zeros_like(phi)

    # Calculate gradient using central differences, handling boundaries
    phi_x = np.zeros_like(phi)
    phi_y = np.zeros_like(phi)

    # Central differences for internal points
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * h)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * h)

    # Neumann boundary conditions: du/dn = 0
    # Use one-sided differences consistent with the boundary condition
    phi_x[:, 0] = (phi[:, 1] - phi[:, 0]) / h  # Left boundary
    phi_x[:, -1] = (phi[:, -1] - phi[:, -2]) / h  # Right boundary
    phi_y[0, :] = (phi[1, :] - phi[0, :]) / h  # Top boundary
    phi_y[-1, :] = (phi[-1, :] - phi[-2, :]) / h  # Bottom boundary

    G = np.sqrt(phi_x**2 + phi_y**2)
    c = 1 / (1 + (G / K)**2)
    return c


def construct_matrix_A(phi,phi0, c, lambda_val, h, nx, ny,dt):
    """Constructs the sparse matrix A using FVM discretization with Neumann BC."""
    N = nx * ny
    b = np.zeros(N)
    A = sparse.lil_matrix((N, N))

    for i in range(nx):
        for j in range(ny):
            k = j * nx + i  # 1D index


            # Diffusion coefficients at cell faces (averaging)
            c_east = (c[i, j] + c[min(i + 1, nx - 1), j]) / 2
            c_west = (c[i, j] + c[max(i - 1, 0), j]) / 2
            c_north = (c[i, j] + c[i, min(j + 1, ny - 1)]) / 2
            c_south = (c[i, j] + c[i, max(j - 1, 0)]) / 2

            # Diagonal element
            A[k, k] = (1 + dt * lambda_val) +  dt * (c_east + c_west + c_north + c_south) / h**2 

            # Off-diagonal elements (neighbors)
            if i < nx - 1:  # East neighbor
                A[k, k + 1] = -dt*c_east / h**2
            if i > 0:  # West neighbor
                A[k, k - 1] = -dt*c_west / h**2
            if j < ny - 1:  # North neighbor
                A[k, k + nx] = -dt*c_north / h**2
            if j > 0:  # South neighbor
                A[k, k - nx] = -dt* c_south / h**2
                
            b[k] = phi[i,j] + dt * lambda_val * phi0[i,j]  # Correct RHS
                
            
    
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
    lambda_val = 10000
    K = 5
    h = 1 / (nx - 1)  # Grid spacing

    # Choose a suitable dt
    dt = 1e-5  # Larger dt for stationary solution
    n_steps = 40  # Adjust n_steps accordingly

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
        
    # Test diffusioin coefficient calculation
    c_test = calculate_diffusion_coefficient(phi0, K, h)
    plt.figure()
    plt.imshow(c_test, cmap='gray')
    plt.title('Diffusion Coefficient')
    plt.axis('off')
    plt.show()
    
    print(c_test.shape)