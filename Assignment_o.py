import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt  # Import matplotlib
from PIL import Image

def implicit_euler_anisotropic_diffusion(phi0, lambda_val, K, h, dt, n_steps, c_const = False):
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
        A = construct_matrix_A(phi, c, lambda_val, h, nx, ny)

        # 3. Construct the right-hand side vector f
        f = dt * lambda_val * phi0.flatten() # * (h**2) 

        # 4. Apply Implicit Euler update
        # Solve (I - dt*A) phi_new = phi + dt*f
        
        A_imp = sparse.eye(N) -(dt * A) #+ ( dt * lambda_val * sparse.eye(N))
        phi = splinalg.spsolve(-A_imp, phi.flatten() +  f).reshape(nx, ny)
        summed_per_time.append( np.sum(phi)/np.size(phi) )
        error_per_time.append( np.std(phi))
        # Calculate total energy at each time step



    # Calculate the absolute difference between total energy at each time step and initial total energy

    return phi,summed_per_time,error_per_time


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
    return c

def construct_matrix_A(phi, c, lambda_val, h, nx, ny):
    """Constructs the sparse matrix A using FVM discretization with Neumann BC."""
    N = nx * ny
    A = sparse.lil_matrix((N, N))

    for i in range(nx):
        for j in range(ny):
            k = j * nx + i  # 1D index

            # Diffusion coefficients at cell faces (using averaging)
            c_east = (c[i,j] + c[(i+1)%nx, j]) / 2 if i < nx - 1 else c[i,j]
            c_west = (c[i,j] + c[(i-1)%nx, j]) / 2 if i > 0 else c[i,j]
            c_north = (c[i,j] + c[i, (j+1)%ny]) / 2 if j < ny - 1 else c[i,j]
            c_south = (c[i,j] + c[i, (j-1)%ny]) / 2 if j > 0 else c[i,j]

            # Diagonal element
            A[k, k] = (c_east + c_west + c_north + c_south) / h**2 + lambda_val

            # Off-diagonal elements (neighbors)
            if i < nx - 1:  # East neighbor
                A[k, k + 1] = -c_east / h**2
            else:  # Neumann BC
                A[k,k] -= c_east / h**2  # Adjust diagonal instead of setting off-diagonal

            if i > 0:  # West neighbor
                A[k, k - 1] = -c_west / h**2
            else:  # Neumann BC
                A[k,k] -= c_west / h**2  # Adjust diagonal instead of setting off-diagonal

            if j < ny - 1:  # North neighbor
                A[k, k + nx] = -c_north / h**2
            else:  # Neumann BC
                A[k,k] -= c_north / h**2  # Adjust diagonal instead of setting off-diagonal

            if j > 0:  # South neighbor
                A[k, k - nx] = -c_south / h**2
            else:  # Neumann BC
                A[k,k] -= c_south / h**2 # Adjust diagonal instead of setting off-diagonal

    return A.tocsc()  # Convert to CSC format


def calculate_total_energy(phi, h):
    """Calculates the total energy of the image."""
    return np.sum(phi) * h**2

# Example Usage:
if __name__ == '__main__':
    
    # Load the images
    Im = Image.open('SL_simulated.gif')
    fm = np.array([])
    fn = np.array(Im)/255.0
    nx, ny = fn.shape

    fm = fn  # Original image (noiseless)
    sigma = 0.1
    np.random.seed(0)
    fn = fm + sigma * np.random.randn(nx, ny) # Noisy image
    
    phi0 = fn.copy()  # Initial image (noisy)

    # Set parameters
    lambda_val = 10000
    K = 5.0
    h = 1 / (nx - 1)  # Grid spacing

    # Choose a suitable dt
    dt = 0.001 # Larger dt for stationary solution
    n_steps = 50 # Adjust n_steps accordingly

    # Apply anisotropic diffusion filtering
    phi_filtered,summed_per_time,error_per_time = implicit_euler_anisotropic_diffusion(phi0, lambda_val, K, h, dt, n_steps, c_const = False)
    phi_filtered_const,summed_per_time,error_per_time = implicit_euler_anisotropic_diffusion(phi0, lambda_val, K, h, dt, n_steps, c_const = True)

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
    sum_x =  summed_per_time.copy()
    err_x = error_per_time.copy()

    # Plot the sum of the average pixel values:
    t = np.arange(0, (len(sum_x))*dt, dt)
    plt.figure()
    plt.plot(t, sum_x, '-x')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.title('Sum of pixel values, Improved Euler')
    plt.xlabel('Time')
    plt.xlim(left=0)
    plt.ylabel('$\Sigma x/N$')

    plt.figure()
    plt.title('Improved Euler')
    plt.imshow(fs, extent=[0, 1, 0, 1], cmap = 'gray')
    plt.axis('square')
    plt.axis('off')

    if fm.size > 0 :
        plt.figure()
        t = np.arange(0, (len(err_x))*dt, dt)
        plt.plot(t, err_x, '-x')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.title('Standard deviation versus time, Improved Euler')
        plt.xlabel('Time')
        plt.xlim(left=0)
        plt.ylabel('$\sigma$')

    plt.show()