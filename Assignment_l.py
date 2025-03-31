import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt  # Import matplotlib
from PIL import Image
import signal_noise as sn 

def explicit_euler_anisotropic_diffusion(phi0, lambda_val, K, h, dt, n_steps, c_const = False):
    """
    Applies anisotropic diffusion filtering using Explicit Euler method.

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
    error_per_time = []
    summed_per_time = []
    
    nx, ny = phi0.shape
    phi = phi0.copy()

    for n in range(n_steps):
        # 1. Calculate c(phi)
        if c_const:
            c = np.ones_like(phi)
        else:
            c = calculate_diffusion_coefficient(phi, K, h)

        # 2. Compute the discretized diffusion term at each grid point
        diffusion_term = np.zeros_like(phi, dtype=float)
        for i in range(nx):
            for j in range(ny):
                # Diffusion coefficients at cell faces (using averaging from hand-in J)
                c_east = (c[i,j] + c[(i+1)%nx, j]) / 2
                c_west = (c[i,j] + c[(i-1)%nx, j]) / 2
                c_north = (c[i,j] + c[i, (j+1)%ny]) / 2
                c_south = (c[i,j] + c[i, (j-1)%ny]) / 2

                # FVM discretization of the divergence term
                diffusion_term[i, j] = (
                    (c_east * (phi[(i+1)%nx, j] - phi[i, j]) - c_west * (phi[i, j] - phi[(i-1)%nx, j])) / h**2 +
                    (c_north * (phi[i, (j+1)%ny] - phi[i, j]) - c_south * (phi[i, j] - phi[i, (j-1)%ny])) / h**2
                )

        # 3. Apply Explicit Euler update
        phi = phi + dt * (diffusion_term + lambda_val * (phi0 - phi))
        summed_per_time.append( np.sum(phi)/np.size(phi) )
        error_per_time.append( np.std(phi))

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
            A[k, k] = -(c_east + c_west + c_north + c_south) / h**2 - lambda_val

            # Off-diagonal elements (neighbors)
            if i < nx - 1:  # East neighbor
                A[k, k + 1] = c_east / h**2
            else: #Periodic Boundary Condition
                A[k, k - (nx-1)] = c_east / h**2

            if i > 0:  # West neighbor
                A[k, k - 1] = c_west / h**2
            else: #Periodic Boundary Condition
                A[k, k + (nx-1)] = c_west / h**2

            if j < ny - 1:  # North neighbor
                A[k, k + nx] = c_north / h**2
            else: #Periodic Boundary Condition
                A[k, k - (nx*(ny-1))] = c_north / h**2

            if j > 0:  # South neighbor
                A[k, k - nx] = c_south / h**2
            else: #Periodic Boundary Condition
                A[k, k + (nx*(ny-1))] = c_south / h**2

    return A.tocsc()  # Convert to CSC format for efficient sparse linear algebra


# Example Usage:
if __name__ == '__main__':
    
    Im = Image.open('SL_simulated.gif')

    fn = np.array(Im)/255.0
    print('the max value in fn is {} and the min value is {}'.format(np.max(fn), np.min(fn)))
    nx, ny = fn.shape
    fn = np.clip(fn, 0, 1)
    dither = Image.effect_noise((nx,ny), 12.8)
    # effect_noise((nx,ny), 0.1)
    sigma = 0.1
    np.random.seed(0)
    # fn = fm + sigma * np.random.randn(nx, ny) # Noisy image
    fn = np.array(Im)/255 + sigma * np.random.randn(nx, ny) # Noisy image
    # fm = np.array([])
    # print('the max value in fn is {} and the min value is {}'.format(np.max(fn), np.min(fn)))
    # phi0 = (fn.copy()-np.min(fn))/(np.max(fn)-np.min(fn))
    # phi0 = np.clip(fn.copy(),a_min = np.zeros_like(fn), a_max = np.ones_like(fn))
    # print('the max value in fn is {} and the min value is {}'.format(np.max(phi0), np.min(phi0)))
    phi0 = fn.copy()

    # Set parameters
    lambda_val = 10000
    K = 5
    h = 1 / (nx - 1)  # Grid spacing
    dt = 1e-5
    n_steps = 40

    # Apply anisotropic diffusion filtering
    phi_filtered,summed_per_time,error_per_time = explicit_euler_anisotropic_diffusion(phi0, lambda_val, K, h, dt, n_steps, c_const = False)
    phi_filtered_const,summed_per_time,error_per_time = explicit_euler_anisotropic_diffusion(phi0, lambda_val, K, h, dt, n_steps, c_const = True)

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
    # Euler#
    method  = 2
    fidelty = 0

    # Time step:
    dt = 1.e-5
    nt = 40

    # Add your code here. At this moment the filtered image is just a copy of the original image
    fs = phi_filtered.copy() # This is the filtered image, you have to calculate it
    sum_x = summed_per_time # These you have to calculate
    err_x = error_per_time # These you have to calculate

    # Plot the sum of the average pixel values:
    t = np.arange(0, (len(sum_x))*dt, dt)
    plt.figure()
    plt.plot(t, sum_x, '-x')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.title('Sum of pixel values, Explicit Euler')
    plt.savefig('sum_pixel_values.png')
    
    plt.xlabel('Time')
    plt.xlim(left=0)
    plt.ylabel('$\sum x/N$')
    
    plt.figure()
    plt.title('Explicit Euler')
    plt.imshow(fs, extent=[0, 1, 0, 1], cmap = 'gray')
    plt.axis('square')
    plt.axis('off')

    if fn.size > 0 :
        plt.figure()
        t = np.arange(0, (len(err_x))*dt, dt)
        plt.plot(t, err_x, '-x')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.title('Standard deviation versus time, Explicit Euler')
        plt.xlabel('Time')
        plt.xlim(left=0)
        plt.ylabel('$\sigma$')
        plt.savefig('std_vs_time.png')
        plt.show()
            