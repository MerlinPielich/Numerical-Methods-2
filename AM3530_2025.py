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

if gif_name == 'SL_simulated.gif':
    fm = fn
    sigma = 0.1
    np.random.seed(0)
    fn = fm + sigma * np.random.randn(nx, ny)
    # Plot the model image
##    plt.figure()
##    plt.title('Noise-free image')
##    plt.imshow(fm, extent=[0, 1, 0, 1], cmap = 'gray')
##    plt.axis('square')
##    plt.axis('off')
##
### Plot the noisy image
##plt.figure()
##plt.title('Noisy image')
##plt.imshow(fn, extent=[0, 1, 0, 1], cmap = 'gray')
##plt.axis('square')
##plt.axis('off')

##############

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

# Loop over the values of lambda to compute the solutions
solutions = {}
for lam in lambdas:
    # Construct the matrix A for the current lambda
    main_diag = np.zeros(nx * ny)
    right_diag = np.zeros(nx * ny)
    left_diag = np.zeros(nx * ny)
    top_diag = np.zeros(nx * ny)
    bottom_diag = np.zeros(nx * ny)

    for j in range(ny):
        for i in range(nx):
            k = j * nx + i  # 1D index corresponding to (i, j)

            # Diagonal elements (handling interior and boundary points)
            if 0 < i < nx-1 and 0 < j < ny-1:  # Interior points
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
            elif i == 0 and 0 < j < ny-1:  # Left boundary
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
            elif i == nx-1 and 0 < j < ny-1:  # Right boundary
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
            elif 0 < i < nx-1 and j == 0:  # Bottom boundary
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
            elif 0 < i < nx-1 and j == ny-1:  # Top boundary
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
            elif i == 0 and j == 0:  # Bottom-left corner
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
            elif i == nx-1 and j == 0:  # Bottom-right corner
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
            elif i == 0 and j == ny-1:  # Top-left corner
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
            elif i == nx-1 and j == ny-1:  # Top-right corner
                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam

            # Off-diagonal elements (neighbors)
            if i < nx-1:
                right_diag[k] = -1 / h**2
            if i > 0:
                left_diag[k] = -1 / h**2
            if j < ny-1:
                top_diag[k] = -1 / h**2
            if j > 0:
                bottom_diag[k] = -1 / h**2

    # Create the sparse matrix A
    diagonals = [main_diag, right_diag[1:], left_diag[:-1], top_diag[nx:], bottom_diag[:-nx]]
    offsets = [0, 1, -1, nx, -nx]
    A = diags(diagonals, offsets, shape=(nx * ny, nx * ny), format='csr')

    # Solve for phi (the solution at this lambda)
    u_lambda = spsolve(A, phi)

    # Reshape the solution back to a 2D grid
    u_lambda_grid = u_lambda.reshape((nx, ny))

    # Store the solutions for later plotting
    solutions[lam] = u_lambda_grid

### Plotting the four selected lambda values (smallest, two middle, largest)
##selected_lambdas = [lambdas[0], lambdas[len(lambdas)//2-1], lambdas[len(lambdas)//2], lambdas[-1]]
##
##plt.figure(figsize=(12, 10))
##
##for i, lam in enumerate(selected_lambdas, start=1):
##    plt.subplot(2, 2, i)
##    plt.imshow(solutions[lam], cmap='gray')
##    plt.title(f'(λ = {lam})')
##    plt.axis('off')
##
##plt.tight_layout()
##plt.show()

sigma_values = {}

for lam in lambdas:
    u_lambda = solutions[lam]
    # Flatten the images to 1D for correct calculation of the L2 norm
    u_lambda_flat = u_lambda.flatten()
    fm_flat = fm.flatten()  # Noiseless image (Shepp-Logan phantom)
    
    # Calculate the L2 norm of the difference between the true image and the computed solution
    diff = u_lambda_flat - fm_flat
    norm_diff = np.linalg.norm(diff)
    
    # Calculate the standard deviation as specified: sigma(lambda) = ||u_mod - u_lambda||_2 / sqrt(N)
    sigma_lambda = norm_diff / np.sqrt(nx * ny)
    
    # Store the value of sigma for this lambda
    sigma_values[lam] = sigma_lambda

# Print the sigma values for each lambda
for lam, sigma_lambda in sigma_values.items():
    print(f"λ = {lam}, σ(λ) = {sigma_lambda}")


###########

### Picard iteration
##method = 1
##plt.figure(figsize=(8,6))
##plt.suptitle('Picard iteration')
##sigma = np.zeros(9)
##for i in range(9):
##    fidelity = 10**(i)
##    # Add your code here. At this moment the filtered image is just a copy of the original image
##    fs = fn
##
##    plt.subplot(3, 3, i+1)
##    plt.title(f'Picard, $\lambda$ = {fidelity}')
##    plt.imshow(fs, extent=[0, 1, 0, 1], cmap = 'gray')
##    plt.axis('square')
##    plt.axis('off')
##
##    if fm.size > 0:
##        sigma[i] = np.linalg.norm(fs - fm, ord='fro') / np.sqrt(nx*ny)
##
##if fm.size > 0:
##    plt.figure()
##    plt.title('Standard deviation versus fidelity')
##    plt.plot(range(9), sigma)
##    plt.xlabel('Logarithm Fidelity')
##    plt.ylabel('$\sigma$')
##
##
##
### Explicit Euler
##method  = 2
##fidelty = 0
##
### Add your code here. At this moment the filtered image is just a copy of the original image
##fs = fn
##sum_x = np.ones(nt+1) # These you have to calculate
##err_x = np.ones(nt+1) # These you have to calculate
##
### Plot the sum of the average pixel values:
##t = np.arange(0, (len(sum_x))*dt, dt)
##plt.figure()
##plt.plot(t, sum_x, '-x')
##plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
##plt.title('Sum of pixel values, Explicit Euler')
##plt.xlabel('Time')
##plt.xlim(left=0)
##plt.ylabel('$\sum x/N$')
##
##plt.figure()
##plt.title('Explicit Euler')
##plt.imshow(fs, extent=[0, 1, 0, 1], cmap = 'gray')
##plt.axis('square')
##plt.axis('off')
##
##if fm.size > 0 :
##    plt.figure()
##    t = np.arange(0, (len(err_x))*dt, dt)
##    plt.plot(t, err_x, '-x')
##    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
##    plt.title('Standard deviation versus time, Explicit Euler')
##    plt.xlabel('Time')
##    plt.xlim(left=0)
##    plt.ylabel('$\sigma$')
##
##
##
### Improved Euler
##method = 3;
##fidelity = 0;
### Add your code here. At this moment the filtered image is just a copy of the original image
##fs = fn
##sum_x = np.ones(nt+1) # These you have to calculate
##err_x = np.ones(nt+1) # These you have to calcultate
##
### Plot the sum of the average pixel values:
##t = np.arange(0, (len(sum_x))*dt, dt)
##plt.figure()
##plt.plot(t, sum_x, '-x')
##plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
##plt.title('Sum of pixel values, Improved Euler')
##plt.xlabel('Time')
##plt.xlim(left=0)
##plt.ylabel('$\Sigma x/N$')
##
##plt.figure()
##plt.title('Improved Euler')
##plt.imshow(fs, extent=[0, 1, 0, 1], cmap = 'gray')
##plt.axis('square')
##plt.axis('off')
##
##if fm.size > 0 :
##    plt.figure()
##    t = np.arange(0, (len(err_x))*dt, dt)
##    plt.plot(t, err_x, '-x')
##    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
##    plt.title('Standard deviation versus time, Improved Euler')
##    plt.xlabel('Time')
##    plt.xlim(left=0)
##    plt.ylabel('$\sigma$')
##
##plt.show()







