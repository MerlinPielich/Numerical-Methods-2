import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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
    plt.figure()
    plt.title('Noise-free image')
    plt.imshow(fm, extent=[0, 1, 0, 1], cmap = 'gray')
    plt.axis('square')
    plt.axis('off')

# Plot the noisy image
plt.figure()
plt.title('Noisy image')
plt.imshow(fn, extent=[0, 1, 0, 1], cmap = 'gray')
plt.axis('square')
plt.axis('off')



#######################G implementatie
##lambdas = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
##h = 1.0 / nx
##phi = fn.flatten()
##solutions = {}
##sigma_lambda = []
##
##for lam in lambdas:
##    main_diag = np.zeros(nx * ny)
##    right_diag = np.zeros(nx * ny)
##    left_diag = np.zeros(nx * ny)
##    top_diag = np.zeros(nx * ny)
##    bottom_diag = np.zeros(nx * ny)
##
##    for j in range(ny):
##        for i in range(nx):
##            k = j * nx + i  #(i, j)
##
##            #diagonal
##            if 0 < i < nx-1 and 0 < j < ny-1:  
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##            elif i == 0 and 0 < j < ny-1:
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##            elif i == nx-1 and 0 < j < ny-1:  
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##            elif 0 < i < nx-1 and j == 0:  
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##            elif 0 < i < nx-1 and j == ny-1:
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##            elif i == 0 and j == 0:
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##            elif i == nx-1 and j == 0:  
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##            elif i == 0 and j == ny-1:  
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##            elif i == nx-1 and j == ny-1:
##                main_diag[k] = (1 + 1 + 1 + 1) / h**2 + lam
##
##            #off-diagonal
##            if i < nx-1:
##                right_diag[k] = -1 / h**2
##            if i > 0:
##                left_diag[k] = -1 / h**2
##            if j < ny-1:
##                top_diag[k] = -1 / h**2
##            if j > 0:
##                bottom_diag[k] = -1 / h**2
##
##    #create A
##    diagonals = [main_diag, right_diag[1:], left_diag[:-1], top_diag[nx:], bottom_diag[:-nx]]
##    offsets = [0, 1, -1, nx, -nx]
##    A = diags(diagonals, offsets, shape=(nx * ny, nx * ny), format='csr')
##    u_lambda = spsolve(A, lam * fn.flatten())  #solve noise im
##    u_lambda_grid = u_lambda.reshape((nx, ny))
##    solutions[lam] = u_lambda_grid
##    sigma_lambda.append(np.linalg.norm(fm - u_lambda_grid) / np.sqrt(nx * ny))
##
##for lam, sig in zip(lambdas, sigma_lambda):
##    print(f"lambda = {lam}, sigma(lambda) = {sig:.6f}")
##
###Plot 4selected lambda values
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
##############################


# Define grid spacing
h = 1.0 / (nx -1)

# Picard iteration
method = 1
plt.figure(figsize=(8,6))
plt.suptitle('Picard iteration')
# Picard iteration
sigma_vals = np.zeros(9)

for lam_idx in range(9):  # Iterate over 9 lambda values
    fidelity = 10**(lam_idx)  # Different lambda (fidelity)
    
    # Initialize phi and fs
    phi = fn.copy()  # Start with noisy image
    fs = fn.copy()
    
    for _ in range(nt):
        A = lil_matrix((nx*ny, nx*ny))
        b = np.zeros(nx*ny)
        
        # Construct the system
        for j in range(ny):
            for i in range(nx):
                k = j * nx + i
                
                if 0 < i < nx - 1 and 0 < j < ny - 1:  # Interior points
                    c_ip_half_j = 1 / (1 + ((phi[i, j] - phi[i-1, j]) / (2*h))**2 + ((phi[i, j+1] - phi[i, j-1]) / (2*h))**2)
                    c_im_half_j = 1 / (1 + ((phi[i, j] - phi[i+1, j]) / (2*h))**2 + ((phi[i, j+1] - phi[i, j-1]) / (2*h))**2)
                    c_i_jp_half = 1 / (1 + ((phi[i+1, j] - phi[i-1, j]) / (2*h))**2 + ((phi[i, j+1] - phi[i, j]) / (2*h))**2)
                    c_i_jm_half = 1 / (1 + ((phi[i+1, j] - phi[i-1, j]) / (2*h))**2 + ((phi[i, j] - phi[i, j-1]) / (2*h))**2)
                    
                    A[k, k] = (c_ip_half_j + c_im_half_j + c_i_jp_half + c_i_jm_half) / h**2 + fidelity
                    A[k, k+1] = -c_ip_half_j / h**2
                    A[k, k-1] = -c_im_half_j / h**2
                    A[k, k+nx] = -c_i_jp_half / h**2
                    A[k, k-nx] = -c_i_jm_half / h**2
                    b[k] = fidelity * fn[i, j]
                else:  # Boundary conditions
                    A[k, k] = 1  # Keep boundary value fixed
                    b[k] = fn[i, j]  # Set right-hand side to boundary value (noisy image)
        
        # Solve the system
        phi_vec = spsolve(A.tocsr(), b)
        phi = phi_vec.reshape(nx, ny).T  # Update phi
    
    # Plot the result
    plt.subplot(3, 3, lam_idx + 1)  # Now correctly mapped to 1-9
    plt.title(f'Picard, $\lambda$ = {fidelity}')
    plt.imshow(phi, extent=[0, 1, 0, 1], cmap='gray')
    plt.axis('square')
    plt.axis('off')

    # Calculate sigma for comparison with the noise-free image
    if fm.size > 0:
        sigma_vals[lam_idx] = np.linalg.norm(phi - fm, ord='fro') / np.sqrt(nx * ny)
    
if fm.size > 0:
    plt.figure()
    plt.title('Standard deviation versus fidelity')
    plt.plot(range(9), sigma_vals)
    plt.xlabel('Logarithm Fidelity')
    plt.ylabel('$\\sigma$')
plt.show()





