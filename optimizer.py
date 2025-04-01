import numpy as np
from scipy.optimize import minimize
from test8 import implicit_euler_anisotropic_diffusion, calculate_total_energy
from PIL import Image

def objective(x, phi0, fm, K, h, n_steps):
    # x[0]: lambda, x[1]: dt
    lam = max(x[0], 1e-8)
    dt_optimized = max(x[1], 1e-10)
    # Run the diffusion filter using the current lambda and dt    
    phi_filtered, _, _ = implicit_euler_anisotropic_diffusion(phi0, lam, K, h, dt_optimized, n_steps, c_const=False)
    # Compute the squared L2 error between the filtered image and the original image (fm)
    error = np.linalg.norm(phi_filtered - fm)**2
    return error

# Example Usage:
# (Assuming phi0, K, h, dt, n_steps are defined, and target_energy is chosen)
# For instance, target_energy could be set to the energy of the original image:
# Load the images
Im = Image.open('SL_simulated.gif')
fm = np.array(Im) / 255.0
fn = np.array(Im) / 255.0
nx, ny = fn.shape

fm = fn  # Original image (noiseless)
sigma = 0.1
np.random.seed(0)
fn = fm + sigma * np.random.randn(nx, ny)  # Noisy image
h = 1 / (nx - 1)  # Grid spacing
phi0 = fn.copy()  # Initial image (noisy)
K = 5
n_steps = 250  # Number of steps

initial_guess = [4470.545264569677, 1e-5]  # initial guess for lambda and dt

res = minimize(objective, x0=initial_guess, args=(phi0, fm, K, h, n_steps), method='Nelder-Mead')
optimal_lambda, optimal_dt = res.x
print("Optimal lambda:", optimal_lambda)
print("Optimal dt:", optimal_dt)