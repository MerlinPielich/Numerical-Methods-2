import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


# Picard iteration
method = 1
plt.figure(figsize=(8,6))
plt.suptitle('Picard iteration')
sigma = np.zeros(9)
for i in range(9):
    fidelity = 10**(i)
    # Add your code here. At this moment the filtered image is just a copy of the original image
    fs = fn

    plt.subplot(3, 3, i+1)
    plt.title(f'Picard, $\lambda$ = {fidelity}')
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



# Explicit Euler
method  = 2
fidelty = 0

# Add your code here. At this moment the filtered image is just a copy of the original image
fs = fn
sum_x = np.ones(nt+1) # These you have to calculate
err_x = np.ones(nt+1) # These you have to calculate

# Plot the sum of the average pixel values:
t = np.arange(0, (len(sum_x))*dt, dt)
plt.figure()
plt.plot(t, sum_x, '-x')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.title('Sum of pixel values, Explicit Euler')
plt.xlabel('Time')
plt.xlim(left=0)
plt.ylabel('$\sum x/N$')

plt.figure()
plt.title('Explicit Euler')
plt.imshow(fs, extent=[0, 1, 0, 1], cmap = 'gray')
plt.axis('square')
plt.axis('off')

if fm.size > 0 :
    plt.figure()
    t = np.arange(0, (len(err_x))*dt, dt)
    plt.plot(t, err_x, '-x')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.title('Standard deviation versus time, Explicit Euler')
    plt.xlabel('Time')
    plt.xlim(left=0)
    plt.ylabel('$\sigma$')



# Improved Euler
method = 3;
fidelity = 0;
# Add your code here. At this moment the filtered image is just a copy of the original image
fs = fn
sum_x = np.ones(nt+1) # These you have to calculate
err_x = np.ones(nt+1) # These you have to calcultate

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







