#!/usr/env python
import sys
import time
import argparse
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import rotate

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-a', '--use-afnumpy',action='store_true')
group.add_argument('-n', '--use-numpy',  action='store_true')
parser.add_argument('-p', '--plotting', action='store_true')
parser.add_argument('-d', '--debug', action='store_true')
args = parser.parse_args()

# Switch between numpy/afnumpy
if args.use_afnumpy:
    import afnumpy as np
    import afnumpy.fft as fft
    use = 'afnumpy/GPU'
elif args.use_numpy:
    import numpy as np
    import numpy.fft as fft
    use = 'numpy/CPU'

# Generate test dataset
shape = (512,512)
center = (int((shape[0])/2), int((shape[1])/2))
radius = 50
solution = np.zeros(shape)
solution[center[0]-radius:center[0]+radius-1,center[1]-radius:center[1]+radius-1] = 1.
solution += rotate(solution, -45, reshape=False)
solution = gaussian_filter(solution, radius//10)
fourier = np.fft.fftshift(np.fft.fft2(solution))
intensities = np.abs(fourier)**2

# Initial (random) image
image = (1j + np.random.random(intensities.shape)).astype(np.complex64)

# Define support
yy,xx = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
rr = np.sqrt((xx-image.shape[1]/2)**2 + (yy-image.shape[0]/2)**2)
support = rr < np.sqrt(2) * radius - 1

# Define Nr. of iterations
nr_iterations = 500

# Define error that defines convergence
def get_error(fourier, intensities):
    return np.abs((np.sqrt(intensities) - np.abs(fourier)).sum()) / np.sqrt(intensities).sum()
    
# Time the reconstruction
t0 = time.time()

# Store error
error = []

# Start the reconstruction (using ER)
for i in range(nr_iterations):
    
    # Forward propagation
    fourier = fft.fftn(image)

    # Check convergence
    error.append(get_error(fourier, intensities))
    if args.debug:
        print("Iteration: %d, error: %f" %(i, error[-1]))
    
    # Apply data constraint
    fourier /= np.abs(fourier)
    fourier *= np.sqrt(intensities)

    # Backward propagation
    image = fft.ifftn(fourier)

    # Apply support constraint
    image *= support

# Timing
t1 = time.time() - t0
success = np.sum(((np.abs(image) - solution)**2)*support) / np.prod(shape) < 1e-2
print("Success: %d, %d Iterations took %2.f seconds (%.2f iterations per second) using %s" %(success, nr_iterations, t1, float(nr_iterations)/t1, use))

# Check for plotting
if not args.plotting:
    sys.exit(0)

# Plotting the result
try:
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
except ImportError:
    print("Could not find matplotlib, no plots produced")
    sys.exit(0)

fig = plt.figure(figsize=(7,10))
ax5 = plt.subplot(313)
ax1 = plt.subplot(321)
ax2 = plt.subplot(322)
ax3 = plt.subplot(323)
ax4 = plt.subplot(324)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
axes = [ax1,ax2,ax3,ax4]
for i in range(4):
    axes[i].set_yticklabels([])
    axes[i].set_xticklabels([])
    axes[i].set_yticks([])
    axes[i].set_xticks([])
            
plt.tight_layout()
ax1.imshow(np.abs(image), vmin=solution.min(), vmax=solution.max())
ax2.imshow(np.abs(image)-solution, vmin=-1, vmax=1)
ax3.imshow(np.angle(fourier), vmin=-np.pi, vmax=np.pi)
ax4.imshow(intensities, norm=colors.LogNorm())
l, = ax5.plot(error)
ax5.semilogy()
ax5.text(0.5, 0.9, 'Iteration = %d' %nr_iterations, transform=ax5.transAxes, va='center', ha='center')
plt.show()
