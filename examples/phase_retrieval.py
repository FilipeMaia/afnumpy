#!/usr/env python
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.ndimage import gaussian_filter

# Switch between numpy/afnumpy
#import numpy as np
#import numpy.fft as fft
import afnumpy as np
import afnumpy.fft as fft

# Load diffraction pattern
with h5py.File('virus.cxi', 'r') as f:
    intensities  = np.array(f['entry_1/data_1/data'][0])
    data_fourier = np.array(f['entry_1/data_1/data_fourier'][0])

# True image
true_image = fft.fftshift(fft.ifftn(data_fourier))

# Initial image
init_image = np.random.random(intensities.shape)
#init_image = true_image

# Define support
support = np.abs(true_image)>1

# Define data constraint
def data_constraint(fourier, intensities):
    return (fourier / np.abs(fourier)) * np.sqrt(intensities)

def support_constraint(real, support):
    real = real.flatten()
    real[np.where(support.flatten()==0)] = 0
    return real.reshape((256,256))

def get_error(fourier, intensities):
    return np.abs((np.sqrt(intensities) - np.abs(fourier)).sum()) / np.sqrt(intensities).sum()

# Error reduction
real = 1j + init_image
for i in range(1000):
    
    # Forward propagation
    fourier = fft.fftn(real)

    # Check convergence
    error = get_error(fourier, intensities)
    print "Iteration: %d, error: %f" %(i, error)
    
    # Apply data constraint
    fourier = data_constraint(fourier, intensities)

    # Backward propagation
    real = fft.ifftn(fourier)

    # Apply support constraint
    real = support_constraint(real, support)


# Reshaping
shape = (256,256)
true_image = true_image.reshape(shape)
real = real.reshape(shape)
fourier = fourier.reshape(shape)
intensities = intensities.reshape(shape)
    
# Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.imshow(np.abs(true_image))
ax2.imshow(np.abs(real))
ax3.imshow(np.abs(fourier)**2, norm=LogNorm())
ax4.imshow(intensities, norm=LogNorm())
plt.show()

