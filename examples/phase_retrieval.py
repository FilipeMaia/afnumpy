#!/usr/env python
import sys
import time
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.ndimage import gaussian_filter

# Switch between numpy/afnumpy
use_gpu = int(sys.argv[1])
if use_gpu:
    import afnumpy as np
    import afnumpy.fft as fft
    use = 'afnumpy/GPU'
else:
    import numpy as np
    import numpy.fft as fft
    use = 'numpy/CPU'

# Load diffraction pattern
with h5py.File('virus.cxi', 'r') as f:
    intensities  = np.array(f['entry_1/data_1/data'][0])
    data_fourier = np.array(f['entry_1/data_1/data_fourier'][0])

# True image
true_image = fft.fftshift(fft.ifftn(data_fourier))

# Initial image
image = 1j + np.random.random(intensities.shape)

# Define support
support = np.abs(true_image)>1

# Define Nr. of iterations
nr_iterations = 100

# Define data constraint
def data_constraint(fourier, intensities):
    return (fourier / np.abs(fourier)) * np.sqrt(intensities)

# Define support constraint
def support_constraint(img, support):
    img = img.flatten()
    img[np.where(support.flatten()==0)] = 0
    return img.reshape((256,256))

# Define error that defines convergence
def get_error(fourier, intensities):
    return np.abs((np.sqrt(intensities) - np.abs(fourier)).sum()) / np.sqrt(intensities).sum()

# Plotting
figs = []
axes = []
ims  = []
errs = []
line = []
def update_plot(iteration, image, fourier, error, support, intensities):
    errs.append(error)
    if not len(figs):
        plt.ion()
        fig = plt.figure(figsize=(6.5,10))
        ax5 = plt.subplot(313)
        ax1 = plt.subplot(321)
        ax2 = plt.subplot(322)
        ax3 = plt.subplot(323)
        ax4 = plt.subplot(324)
        
        figs.append(fig)
        [axes.append(ax) for ax in [ax1,ax2,ax3,ax4,ax5]]
        for i in range(4):
            axes[i].set_yticklabels([])
            axes[i].set_xticklabels([])
            axes[i].set_yticks([])
            axes[i].set_xticks([])
            
        #plt.tight_layout()
        ims.append(ax1.imshow(np.abs(image)))
        ims.append(ax2.imshow(np.abs(image)*support))
        ims.append(ax3.imshow(np.abs(fourier)**2, norm=LogNorm()))
        ims.append(ax4.imshow(intensities, norm=LogNorm()))
        l, = ax5.plot(errs)
        line.append(l)
        ax5.semilogy()
        figs.append(fig.suptitle('Iteration %d' %iteration))
        fig.set_tight_layout(True)

    else:
        figs[1].set_text('Iteration %d' %iteration)
        ims[0].set_data(np.abs(image))
        ims[0].set_clim([np.abs(image).min(), np.abs(image).max()])
        ims[1].set_data(np.abs(image)*support)
        ims[1].set_clim([np.abs(image).min(), np.abs(image).max()])
        ims[2].set_data(np.abs(fourier)**2)
        ims[3].set_data(intensities)
        line[0].set_xdata(range(iteration+1))
        line[0].set_ydata(errs)
        axes[4].set_xlim([0,iteration+1])
        axes[4].set_ylim([min(errs), max(errs)])
    plt.draw()

# Time the reconstruction
t0 = time.time()

# Start the reconstruction (using ER)
for i in range(nr_iterations):
    
    # Forward propagation
    fourier = fft.fftn(image)

    # Check convergence
    error = get_error(fourier, intensities)
    print "Iteration: %d, error: %f" %(i, error)

    # Update plot
    update_plot(i, image, fourier, error, support, intensities)
    
    # Apply data constraint
    fourier = data_constraint(fourier, intensities)

    # Backward propagation
    image = fft.ifftn(fourier)

    # Apply support constraint
    image = support_constraint(image, support)

# Timing
print "%d Iterations took %d seconds using %s" %(nr_iterations, time.time() - t0, use)
