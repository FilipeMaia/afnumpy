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
    intensities  = np.array(f['entry_1/data_1/data'][0]).astype(np.float32)
    data_fourier = np.array(f['entry_1/data_1/data_fourier'][0]).astype(np.complex64)

#intensities  = np.zeros((1024,1024)).astype(np.float32)
#data_fourier = np.zeros((1024,1024)).astype(np.complex64)
    
# True image
true_image = fft.fftshift(fft.ifftn(data_fourier))
#import arrayfire
#print arrayfire.backend.name
#print type(true_image)
#print true_image.dtype
#print intensities.dtype
#print data_fourier.dtype

# Initial image
image = (1j + np.random.random(intensities.shape)).astype(np.complex64)

# Define support
support = np.abs(true_image)>1

# Define Nr. of iterations
nr_iterations = 500

# Define data constraint
def data_constraint(fourier, intensities):
    return (fourier / np.abs(fourier)) * np.sqrt(intensities)

# Define support constraint
def support_constraint(img, support):
    img = img.flatten()
    img[support == 0] = 0
    #img *= support
    return img.reshape((256,256))
    #return  img

# Define error that defines convergence
def get_error(fourier, intensities):
    return np.abs((np.sqrt(intensities) - np.abs(fourier)).sum()) / np.sqrt(intensities).sum()

# Plotting
figs = []
axes = []
ims  = []
line = []
text = []
def update_plot(iteration, image, fourier, error, support, intensities):
    if not len(figs):
        plt.ion()
        fig = plt.figure(figsize=(7,10))
        ax5 = plt.subplot(313)
        ax1 = plt.subplot(321)
        ax2 = plt.subplot(322)
        ax3 = plt.subplot(323)
        ax4 = plt.subplot(324)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
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
        l, = ax5.plot(error)
        line.append(l)
        ax5.semilogy()
        text.append(axes[4].text(0.5, 0.9, 'Iteration = %d' %iteration, transform=axes[4].transAxes, va='center', ha='center'))
        #figs.append(fig.suptitle('Iteration %d' %iteration))
        
    else:
        #figs[1].set_text('Iteration %d' %iteration)
        ims[0].set_data(np.abs(image))
        ims[0].set_clim([np.abs(image).min(), np.abs(image).max()])
        ims[1].set_data(np.abs(image)*support)
        ims[1].set_clim([np.abs(image).min(), np.abs(image).max()])
        ims[2].set_data(np.abs(fourier)**2)
        ims[2].set_clim([intensities.min(), intensities.max()])
        ims[3].set_data(intensities)
        ims[3].set_clim([intensities.min(), intensities.max()])
        line[0].set_xdata(range(iteration+1))
        line[0].set_ydata(error)
        axes[4].set_xlim([0,iteration+1])
        axes[4].set_ylim([0.03, 0.1])
        text[0].set_text('Iteration = %d' %iteration)
    plt.draw()


fourier = fft.fftn(image)
update_plot(0, image[200:-200,200:-200], fourier, get_error(fourier, intensities), support[200:-200,200:-200], intensities)
    
print "Sleep for 20 seconds"
time.sleep(2)
print "Starting now"

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
    #error = 0
    #print "Iteration: %d, error: %f" %(i, error)

    # Update plot
    if (not i%10):
        update_plot(i, image[200:-200,200:-200], fourier, error, support[200:-200,200:-200], intensities)
    
    # Apply data constraint
    #fourier = data_constraint(fourier, intensities)
    fourier /= np.abs(fourier)
    fourier *= np.sqrt(intensities)

    # Backward propagation
    image = fft.ifftn(fourier)

    # Apply support constraint
    image *= support
    #image = support_constraint(image, support)

# Timing
t1 = time.time() - t0
print "%d Iterations took %2.f seconds (%.2f iterations per second) using %s" %(nr_iterations, t1, float(nr_iterations)/t1, use)

update_plot(i, image[200:-200, 200:-200], fourier, error, support[200:-200,200:-200], intensities)

plt.ioff()
plt.show()
