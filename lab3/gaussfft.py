import numpy as np
from numpy.fft import fft2, ifft2, fftshift


def gaussian_filter(kernel_size, variance=1, muu=0):
    # Create meshgrid of size kernel_size
    x, y = np.meshgrid(np.linspace(-kernel_size//2, (kernel_size//2)-1, kernel_size),
                       np.linspace(-kernel_size//2, (kernel_size//2)-1, kernel_size))
    # Create 2D gaussian filter
    gauss = np.exp(-((x-muu)**2 + (y-muu)**2) / (2.0 * variance))
    # Normalize the filter
    gauss = gauss / np.sum(gauss)
    print(np.max(gauss))
    return gauss


def gaussfft(I, variance=1.0):
    # Generate filter based on sampled Gaussion function the same size as the input image I
    [h, w] = np.shape(I)
    # x = np.linspace(0, w-1, w)
    # y = np.linspace(0, h-1, h)
    # [X, Y] = np.meshgrid(x, y)
    # X, Y = np.meshgrid(x, y)
    # Create 2D Gaussian filter of size h, w
    gf = gaussian_filter(h, variance=variance, muu=0)
    # Compute FFT of image and filter
    iFFT = np.fft.fft2(I)
    gFFT = np.fft.fft2(gf)
    # Multiply in frequency domain and compute inverse FFT
    G = np.fft.ifft2(iFFT * gFFT)
    # Center the output
    G = np.fft.fftshift(G)
    return np.real(G), gFFT
