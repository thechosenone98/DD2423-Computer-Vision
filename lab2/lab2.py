import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d
from pick import pick
import tkinter as tk
import copy

from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave

def compute_sobel_x(I, shape="same"):
    """Calculate image dx using Sobel filter"""
    # Create Sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Convolve image with Sobel filter
    dx = convolve2d(I, sobel_x, mode=shape)
    return dx

def compute_sobel_y(I, shape="same"):
    """Calculate image dy using Sobel filter"""
    # Create Sobel filter
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # Convolve image with Sobel filter
    dy = convolve2d(I, sobel_y, mode=shape)
    return dy

def compute_dx(I, shape="same"):
    """Calculate the image dx using central differences"""
    # Create central difference filter
    dx_filter = np.array([[0,   0,   0],
                          [0.5, 0, -0.5],
                          [0,   0,   0]])
    # Convolve image with central difference filter
    dx = convolve2d(I, dx_filter, mode=shape)
    return dx

def compute_dy(I, shape="same"):
    """Calculate the image dy using central differences"""
    # Create central difference filter
    dy_filter = np.array([[0,   0.5,   0],
                          [0,    0,    0],
                          [0,  -0.5,    0]])
    # Convolve image with central difference filter
    dy = convolve2d(I, dy_filter, mode=shape)
    return dy

def compute_dxx(I, shape="same"):
    """Calculate the image dxx using central differences"""
    # Create central difference filter
    dxx_filter = np.array([[0,  0, 0],
                           [1, -2, 1],
                           [0,  0, 0]])
    # Convolve image with central difference filter
    dxx = convolve2d(I, dxx_filter, mode=shape)
    return dxx

def compute_dyy(I, shape="same"):
    """Calculate the image dyy using central differences"""
    # Create central difference filter
    dyy_filter = np.array([[0,  1, 0],
                           [0, -2, 0],
                           [0,  1, 0]])
    # Convolve image with central difference filter
    dyy = convolve2d(I, dyy_filter, mode=shape)
    return dyy

#################################################### Differential Operators ####################################################
# 5x5 matrix of dx difference operator
dx_filter = np.array([[0,  0,   0,   0,   0],
                      [0,  0,   0,   0,   0],
                      [0, 0.5,  0, -0.5,  0],
                      [0,  0,   0,   0,   0],
                      [0,  0,   0,   0,   0]])
# 5x5 matrix of dy difference operator
dy_filter = np.transpose(dx_filter)
# 5x5 matrix of dxx difference operator
dxx_filter = np.array([[0,  0,  0,  0,  0],
                       [0,  0,  0,  0,  0],
                       [0,  1, -2,  1,  0],
                       [0,  0,  0,  0,  0],
                       [0,  0,  0,  0,  0]])
# 5x5 matrix of dyy difference operator
dyy_filter = np.transpose(dxx_filter)
# 5x5 matrix of dxy difference operator
dxy_filter = convolve2d(dx_filter, dy_filter, mode="same")
# 5x5 matrix of dxxy difference operator
dxxy_filter = convolve2d(dxx_filter, dy_filter, mode="same")
# 5x5 matrix of dxyy difference operator
dxyy_filter = convolve2d(dx_filter, dyy_filter, mode="same")
# 5x5 matrix of dxxx difference operator
dxxx_filter = convolve2d(dxx_filter, dx_filter, mode="same")
# 5x5 matrix of dyyy difference operator
dyyy_filter = convolve2d(dyy_filter, dy_filter, mode="same")
#################################################### Differential Operators ####################################################

def Lv(image_in, shape = "same"):
    Lx = compute_sobel_x(image_in, shape)
    Ly = compute_sobel_y(image_in, shape)
    return np.sqrt(Lx**2 + Ly**2)


def Lvvtilde(image_in, shape="same"):
    """Calculate Lvv of image_in"""
    Lx = convolve2d(image_in, dx_filter, mode=shape)
    Ly = convolve2d(image_in, dy_filter, mode=shape)
    Lxx = convolve2d(image_in, dxx_filter, mode=shape)
    Lyy = convolve2d(image_in, dyy_filter, mode=shape)
    Lxy = convolve2d(image_in, dxy_filter, mode=shape)
    Lvv_tilde = Lx**2 * Lxx + 2 * Lx * Ly * Lxy + Ly**2 * Lyy
    return Lvv_tilde

def Lvvvtilde(image_in, shape="same"):
    """Calculate Lvvv of image_in"""
    Lx = convolve2d(image_in, dx_filter, mode=shape)
    Ly = convolve2d(image_in, dy_filter, mode=shape)
    Lxxy = convolve2d(image_in, dxxy_filter, mode=shape)
    Lxyy = convolve2d(image_in, dxyy_filter, mode=shape)
    Lxxx = convolve2d(image_in, dxxx_filter, mode=shape)
    Lyyy = convolve2d(image_in, dyyy_filter, mode=shape)
    Lvvv_tilde = Lx**3 * Lxxx + 3 * Lx**2 * Ly * Lxxy + 3 * Lx * Ly**2 * Lxyy + Ly**3 * Lyyy
    return Lvvv_tilde


def extractedges(inpic, scale, threshold, shape="same"):
    # Compute blur of image to get a certain scale
    gaussian_smooth, _ = gaussfft(inpic, scale)
    # Compute the gradient magitude (first derivative magnitude)
    gradient_magnitude = Lv(gaussian_smooth, "same")

    # Compute the Laplacian of the image (second derivative)
    Lvv = Lvvtilde(gaussian_smooth, shape)
    # Compute the Laplacian of the Laplacian of the image (third derivative)
    Lvvv = Lvvvtilde(gaussian_smooth, shape)

    # Remove small value in first derivative to remove noise being detected as edge
    Lv_mask = gradient_magnitude > threshold
    # Check when the third derivative is negative (local maxima condition)
    Lvvv_mask = Lvvv < 0

    curves = zerocrosscurves(Lvv, Lvvv_mask)
    contours = thresholdcurves(curves, Lv_mask)
    return contours

def houghline(curves, magnitude, nrho, ntheta, threshold, nlines, verbose, increment_function=lambda x: 1):
    """Computes Hough line transform and returns array of line parameters"""
    # Allocate accumulator space
    accumulator = np.zeros((nrho, ntheta))
    # Define a coordinate system in the accumulator space
    r = np.sqrt(magnitude.shape[0]**2 + magnitude.shape[1]**2)
    rho = np.linspace(-r, r, nrho)
    theta = np.linspace(-np.pi / 2, np.pi / 2, ntheta)
    # Loop over all the edge points
    for i in range(len(curves[0])):
        x = curves[0][i]
        y = curves[1][i]
        curve_magnitude = magnitude[x][y]
        # If the magnitude of the edge is high enough, we consider it for the line detection
        if curve_magnitude > threshold:
            # Loop over all the angles
            for j in range(ntheta):
                # Calculate the corresponding rho
                rho_value = x * np.cos(theta[j]) + y * np.sin(theta[j])
                # Find the closest rho in the accumulator space
                rho_index = np.argmin(np.abs(rho - rho_value))
                # Increment the accumulator
                accumulator[rho_index][j] += increment_function(curve_magnitude)

    line_params = []
    # Extract local maxima from the accumulator
    pos, value, _ = locmax8(accumulator)
    # Sort the local maxima by their value (we use argsort in order to use the index to retrieve the corresponding position)
    index_vector = np.argsort(value)[-nlines:]
    # Extract the position that match those local maxima using the index_vector
    pos = pos[index_vector]
    
    # Retrieve the parameter corresponding to those position in the rho theta space
    # Also plot some stuff if verbose is True
    if verbose:
        f, ax = plt.subplots(1, 4, figsize=(20, 10))
        f.subplots_adjust(hspace=0.5, wspace=0.5)
        ax[0].set_title('curves')
        [h, w] = np.shape(magnitude)
        grayscale_curve = np.zeros((h, w, 1), dtype=np.uint8)
        (Y, X) = curves
        grayscale_curve[Y, X] = 255
        showgrey(grayscale_curve, False, plot=ax[0])
        ax[1].set_title('magnitude')
        magnitude_tresholded = magnitude.copy()
        magnitude_tresholded[magnitude < threshold] = 0
        showgrey(magnitude_tresholded, False, plot=ax[1])
        ax[2].set_title('accumulator')
        showgrey(accumulator, False, plot=ax[2])
        ax[3].set_title('lines')
        ax[3].set_aspect('equal')
        ax[3].set_xlim(0, w)
        ax[3].set_ylim(h, 0)
        ax[3].imshow(magnitude, cmap='gray')
    for theta_idx, rho_idx in pos[:nlines]:
        rho_value = rho[rho_idx]
        theta_value = theta[theta_idx]
        line_params.append([rho_value, theta_value])
        if verbose:
            # Print position, line parameters
            print(f"Pos: {(rho_idx, theta_idx)}, rho: {rho_value:.3f}, theta: {theta_value:.3f}")
            x0 = rho_value * np.cos(theta_value)
            y0 = rho_value * np.sin(theta_value)
            dx = r * (-np.sin(theta_value))
            dy = r * (np.cos(theta_value))
            # Force aspect ratio to be the same as magnitude
            ax[3].plot([y0 - dy, y0, y0 + dy], [x0 - dx, x0, x0 + dx], "r-")
    if verbose:
        plt.show()
    return line_params, accumulator

def houghedgelines(pic, scale, gradient_magnitude_treshold, n_rho, n_theta, n_lines, verbose=False, increment_function=lambda x: 1):
    """Compute the Hough line transform of the edges of the image"""
    # Extract edges
    contours = extractedges(pic, scale, gradient_magnitude_treshold)
    # Compute the gradient magnitude of the image
    magnitude = Lv(pic, "same")
    # Compute the Hough line transform
    line_params, accumulator = houghline(contours, magnitude, n_rho, n_theta, gradient_magnitude_treshold, n_lines, verbose, increment_function)
    return line_params, accumulator

def overlay_lines(img, line_params):
    r = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)
    plt.imshow(img, cmap='gray')
    for rho, theta in line_params:
        x0 = rho * np.cos(theta)
        y0 = rho * np.sin(theta)
        dx = r * (-np.sin(theta))
        dy = r * (np.cos(theta))
        plt.plot([y0 - dy, y0, y0 + dy], [x0 - dx, x0, x0 + dx], "r-")
    plt.show()


def question1():
    tools = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/few256.npy")
    dx_tools = compute_sobel_x(tools)
    dy_tools = compute_sobel_y(tools)
    # Plot result and original image
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(tools, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(dx_tools, cmap='gray')
    ax[1].set_title('dx')
    ax[2].imshow(dy_tools, cmap='gray')
    ax[2].set_title('dy')
    # Print output images size
    print("original size: ", np.shape(tools))
    print("dx size: ", np.shape(dx_tools))
    print("dy size: ", np.shape(dy_tools))
    plt.show()


def question2():
    tools = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/few256.npy")
    dx_tools = compute_sobel_x(tools)
    dy_tools = compute_sobel_y(tools)
    # Compute the magnitude of the gradient
    mag_tools = np.sqrt(dx_tools**2 + dy_tools**2)
    # Compute the image histogram
    hist_tools = np.histogram(mag_tools, bins=16)
    # Based on the histogram, choose a threshold
    threshold = 100
    # Treshold the gradient magnitude image to get the edges
    edge_tool = mag_tools > threshold
    # Plot the original image, the histogram and the tresholded image
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(tools, cmap='gray')
    ax[0].set_title('Original')
    ax[1].plot(hist_tools[1][:-1], hist_tools[0])
    ax[1].set_title('Histogram of gradient magnitude')
    ax[2].imshow(edge_tool, cmap='gray')
    ax[2].set_title('Edges')
    plt.show()

    # Load godthem256 image
    godthem = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/godthem256.npy")
    # Blur the image with a gaussian Blur
    godthem_blur, _ = gaussfft(godthem, 0.5)
    # Apply Sobel filter to original image
    mag_gothem = Lv(godthem)
    # Apply Sobel filter to get gradient magnitude
    mag_godthem_blur = Lv(godthem_blur)
    # Compute the image histogram of the orignal image
    hist_godthem = np.histogram(mag_gothem, bins=16)
    # Compute the image histogram of the blurred image
    hist_godthem_blur = np.histogram(mag_godthem_blur, bins=16)
    # Based on the histogram, choose a threshold
    threshold = 100
    # Threshold the gradient magnitude of the original image to get edges
    edge_godthem = mag_gothem > threshold
    # Treshold the gradient magnitude image to get the edges
    edge_godthem_blur = mag_godthem_blur > threshold
    # Plot the original image, the blurred image, the blurred histogram and the blurred tresholded image
    # along with the original histogram and the original tresholded image
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(godthem, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 1].plot(hist_godthem[1][:-1], hist_godthem[0])
    ax[0, 1].set_title('Histogram of gradient magnitude')
    ax[0, 2].imshow(edge_godthem, cmap='gray')
    ax[0, 2].set_title('Edges')
    ax[1, 0].imshow(godthem_blur, cmap='gray')
    ax[1, 0].set_title('Blurred')
    ax[1, 1].plot(hist_godthem_blur[1][:-1], hist_godthem_blur[0])
    ax[1, 1].set_title('Histogram of gradient magnitude')
    ax[1, 2].imshow(edge_godthem_blur, cmap='gray')
    ax[1, 2].set_title('Edges')
    plt.show()

def question3():
    pass

def question4():
    [x, y] = np.meshgrid(range(-5, 6), range(-5, 6))
    print(x)
    # Calculate dxxx
    print("dxxx of x**3")
    print(convolve2d(x**3, dxxx_filter, mode="valid"))
    # Calculate dxx
    print("dxx of x**3")
    print(convolve2d(x**3, dxx_filter, mode="valid"))
    # Calculate dxxy
    print("dxxy of x**2 * y")
    print(convolve2d(x**2 * y, dxxy_filter, mode="valid"))
    # Calculate dxxxx
    print("dxxxx of x**3")
    print(convolve2d(convolve2d(x**3, dxxx_filter, mode="valid"), dx_filter, mode="valid"))
    # Calculate dx of y
    print("dx of y")
    print(convolve2d(y, dx_filter, mode="valid"))

    house = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/godthem256.npy")
    # Plot a contour plot at different scales of Lvv
    fig, ax = plt.subplots(5, 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, scale in enumerate([0.0001, 1.0, 4.0, 16.0, 64.0]):
        # house_blur, _ = gaussfft(house, scale)
        house_blur, _ = discgaussfft(house, scale)
        Lvv = Lvvtilde(house_blur)
        contour_Lvv = contour(Lvv)
        ax[i][0].imshow(house_blur, cmap='gray')
        ax[i][0].set_title('Blurred')
        ax[i][1].imshow(Lvv, cmap='gray')
        ax[i][1].set_title('Lvv')
        ax[i][2].imshow(contour_Lvv, cmap='gray')
        ax[i][2].set_title('Contour')
    plt.show()

    tools = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/few256.npy")
    # Plot Lvvvtilde at different scales
    fig, ax = plt.subplots(5, 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, scale in enumerate([0.0001, 1.0, 4.0, 16.0, 64.0]):
        tools_blur, _ = discgaussfft(tools, scale)
        Lvvv = Lvvvtilde(tools_blur)
        filtered_Lvvv = Lvvv < 0
        ax[i][0].imshow(tools_blur, cmap='gray')
        ax[i][0].set_title('Blurred')
        ax[i][1].imshow(Lvvv, cmap='gray')
        ax[i][1].set_title('Lvvv')
        ax[i][2].imshow(filtered_Lvvv, cmap='gray')
        ax[i][2].set_title('Contour')
    plt.show()

def question5():
    pass

def question6():
    pass

def question7():
    # Use a grid search to vary the parameters of extractedges with various images
    # and plot the results
    images = [  "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/godthem256.npy",
                "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/few256.npy"]
    img = np.load(images[1])
    # img = np.load(images[1])
    edges = extractedges(img, 5, 35)
    overlaycurves(img, edges, True)
    plt.show()

def question8():
    # images = [  "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/godthem256.npy",
    #             "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/few256.npy"]
    testimage1 = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/godthem256.npy")
    lineparams, _ = houghedgelines(testimage1, 1.5, 50, 400, 400, 20, False)
    overlay_lines(testimage1, lineparams)

def question9():
    testimage1 = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/triangle128.npy")
    subsampled_img = binsubsample(testimage1)
    lineparams, _ = houghedgelines(subsampled_img, 0.1, 40, 100, 100, 5, True)
    overlay_lines(subsampled_img, lineparams)

def hough_incrementer(value):
    return value

def question10():
    testimage1 = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/godthem256.npy")
    lineparams, _ = houghedgelines(testimage1, 1.5, 50, 400, 400, 40, True, increment_function=hough_incrementer)
    overlay_lines(testimage1, lineparams)


funct_dict = {"Question 1": question1, "Question 2": question2, "Question 3": question3, "Question 4": question4, "Question 5": question5, "Question 6": question6, "Question 7": question7, "Question 8": question8, "Question 9": question9, "Question 10": question10}


def main():
    title = "Please choose a question to run: "
    options = list(funct_dict.keys())
    option, index = pick(options, title)
    funct_dict[option]()

if __name__ == "__main__":
    main()