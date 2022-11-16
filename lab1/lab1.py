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


def question1():
    for i, point in enumerate(((5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1))):
        fftwave(*point)
    plt.show()


def draw(event, canvas, ax, canvas2):
    # Get the x and y coordinates of the mouse click
    x = event.x
    y = event.y
    # Clear the canvas (only one point can be put at a time)
    canvas.delete("all")
    # Draw a circle on the canvas
    canvas.create_rectangle(x, y, x, y, outline="white")
    # Set the corresponding value in the Fhat matrix
    Fhat = np.zeros([128, 128])
    Fhat[y - 128//2, x - 128//2] = 1
    # Plot the image on the right
    F = np.fft.ifft2(Fhat)
    Fshift = np.fft.fftshift(F)
    showgrey(Fshift, display=False, plot=ax)
    canvas2.draw()


def question2():
        # Plot a 3d sine wave
    # Creating dataset
    x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
    y = x.copy().T # transpose
    z = np.sin(6*x)
    
    # Creating figure
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    
    # Creating color map
    my_cmap = plt.get_cmap('Greys')
    
    # Creating plot
    surf = ax.plot_surface(x, y, z,
                        rstride = 1,
                        cstride = 1,
                        alpha = 0.8,
                        cmap = my_cmap)
    cset = ax.contourf(x, y, z,
                    zdir ='z',
                    offset = np.min(z),
                    cmap = my_cmap)
    # cset = ax.contourf(x, y, z,
    #                 zdir ='x',
    #                 offset =-5,
    #                 cmap = my_cmap)
    cset = ax.contourf(x, y, z,
                    zdir ='y',
                    offset = 5,
                    cmap = my_cmap)
    fig.colorbar(surf, ax = ax,
                shrink = 0.5,
                aspect = 5,
                cmap = my_cmap)
    
    # Adding labels
    ax.set_xlabel('X-axis')
    ax.set_xlim(-5, 5)
    ax.set_ylabel('Y-axis')
    ax.set_ylim(-5, 5)
    ax.set_zlabel('Z-axis')
    ax.set_zlim(np.min(z), np.max(z))
    ax.set_title('3D surface having 2D contour plot projections')
    
    # show plot
    plt.show()


def question3():
    """No code required for this question"""
    print("Question 3 does not require any code, it is a theoretical question")
    # Just for fun, as nothing to do with question 3
    # (was my previous answer to question 2 and I wanted to keep it).
    # Create a new window
    root = tk.Tk()
    root.title("Question 1")
    root.geometry("1000x1000")
    # Create a frame for the left and right column
    left_frame = tk.Frame(root)
    right_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT)
    right_frame.pack(side=tk.RIGHT)
    # Create a figure
    fig = Figure(figsize=(10, 10), dpi=100)
    # Create a subplot
    ax = fig.add_subplot(111)
    # Plot the image on the right
    Fhat = np.zeros([128, 128])
    F = np.fft.ifft2(Fhat)
    Fshift = np.fft.fftshift(Fhat)
    showgrey(Fshift, display=False, plot=ax)
    # Create a canvas
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    # Create a tkinter canvas on the left side
    canvas2 = tk.Canvas(left_frame, width=128, height=128, bg="black")
    # Make the canvas drawable with the mouse
    canvas2.bind("<Button-1>", lambda event: draw(event, canvas2, ax, canvas))
    canvas2.pack()
    # Start the GUI
    root.mainloop()


def question4():
    """No code required for this question"""
    print("Question 4 does not require any code, it is a theoretical question")


def question5():
    """Plot a sinewave with a frequency of 1 Hz with sampling points at double the frequency"""
    # Create a figure
    fig = plt.figure()
    # Create a subplot
    ax = fig.add_subplot(111)
    # Create a time vector
    t = np.linspace(0, 10, 10000)
    # Create a sinewave with a frequency of 1 Hz
    y = np.sin(2 * np.pi * 1 * t)
    # Plot the sinewave
    ax.plot(t, y)
    # Plot sampling points
    ax.plot(t[::300], y[::300], "ro")
    # Set the title and labels
    ax.set_title("A sinewave with a frequency of 1 Hz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    # Show the plot
    plt.show()


def question6():
    """No code required for this question"""
    print("Question 6 does not require any code, it is a theoretical question")


def question7():
    F = np.concatenate(
        [np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    G = F.T
    H = F + 2*G
    fig1 = plt.figure()
    showgrey(F, display=False, plot=fig1.add_subplot(131))
    showgrey(G, display=False, plot=fig1.add_subplot(132))
    showgrey(H, display=False, plot=fig1.add_subplot(133))
    # Create fourier transform of each
    Fhat = fft2(F)
    Ghat = fft2(G)
    Hhat = fft2(H)
    # Create figure
    fig2 = plt.figure()
    showgrey(np.log(1 + np.abs(Fhat)), display=False,
             plot=fig2.add_subplot(131))
    showgrey(np.log(1 + np.abs(Ghat)), display=False,
             plot=fig2.add_subplot(132))
    showgrey(np.log(1 + np.abs(Hhat)), display=False,
             plot=fig2.add_subplot(133))
    # Create figure
    fig3 = plt.figure()
    showgrey(np.log(1 + np.abs(fftshift(Hhat))),
             display=False, plot=fig3.add_subplot(131))
    # plot everything
    plt.show()


def question8():
    """No code required for this question"""
    print("Question 8 does not require any code, it is a theoretical question")


def question9():
    """No code required for this question"""
    print("Question 9 does not require any code, it is a theoretical question")


def question10():
    # ASK MATTHEW CUZ IM CONFUZZELD
    F = np.concatenate(
        [np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    G = F.T
    fig1 = plt.figure()
    showgrey(F*G, display=False, plot=fig1.add_subplot(131))
    showfs(fft2(F*G), display=False, plot=fig1.add_subplot(132))
    # showgrey(ifft2(convolve2d(fft2(F), fft2(G), mode="same")),
    #        display=False, plot=fig1.add_subplot(133))
    plt.show()


def question11():
    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
         np.concatenate([np.zeros((128, 48)), np.ones(
             (128, 32)), np.zeros((128, 48))], axis=1)
    fig1 = plt.figure()
    showgrey(F, display=False, plot=fig1.add_subplot(121))
    showfs(fft2(F), display=False, plot=fig1.add_subplot(122))
    plt.show()


def question12():
    # Ask matthew why I don't get back the orignal image or at least close to it
    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
         np.concatenate([np.zeros((128, 48)), np.ones(
             (128, 32)), np.zeros((128, 48))], axis=1)

    # Iterate through different rotation angles
    for i in (0, 30, 45, 60, 90):
        # Rotate F by i degrees
        G = rot(F, i)
        # Create the FFT of G
        Ghat = fft2(G)
        # Rotate it back by -i degrees while centered
        Hhat = rot(fftshift(Ghat), -i)
        Hhat_not_shifted = rot(Ghat, -i)
        # Create the inverse FFT
        H = np.real(np.fft.ifft2(Hhat_not_shifted))
        # Create figure
        fig = plt.figure()
        # Plot the rotated image
        showgrey(G, display=False, plot=fig.add_subplot(141))
        # Plot the rotated image's FFT
        showfs(Ghat, display=False, plot=fig.add_subplot(142))
        # Plot the rotated back image
        showgrey(np.log(1 + abs(Hhat)), display=False,
                 plot=fig.add_subplot(143))
        # Plot the inverse FFT
        showfs(H, display=False, plot=fig.add_subplot(144))

    # Plot everything
    plt.show()


def question13():
    img = np.load(
        "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/phonecalc128.npy")
    fig1 = plt.figure()
    showgrey(img, display=False, plot=fig1.add_subplot(141))
    showfs(fft2(img), display=False, plot=fig1.add_subplot(142))
    showgrey(pow2image(img), display=False, plot=fig1.add_subplot(143))
    showgrey(randphaseimage(img), display=False, plot=fig1.add_subplot(144))
    plt.show()


def question14():
    df = deltafcn(128, 128)
    fig = plt.figure()
    nrow = 5
    ncol = 2
    for i, var in enumerate([0.1, 0.3, 1.0, 10.0, 100.0]):
        G, hFFT = gaussfft(df, var)
        p = fig.add_subplot(nrow, ncol, i*ncol+1)
        p.set_title('Gaussian blur using FFT of Dirac function with variance = ' + str(var))
        showgrey(G, display=False, plot=p)
        showgrey(hFFT, display=False,
                 plot=fig.add_subplot(nrow, ncol, i*ncol+2))
        # Calculate covariance matrix
        cov = variance(G)
        # Print the covariance matrix
        print("Covariance matrix for sigma = " + str(var) + " is: ")
        print(cov)
    plt.show()


def question15():
    df = deltafcn(128, 128)
    fig = plt.figure()
    nrow = 5
    ncol = 2
    for i, sigma in enumerate([0.1, 0.3, 1.0, 10.0, 100.0]):
        G, ffft = discgaussfft(df, sigma)
        showgrey(G, display=False, plot=fig.add_subplot(nrow, ncol, i*ncol+1))
        showgrey(ffft, display=False,
                 plot=fig.add_subplot(nrow, ncol, i*ncol+2))
        # Calculate covariance matrix
        cov = variance(G)
        # Print the covariance matrix
        print("Covariance matrix for sigma = " + str(sigma) + " is: ")
        print(cov)
    plt.show()


def question16():
    img = np.load(
        "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/phonecalc128.npy")
    # Low pass filter using gaussfft with various blur factors
    fig = plt.figure()
    p_orig = fig.add_subplot(1, 6, 1)
    p_orig.set_title('Original image')
    showgrey(img, display=False, plot=p_orig)
    for i, blur in enumerate([1, 4, 16, 64, 256]):
        G, _ = gaussfft(img, blur)
        p = fig.add_subplot(1, 6, i+2)
        p.set_title('Blur = ' + str(blur))
        showgrey(G, display=False, plot=p)
    plt.show()


def question17():
    office = np.load(
        "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/office256.npy")
    fig = plt.figure()
    add = gaussnoise(copy.deepcopy(office), 16)
    sap = sapnoise(copy.deepcopy(office), 0.1, 255)
    showgrey(office, display=False, plot=(p:=fig.add_subplot(3, 3, 1)))
    p.set_title('Original')
    showgrey(add, display=False, plot=(p:=fig.add_subplot(3, 3, 2)))
    p.set_title('Gaussian noise')
    showgrey(sap, display=False, plot=(p:=fig.add_subplot(3, 3, 3)))
    p.set_title('Salt and pepper noise')

    add_denoise_gauss_filt, _ = gaussfft(copy.deepcopy(add), 5)
    add_denoise_median_filt = medfilt(copy.deepcopy(add), 5)
    add_denoise_low_pass_filt = ideal(copy.deepcopy(add), 0.15)

    showgrey(add_denoise_gauss_filt, display=False,
             plot=(p:=fig.add_subplot(3, 3, 4)))
    p.set_title('Gaussian noise denoised with Gaussian filter')
    showgrey(add_denoise_median_filt, display=False,
             plot=(p:=fig.add_subplot(3, 3, 5)))
    p.set_title('Gaussian noise denoised with median filter')
    showgrey(add_denoise_low_pass_filt, display=False,
             plot=(p:=fig.add_subplot(3, 3, 6)))
    p.set_title('Gaussian noise denoised with ideal low pass filter')

    sap_denoise_gauss_filt, _ = gaussfft(copy.deepcopy(sap), 2)
    sap_denoise_median_filt = medfilt(copy.deepcopy(sap), 5)
    sap_denoise_low_pass_filt = ideal(copy.deepcopy(sap), 0.15)

    showgrey(sap_denoise_gauss_filt, display=False,
             plot=(p:=fig.add_subplot(3, 3, 7)))
    p.set_title('Salt and pepper noise denoised with Gaussian filter')
    showgrey(sap_denoise_median_filt, display=False,
             plot=(p:=fig.add_subplot(3, 3, 8)))
    p.set_title('Salt and pepper noise denoised with median filter')
    showgrey(sap_denoise_low_pass_filt, display=False,
             plot=(p:=fig.add_subplot(3, 3, 9)))
    p.set_title('Salt and pepper noise denoised with ideal low pass filter')
    plt.show()


def question18():
    """No code required for this question"""
    print("Question 18 does not require any code, it is a theoretical question")


def question19():
    img = np.load("/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-npy/phonecalc256.npy") 
    smoothimg1 = img
    smoothimg2 = img
    N = 5
    f = plt.figure()
    f.subplots_adjust(wspace=0, hspace=0)
    for i in range(N):
        if i > 0:  # generate subsampled versions
            img = rawsubsample(img)
            smoothimg1, _ = gaussfft(smoothimg1, 0.63)
            smoothimg1 = rawsubsample(smoothimg1)
            smoothimg2, _ = gaussfft(smoothimg2, 1)
            smoothimg2 = rawsubsample(smoothimg2)
        f.add_subplot(3, N, i + 1)
        showgrey(img, False)
        f.add_subplot(3, N, i + N + 1)
        showgrey(smoothimg1, False)
        f.add_subplot(3, N, i + 2 * N + 1)
        showgrey(smoothimg2, False)
    plt.show()


def question20():
    pass


funct_dict = {"Question 1": question1, "Question 2": question2, "Question 3": question3, "Question 4": question4, "Question 5": question5, "Question 6": question6, "Question 7": question7, "Question 8": question8, "Question 9": question9, "Question 10": question10,
              "Question 11": question11, "Question 12": question12, "Question 13": question13, "Question 14": question14, "Question 15": question15, "Question 16": question16, "Question 17": question17, "Question 18": question18, "Question 19": question19, "Question 20": question20}


def main():
    # Based on what part of the lab you are interested in, this lets you pick the part to run
    # There are 3 parts to the lab, you can only pick one part at a time
    # Get name of the part to run
    picked_part = pick(
        [f'Part {i}' for i in range(1, 4)], 'Pick a part to run')[0]
    questions = []
    if picked_part == "Part 1":
        questions = [f"Question {i}" for i in range(1, 14)]
    elif picked_part == "Part 2":
        questions = [f"Question {i}" for i in range(14, 17)]
    elif picked_part == "Part 3":
        questions = [f"Question {i}" for i in range(17, 21)]
    questions.insert(0, "All")
    # Get all picked question names in a list of strings
    picked_questions = [q[0] for q in sorted(
        pick(questions,
             'Pick question(s) to run (to select multiple question at once use the spacebar)',
             multiselect=True),
        key=lambda x: x[1])]
    # Run all questions if "All" is picked
    if "All" in picked_questions:
        picked_questions = questions[1:]

    # Run the picked questions
    print(f"Starting {picked_part}...")
    for question in picked_questions:
        print(f"Running {question}...")
        funct_dict[question]()
    print("Completed!")


if __name__ == '__main__':
    main()
