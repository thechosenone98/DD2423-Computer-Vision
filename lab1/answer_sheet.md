## Question 1

The x axis corresponds to horizontal frequencies and the y axis represents the vertical frequencies.
Combination of x and y result in a combination of both an horizontal and a vertical component and thus
produces a diagonal sine wave.

## Question 2

A single point in the Fourier domain represent one pure sine wave and thus is converted to a single pure
sine pattern in the time domain (a repeating black and white pattern on the image in the direction dictated by
the frequency component of the original pixel in the Fourier domain).

## Question 3

The amplitude of the signal is given by:

$$ x = \sqrt{Re^2[f\hat(u,v)] + Im^2[f\hat(u,v)]}$$

## Question 4

The angle of the signal is given by:

$$ \theta = \tan^{-1}(\frac{u}{v})$$

The wavelength of the signal is given by:

$$ T = \frac{1}{\sqrt{u^2 + v^2}}$$

## Question 5

Since the pixel represent the sampler of the signal, in order to sample the signal correctly we need to, at least, sample the signal at double it's maximum frequency. Past the halfway point of the FFT domain, all the frequencies are above that threshold.

## Question 6

It corrects the coordinates we gave for the centered version.

## Question 7

The line are either perfectly vertical or horizontal and thus require no signal going diagonally (represented on the FFT in 4 quadrants excluding the center). The line are thus represented by a single line of frequencies in the FFT domain. Combining both images simply combines their FFT together.

## Question 8

Since the value in the center of the FFT are so large compared to the rest of the higher frequency amplitude, we apply a logarithm to bring all the values in the same range (if you have values like 1 000 000 in the center and a bunch of small 10 in the higher frequencies, the small number will be remapped to zero if you don't apply a log and make those 10 become 1 an the 1 000 000 become a 6).

## Question 9

$$\hat{h}(u,v) = \hat{f}(u,v) + \hat{g}(u,v)$$

## Question 10

We can obtain the same result by first taking the Fourier transform of both images and then convolve them together
in the frequency domain. This work since convolution in the fourier domain is equivalent to point wise multiplication
in the spatial domain and vice versa.

## Question 11

Scaling down an image make the FFT grow larger since more high frequency are represented in the image. Scaling up an image make the FFT shrink since less high frequency are represented in the image.

## Question 12

Since the rotation is rescaling the image some frequencies are remapped incorrectly and thus some information is lost about the original image. Also since you cannot have a perfectly straight edge at various angles, those edge contain multiple frequencies that would be different if the image was in the continuous domain.

## Question 13

The phase controls where the edges are in the image, the magnitude only says how much of a certain frequency is part of the image. This is why phase is a lot more important for images (not so much for sound) since the edges are what makes an image an image.

## Question 14

Variances for different sigma values:

| Sigma | Variance |
|-------|----------|
| 0.1   | 0.013    |
| 0.3   | 0.281    |
| 1.0   | 0.999    |
| 10    | 10.00    |
| 100   | 99.99    |

## Question 15

The variance match the expected values for large variance values but not for smaller values.
This is due sampling error not being able to capture the rapid variation on the gaussian function.

## Question 16

The gaussian filter acts as a smooth low-pass filter. It removes high frequency components from the image and thus makes the image look smoother.

## Question 17

The gaussian filter will blur edges and thus can be more visually appealing but is not beneficial if you are trying to detect edges. The median filter is really good at removing salt and peper noise but makes the image look like a painting and it is thus less visually appealing, but it preserves edges which is really good for edge detection. The low pass filter adds a lot of ringing to the output image and does not do as good a job as the gaussian filter does.

## Question 18

It is important to understand what type of noise is in your image before your try and denoise it. If the noise is gaussian, then a gaussian filter is the best way to denoise it. If the noise is salt and pepper, then a median filter is the best way to denoise it.

## Question 19

If you subsample without smoothing, you get a very harsh picture that contains a lot of aliasing. If you subsample with smoothing, you get a smoother picture that is less aliased.

## Question 20

If you don't apply smoothing before subsampling you get what is called aliasing and it is caused by the high frequencies being remapped to lower frequencies and thus causing artifact in the image. On the other hand, if you smooth it out beforehand, you get a much nicer output since you are removing the high frequency component that anyway cant be represented in the smaller image and thus avoid aliasing. A good filter for that is:

$$\begin{bmatrix} 1/16 & 1/8 & 1/16 \\ 1/8 & 1/4 & 1/8 \\ 1/16 & 1/8 & 1/16 \end{bmatrix}$$