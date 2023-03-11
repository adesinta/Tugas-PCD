import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Perform 2D FFT on the image
f = np.fft.fft2(img)

# Shift the zero-frequency component to the center of the spectrum
fshift = np.fft.fftshift(f)

# Compute the magnitude spectrum (logarithmic scale)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Display the original and magnitude spectrum images
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
