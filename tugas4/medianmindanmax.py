import cv2
import numpy as np

def median_filter(image, kernel_size):
    # apply median filter to image using kernel size
    return cv2.medianBlur(image, kernel_size)

# load image in grayscale
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# apply median filter with kernel size 3x3
filtered_image = median_filter(image, 3)

# display original and filtered images
cv2.imshow('Original', image)
cv2.imshow('Median Filter', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

def min_filter(image, kernel_size):
    # apply min filter to image using kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel)

# load image in grayscale
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# apply min filter with kernel size 5x5
filtered_image = min_filter(image, 5)

# display original and filtered images
cv2.imshow('Original', image)
cv2.imshow('Min Filter', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def max_filter(image, kernel_size):
    # apply max filter to image using kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel)

# load image in grayscale
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# apply max filter with kernel size 7x7
filtered_image = max_filter(image, 5)

# display original and filtered images
cv2.imshow('Original', image)
cv2.imshow('Max Filter', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



