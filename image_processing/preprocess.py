import cv2
import numpy as np


def preprocess(image):
    # Convert to grayscale if the input image is colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Small kernel for noise removal
    kernel = np.ones((2, 2), np.uint8)

    # Remove small noisy regions using morphological opening
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Apply median blur for additional denoising
    cleaned = cv2.medianBlur(cleaned, 3)

    return gray, blur, cleaned
