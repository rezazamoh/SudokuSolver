import cv2
import numpy as np

def extract_digit(cell_gray):
    # 1. Apply adaptive thresholding to get a clean binary image
    # Note: Using grayscale input from main.py as discussed
    thresh = cv2.adaptiveThreshold(
        cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 2. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, False

    # 3. Find the largest contour which should be the digit
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Ignore very small contours (noise)
    if w < 5 or h < 10:
        return None, False

    # 4. Extract the digit ROI
    digit_roi = thresh[y:y+h, x:x+w]

    # 5. Add Padding to prevent the digit from touching the borders
    # Touching borders makes '7' look like '6' to the CNN
    pad = int(max(w, h) * 0.2) # 20% padding
    digit_padded = cv2.copyMakeBorder(
        digit_roi, pad, pad, pad, pad, 
        cv2.BORDER_CONSTANT, value=0
    )

    return digit_padded, True
