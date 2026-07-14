import cv2
import numpy as np

def find_grid(thresh_image):
    # 1. Find all contours in the thresholded image
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Sort contours by area in descending order and pick the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        # Calculate the perimeter of the contour
        peri = cv2.arcLength(cnt, True)
        
        # Approximate the contour to a polygon
        # epsilon is 2% of the perimeter; you can adjust this (0.01 to 0.05)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # The Sudoku grid should be the largest contour with exactly 4 corners
        if len(approx) == 4:
            # Flatten the array to ensure it's in the correct shape for warp
            return approx.reshape(4, 2)

    # Fallback: if no 4-corner contour is found, use the bounding box of the largest contour
    if contours:
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # Replace np.int0 (deprecated/removed in new NumPy versions) with np.intp
        box_int = np.intp(box)
        return box_int.reshape(4, 2)

    raise ValueError("Could not find a valid 4-corner Sudoku grid in the image.")
