import cv2
import numpy as np
import os


def remove_grid_lsd(board, debug=False):

    # Convert to grayscale if the board image is colored
    if len(board.shape) == 3:
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    else:
        gray = board.copy()

    # Apply CLAHE to improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # Convert the image to binary using adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        7
    )

    h, w = thresh.shape
    cell_size = min(h, w) // 9

    # ---------------- LSD ----------------
    # Create a Line Segment Detector
    lsd = cv2.createLineSegmentDetector()
    
    # Detect line segments safely
    output = lsd.detect(thresh)
    lines = output[0]

    # Create an empty mask for detected grid lines
    mask = np.zeros_like(thresh)

    if lines is not None:
        for line in lines:
            # Flatten the output to safely extract coordinates
            coords = line.flatten()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = coords

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.hypot(dx, dy)

            # Keep only sufficiently long lines
            # These are likely to be Sudoku grid lines
            if length < cell_size * 0.7:
                continue

            # Keep nearly horizontal or nearly vertical lines
            # with a tolerance of 10 pixels
            if dy < 10 or dx < 10:
                cv2.line(
                    mask,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    255,
                    3  # Mask thickness
                )

    # Dilate the mask slightly to fully cover grid edges
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    # Remove the detected grid lines from the binary image
    board_without_grid = cv2.bitwise_and(
        thresh,
        cv2.bitwise_not(mask)
    )

    # Invert back so digits become dark on a bright background
    result = 255 - board_without_grid

    if debug:
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/debug_threshold.png", thresh)
        cv2.imwrite("output/debug_lsd_mask.png", mask)
        cv2.imwrite("output/debug_removed_grid.png", result)

    return result
