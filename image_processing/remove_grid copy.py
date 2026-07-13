import cv2
import numpy as np


def remove_grid(board, debug=False):

    if len(board.shape) == 3:
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    else:
        gray = board.copy()

    # -----------------------------
    # CLAHE
    # -----------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # -----------------------------
    # Adaptive Threshold
    # -----------------------------
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        7
    )

    h, w = thresh.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (cell_size // 2, 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, cell_size // 2)
    )

    horizontal = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    candidates = cv2.bitwise_or(
        horizontal,
        vertical
    )

    contours, _ = cv2.findContours(
        candidates,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    x,y,w,h = cv2.boundingRect(contours)
    grid_mask = np.zeros_like(candidates)
    

    cv2.drawContours(
        grid_mask,
        [contours],
        -1,
        255,
        thickness=5
    )

    digits = cv2.bitwise_and(
        thresh,
        cv2.bitwise_not(grid_mask)
    )

    digits = cv2.morphologyEx(
        digits,
        cv2.MORPH_OPEN,
        np.ones((2,2),np.uint8)
    )
    # دوباره به تصویر معمولی برگردان
    result = 255 - digits

    if debug:

        cv2.imwrite(
            "output/debug_clahe.png",
            gray
        )

        cv2.imwrite(
            "output/debug_threshold.png",
            thresh
        )

        cv2.imwrite(
            "output/debug_horizontal.png",
            horizontal
        )

        cv2.imwrite(
            "output/debug_vertical.png",
            vertical
        )

        cv2.imwrite(
            "output/debug_grid.png",
            grid_mask
        )

        cv2.imwrite(
            "output/debug_removed_grid.png",
            result
        )

    return result