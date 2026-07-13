import cv2
import numpy as np


def remove_grid(board, debug=False):

    if len(board.shape) == 3:
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    else:
        gray = board.copy()

    # ---------- CLAHE ----------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # ---------- Threshold ----------
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        7
    )

    # ---------- Remove small noise ----------
    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    # ---------- Detect lines ----------
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=40,
        maxLineGap=15
    )

    grid_mask = np.zeros_like(thresh)

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Horizontal
            if dy < 4:

                cv2.line(
                    grid_mask,
                    (x1, y1),
                    (x2, y2),
                    255,
                    5
                )

            # Vertical
            elif dx < 4:

                cv2.line(
                    grid_mask,
                    (x1, y1),
                    (x2, y2),
                    255,
                    5
                )

    # ---------- Thicken mask ----------
    grid_mask = cv2.dilate(
        grid_mask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )
    )

    # ---------- Remove grid ----------
    digits = cv2.bitwise_and(
        thresh,
        cv2.bitwise_not(grid_mask)
    )

    digits = cv2.morphologyEx(
        digits,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8)
    )

    result = 255 - digits

    if debug:

        cv2.imwrite("output/debug_threshold.png", thresh)
        cv2.imwrite("output/debug_grid_mask.png", grid_mask)
        cv2.imwrite("output/debug_removed_grid.png", result)

    return result