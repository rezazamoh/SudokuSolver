import cv2
import numpy as np


def remove_grid_lsd(board, debug=False):

    # Gray
    if len(board.shape) == 3:
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    else:
        gray = board.copy()

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Adaptive Threshold
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

    lsd = cv2.createLineSegmentDetector()

    lines = lsd.detect(thresh)[0]

    mask = np.zeros_like(thresh)

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            length = np.hypot(dx, dy)

            # فقط خطوط بلند
            if length < cell_size * 0.7:
                continue

            # خط افقی
            if dy < 5:

                cv2.line(
                    mask,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    255,
                    3
                )

            # خط عمودی
            elif dx < 5:

                cv2.line(
                    mask,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    255,
                    3
                )

    # کمی ضخیم‌تر کردن ماسک
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )
    )

    board_without_grid = cv2.bitwise_and(
        thresh,
        cv2.bitwise_not(mask)
    )

    result = 255 - board_without_grid

    if debug:

        cv2.imwrite(
            "output/debug_threshold.png",
            thresh
        )

        cv2.imwrite(
            "output/debug_lsd_mask.png",
            mask
        )

        cv2.imwrite(
            "output/debug_removed_grid.png",
            result
        )

    return result