import cv2
import numpy as np


def extract_digit(cell):

    # ---------------- Gray ----------------
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(
            cell,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = cell.copy()


    h, w = gray.shape


    # ---------------- Threshold ----------------
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )


    # حذف نویز خیلی ریز
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        np.ones((2,2), np.uint8)
    )


    # ---------------- Connected Components ----------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh,
        connectivity=8
    )


    # ---------------- Center Region Check ----------------
    cx = w // 2
    cy = h // 2

    # حدود 40 درصد وسط سلول
    mx = int(w * 0.20)
    my = int(h * 0.20)

    # ---------------- Center Occupancy Test ----------------
    center_binary = thresh[
        cy-my:cy+my,
        cx-mx:cx+mx
    ]

    pixels = cv2.countNonZero(center_binary)

    ratio = pixels / center_binary.size

    # اگر وسط تقریباً خالی است
    if ratio < 0.12:
        return None, False

    # حالا فقط Labelهای داخل مرکز را پیدا کن
    center_labels = labels[
        cy-my:cy+my,
        cx-mx:cx+mx
    ]

    labels_in_center = np.unique(center_labels)
    labels_in_center = labels_in_center[labels_in_center != 0]

    if len(labels_in_center) == 0:
        return None, False

    # ---------------- انتخاب Component مرکزی ----------------
    best_label = None
    best_area = 0


    for label in labels_in_center:

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area > best_area:
            best_area = area
            best_label = label


    if best_label is None:
        return None, False



    # ---------------- Crop Digit ----------------
    x = stats[
        best_label,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        best_label,
        cv2.CC_STAT_TOP
    ]

    ww = stats[
        best_label,
        cv2.CC_STAT_WIDTH
    ]

    hh = stats[
        best_label,
        cv2.CC_STAT_HEIGHT
    ]


    # حذف موارد خیلی کشیده (خط جدول)
    if ww > 0.8*w and hh < 0.2*h:
        return None, False

    if hh > 0.8*h and ww < 0.2*w:
        return None, False



    pad = 3

    x = max(
        0,
        x-pad
    )

    y = max(
        0,
        y-pad
    )


    ww = min(
        w-x,
        ww + 2*pad
    )

    hh = min(
        h-y,
        hh + 2*pad
    )


    digit = gray[
        y:y+hh,
        x:x+ww
    ]


    return digit, True