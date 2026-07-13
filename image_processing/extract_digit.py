import cv2
import numpy as np


def extract_digit(cell):

    # Gray
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell.copy()

    # حذف خطوط دور خانه
    h, w = gray.shape
    margin = 4
    gray = gray[margin:h-margin, margin:w-margin]

    # Binary
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # اگر کانتور خیلی کوچک بود خانه خالی است
    if cv2.contourArea(cnt) < 40:
        return None

    x, y, w, h = cv2.boundingRect(cnt)

    digit = thresh[y:y+h, x:x+w]

    return digit