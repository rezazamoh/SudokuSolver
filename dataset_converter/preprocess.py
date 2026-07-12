import cv2
import numpy as np

IMAGE_SIZE = 32


def preprocess_image(img):
    """
    Standardize an input image from any dataset.
    Output:
        uint8 grayscale image
        size = 32x32
        black digit on white background
    """

    img = np.array(img)

    # اگر تصویر رنگی بود
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # باینری کردن با Otsu
    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # اطمینان از اینکه رقم سیاه و زمینه سفید است
    if np.mean(img) < 127:
        img = 255 - img

    # پیدا کردن ناحیه رقم
    coords = cv2.findNonZero(255 - img)

    if coords is not None:

        x, y, w, h = cv2.boundingRect(coords)

        img = img[y:y+h, x:x+w]

    h, w = img.shape

    size = max(h, w)

    canvas = np.ones(
        (size, size),
        dtype=np.uint8
    ) * 255

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    canvas[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = img

    canvas = cv2.resize(
        canvas,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

    return canvas