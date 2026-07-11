import cv2

def find_grid(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest, True)

    approx = cv2.approxPolyDP(
        largest,
        epsilon,
        True
    )

    return approx