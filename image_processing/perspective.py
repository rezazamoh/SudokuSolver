import cv2
import numpy as np


def order_points(points):

    pts = points.reshape(4,2)

    rect = np.zeros((4,2),dtype="float32")

    s = pts.sum(axis=1)

    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]

    diff=np.diff(pts,axis=1)

    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]

    return rect


def warp(image,points):

    rect=order_points(points)

    dst=np.array([
        [0,0],
        [449,0],
        [449,449],
        [0,449]
    ],dtype="float32")

    M=cv2.getPerspectiveTransform(rect,dst)

    warped=cv2.warpPerspective(
        image,
        M,
        (450,450)
    )

    return warped