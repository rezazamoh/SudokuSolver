import cv2
import numpy as np
import random

IMAGE_SIZE = 28


class EmptyCellGenerator:

    def __init__(self):
        pass

    def generate(self):

        img = np.ones(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8
        ) * 255

        # نویز ملایم
        noise = np.random.normal(
            0,
            random.randint(2, 8),
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        img = img.astype(np.float32)

        img += noise

        img = np.clip(
            img,
            0,
            255
        ).astype(np.uint8)

        # گاهی خطوط جدول
        if random.random() < 0.4:

            thickness = random.randint(1, 2)

            color = random.randint(120, 180)

            cv2.line(
                img,
                (0, 0),
                (31, 0),
                color,
                thickness
            )

            cv2.line(
                img,
                (0, 0),
                (0, 31),
                color,
                thickness
            )

        # Blur
        if random.random() < 0.3:

            img = cv2.GaussianBlur(
                img,
                (3, 3),
                0
            )

        return img