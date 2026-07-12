import os
import cv2

from dataset_converter.preprocess import preprocess_image


class Chars74KLoader:

    def __init__(self, root):

        self.root = root

    def load(self):

        train = []
        test = []

        # Sample002 تا Sample010
        for digit in range(1, 10):

            folder = os.path.join(
                self.root,
                "English",
                "Fnt",
                f"Sample{digit+1:03d}"
            )

            images = []

            for file in sorted(os.listdir(folder)):

                if not file.lower().endswith((".png", ".jpg", ".bmp")):
                    continue

                path = os.path.join(folder, file)

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = preprocess_image(img)

                images.append((img, digit))

            split = int(len(images) * 0.8)

            train.extend(images[:split])

            test.extend(images[split:])

        return train, test