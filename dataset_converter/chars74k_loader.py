import os
import random
import cv2

from dataset_converter.preprocess import preprocess_image


class Chars74KLoader:

    def __init__(self, root, test_split=0.2, seed=42):
        self.root = root
        self.test_split = test_split
        self.seed = seed

    def load(self):
        data = []

        for label in range(1, 10):
            folder = os.path.join(self.root, str(label))

            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")

            for file_name in sorted(os.listdir(folder)):
                path = os.path.join(folder, file_name)

                if not os.path.isfile(path):
                    continue

                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = preprocess_image(image)
                data.append((image, label))

        rng = random.Random(self.seed)
        rng.shuffle(data)

        split_idx = int(len(data) * (1 - self.test_split))
        train = data[:split_idx]
        test = data[split_idx:]

        return train, test
