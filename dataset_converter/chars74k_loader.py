import os
import cv2
import random
from .preprocess import preprocess_image

class Chars74KLoader:
    def __init__(self, base_path, train_ratio=0.8):
        self.base_path = base_path
        self.train_ratio = train_ratio

    def load(self):
        all_data = []

        for digit in range(10):
            folder_path = os.path.join(self.base_path, str(digit))

            if not os.path.exists(folder_path):
                print(f"   Warning: folder not found: {folder_path}, skipping...")
                continue

            print(f"   Loading digit {digit} from: {folder_path}")

            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)

                if not os.path.isfile(img_path):
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                processed = preprocess_image(img)

                if processed is not None:
                    all_data.append((processed, digit))

        random.shuffle(all_data)

        split_idx = int(len(all_data) * self.train_ratio)
        train_data = all_data[:split_idx]
        test_data = all_data[split_idx:]

        return train_data, test_data
