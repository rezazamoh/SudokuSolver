import os

from raw_data.HODA.HodaDatasetReader import read_hoda_cdb
from dataset_converter.preprocess import preprocess_image


class HODALoader:

    def __init__(self, root):
        self.root = root

    def load(self):
        train_images, train_labels = read_hoda_cdb(
            os.path.join(self.root, "Train 60000.cdb")
        )

        test_images, test_labels = read_hoda_cdb(
            os.path.join(self.root, "Test 20000.cdb")
        )

        train = self._prepare(train_images, train_labels)
        test = self._prepare(test_images, test_labels)

        return train, test

    def _prepare(self, images, labels):
        data = []

        for img, label in zip(images, labels):
            label_int = int(label)

            if not 1 <= label_int <= 9:
                continue

            img = preprocess_image(img)
            data.append((img, label_int + 9))

        return data
