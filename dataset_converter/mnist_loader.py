import struct
import numpy as np

from dataset_converter.preprocess import preprocess_image


class MNISTLoader:

    def __init__(self, root):

        self.root = root

    def load(self):

        train_images = self._read_images(
            self.root + "/train-images.idx3-ubyte"
        )

        train_labels = self._read_labels(
            self.root + "/train-labels.idx1-ubyte"
        )

        test_images = self._read_images(
            self.root + "/t10k-images.idx3-ubyte"
        )

        test_labels = self._read_labels(
            self.root + "/t10k-labels.idx1-ubyte"
        )

        train = self._prepare(
            train_images,
            train_labels
        )

        test = self._prepare(
            test_images,
            test_labels
        )

        return train, test

    def _prepare(
        self,
        images,
        labels
    ):

        data = []

        for img, label in zip(images, labels):

            # صفر حذف می‌شود
            if label == 0:
                continue

            img = 255 - img

            img = preprocess_image(img)

            data.append(
                (
                    img,
                    int(label)
                )
            )

        return data

    def _read_images(self, filename):

        with open(filename, "rb") as f:

            magic, size, rows, cols = struct.unpack(
                ">IIII",
                f.read(16)
            )

            images = np.frombuffer(
                f.read(),
                dtype=np.uint8
            )

            images = images.reshape(
                size,
                rows,
                cols
            )

        return images

    def _read_labels(self, filename):

        with open(filename, "rb") as f:

            magic, size = struct.unpack(
                ">II",
                f.read(8)
            )

            labels = np.frombuffer(
                f.read(),
                dtype=np.uint8
            )

        return labels