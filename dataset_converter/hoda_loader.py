import os

from raw_data.HODA.HodaDatasetReader import read_hoda_cdb

from dataset_converter.preprocess import preprocess_image


class HODALoader:

    def __init__(self, root):

        self.root = root

    def load(self):

        train_images, train_labels = read_hoda_cdb(
            os.path.join(
                self.root,
                "DigitDB",
                "Train 60000.cdb"
            )
        )

        test_images, test_labels = read_hoda_cdb(
            os.path.join(
                self.root,
                "DigitDB",
                "Test 20000.cdb"
            )
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

            # حذف رقم صفر
            if int(label) == 0:
                continue

            img = preprocess_image(img)

            data.append(
                (
                    img,
                    int(label) + 9
                )
            )

        return data