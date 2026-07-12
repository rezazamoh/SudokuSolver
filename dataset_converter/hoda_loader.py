import os
from raw_data.HODA.HodaDatasetReader import read_hoda_cdb
from .preprocess import preprocess_image

class HODALoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load(self):
        train_path = os.path.join(self.base_path, "Train 60000.cdb")
        test_path = os.path.join(self.base_path, "Test 20000.cdb")

        print(f"   Reading HODA Train from: {train_path}")
        train_images, train_labels = read_hoda_cdb(train_path)
        
        print(f"   Reading HODA Test from: {test_path}")
        test_images, test_labels = read_hoda_cdb(test_path)

        processed_train = []
        for img, lbl in zip(train_images, train_labels):
            p = preprocess_image(img)
            if p is not None:
                processed_train.append((p, lbl))

        processed_test = []
        for img, lbl in zip(test_images, test_labels):
            p = preprocess_image(img)
            if p is not None:
                processed_test.append((p, lbl))

        return processed_train, processed_test
