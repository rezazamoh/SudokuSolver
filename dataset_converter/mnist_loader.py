from torchvision import datasets
import numpy as np

class MNISTLoader:
    def __init__(self, raw_path):
        self.raw_path = raw_path

    def load(self):
        train_set = datasets.MNIST(root=self.raw_path, train=True, download=True)
        test_set = datasets.MNIST(root=self.raw_path, train=False, download=True)

        train_images = train_set.data.numpy()
        train_labels = train_set.targets.numpy()

        test_images = test_set.data.numpy()
        test_labels = test_set.targets.numpy()

        train_data = [(img, int(label)) for img, label in zip(train_images, train_labels)]
        test_data = [(img, int(label)) for img, label in zip(test_images, test_labels)]

        return train_data, test_data
