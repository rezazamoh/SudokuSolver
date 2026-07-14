import os
import cv2
import random
import shutil
from tqdm import tqdm
from collections import Counter

from dataset_converter.mnist_loader import MNISTLoader
from dataset_converter.hoda_loader import HODALoader
from dataset_converter.chars74k_loader import Chars74KLoader
from dataset_converter.empty_generator import EmptyCellGenerator


OUTPUT_PATH = "dataset"

def check_dataset(name, data):
    labels = [label for _, label in data]
    print(f"\n{name}")
    print("Min:", min(labels))
    print("Max:", max(labels))
    print("Unique:", sorted(set(labels))[:30])
    print("Count:", len(set(labels)))

def make_folders():

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    for split in ["train", "test"]:

        for cls in range(19):

            os.makedirs(
                os.path.join(
                    OUTPUT_PATH,
                    split,
                    str(cls)
                ),
                exist_ok=True
            )


def save_dataset(data, split):

    counter = [0] * 19

    for image, label in tqdm(
        data,
        desc=f"Saving {split}",
        unit="img"
    ):

        filename = f"{counter[label]:06d}.png"

        path = os.path.join(
            OUTPUT_PATH,
            split,
            str(label),
            filename
        )

        cv2.imwrite(path, image)

        counter[label] += 1

def add_empty_cells(train, test):

    generator = EmptyCellGenerator()

    train_counter = Counter(label for _, label in train)
    test_counter = Counter(label for _, label in test)

    train_target = int(
        sum(train_counter.values()) / len(train_counter)
    )

    test_target = int(
        sum(test_counter.values()) / len(test_counter)
    )

    print(f"Generating {train_target} empty train images...")

    for _ in tqdm(
        range(train_target),
        desc="Train Empty Cells",
        unit="img"
    ):
        train.append((generator.generate(), 0))

    print(f"Generating {test_target} empty test images...")

    for _ in tqdm(
        range(test_target),
        desc="Test Empty Cells",
        unit="img"
    ):
        test.append((generator.generate(), 0))

def main():

    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    print("✓ Loading MNIST")
    mnist_train, mnist_test = MNISTLoader("raw_data/MNIST").load()

    print(f"   Train: {len(mnist_train):,}")
    print(f"   Test : {len(mnist_test):,}\n")

    print("✓ Loading HODA")
    hoda_train, hoda_test = HODALoader("raw_data/HODA").load()

    print(f"   Train: {len(hoda_train):,}")
    print(f"   Test : {len(hoda_test):,}\n")

    print("✓ Loading Chars74K")
    chars_train, chars_test = Chars74KLoader("raw_data/Chars74K").load()

    print(f"   Train: {len(chars_train):,}")
    print(f"   Test : {len(chars_test):,}\n")


    check_dataset("MNIST", mnist_train)
    check_dataset("HODA", hoda_train)
    check_dataset("Chars74K", chars_train)


    print("\nMNIST first label:", mnist_train[0][1], type(mnist_train[0][1]))
    print("HODA first label:", hoda_train[0][1], type(hoda_train[0][1]))
    print("Chars first label:", chars_train[0][1], type(chars_train[0][1]))
    train = (
        mnist_train +
        hoda_train +
        chars_train
    )

    test = (
        mnist_test +
        hoda_test +
        chars_test
    )

    add_empty_cells(train, test)

    random.shuffle(train)
    random.shuffle(test)

    make_folders()

    #save_dataset(train, "train")
    #save_dataset(test, "test")

    print()

    print("Dataset created successfully.")

    print()

    print("Train:", len(train))

    print("Test :", len(test))


if __name__ == "__main__":

    main()