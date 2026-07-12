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
NUM_CLASSES = 10 # 0 (Empty) + 1-9 (Digits)

def make_folders():
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    for split in ["train", "test"]:
        for cls in range(NUM_CLASSES):
            os.makedirs(
                os.path.join(OUTPUT_PATH, split, str(cls)),
                exist_ok=True
            )

def filter_out_zero_digit(data):
    return [(img, int(lbl)) for img, lbl in data if int(lbl) != 0]

def save_dataset(data, split):
    counter = [0] * NUM_CLASSES

    for image, label in tqdm(data, desc=f"Saving {split}", unit="img"):
        label = int(label)

        if not 0 <= label < NUM_CLASSES:
            continue

        filename = f"{counter[label]:06d}.png"
        path = os.path.join(OUTPUT_PATH, split, str(label), filename)

        cv2.imwrite(path, image)
        counter[label] += 1

def add_empty_cells(train, test):
    generator = EmptyCellGenerator()
    
    train_target = len(train) // 9
    test_target = len(test) // 9

    print(f"Generating {train_target} empty train images (Label 0)...")
    for _ in tqdm(range(train_target), desc="Train Empty Cells", unit="img"):
        train.append((generator.generate(), 0))

    print(f"Generating {test_target} empty test images (Label 0)...")
    for _ in tqdm(range(test_target), desc="Test Empty Cells", unit="img"):
        test.append((generator.generate(), 0))

def main():
    print("=" * 60)
    print("Sudoku Dataset Converter - Multi-Source")
    print("=" * 60)

    # 1. Load
    mnist_train, mnist_test = MNISTLoader("raw_data/MNIST").load()
    hoda_train, hoda_test = HODALoader("raw_data/HODA").load()
    chars_train, chars_test = Chars74KLoader("raw_data/Chars74K").load()

    # 2. Filter Digit 0
    print("\nFiltering digit '0' from source datasets...")
    train_digits = filter_out_zero_digit(mnist_train + hoda_train + chars_train)
    test_digits = filter_out_zero_digit(mnist_test + hoda_test + chars_test)

    # 3. Add Empty Cells
    train = train_digits
    test = test_digits
    add_empty_cells(train, test)

    # 4. Shuffle & Save
    random.shuffle(train)
    random.shuffle(test)

    make_folders()
    save_dataset(train, "train")
    save_dataset(test, "test")

    print(f"\n✓ Dataset created successfully at '{OUTPUT_PATH}'")
    print(f"Total Train: {len(train):,}")
    print(f"Total Test : {len(test):,}")
    
    final_counts = Counter(l for _, l in train)
    print("\nClass Distribution (Train):")
    for i in range(10):
        print(f"  Class {i}: {final_counts[i]} images")

if __name__ == "__main__":
    main()
