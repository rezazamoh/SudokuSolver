import os
import random
import matplotlib.pyplot as plt
import cv2

DATASET_PATH = "dataset/train"

classes = sorted(os.listdir(DATASET_PATH))

print("=" * 50)
print("Dataset Statistics")
print("=" * 50)

for cls in classes:

    folder = os.path.join(DATASET_PATH, cls)

    images = [
        f for f in os.listdir(folder)
        if f.endswith(".png")
    ]

    print(f"Class {cls}: {len(images)} images")

print("=" * 50)

fig, axes = plt.subplots(
    len(classes),
    10,
    figsize=(15, 15)
)

for row, cls in enumerate(classes):

    folder = os.path.join(DATASET_PATH, cls)

    images = [
        f for f in os.listdir(folder)
        if f.endswith(".png")
    ]

    samples = random.sample(
        images,
        min(10, len(images))
    )

    for col in range(10):

        ax = axes[row][col]

        ax.axis("off")

        if col >= len(samples):
            continue

        img = cv2.imread(
            os.path.join(folder, samples[col]),
            cv2.IMREAD_GRAYSCALE
        )

        ax.imshow(
            img,
            cmap="gray"
        )

        if col == 0:
            ax.set_ylabel(
                cls,
                fontsize=12
            )

plt.tight_layout()

plt.show()