import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from models.cnn import SudokuCNN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DIR = "dataset/test"

MODEL_PATH = "weights/best_model.pth"

BATCH_SIZE = 256

print("Device:", DEVICE)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = SudokuCNN(num_classes=19).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


y_true = []
y_pred = []


os.makedirs(
    "output/misclassified",
    exist_ok=True
)


counter = 0


with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        pred = outputs.argmax(1)

        y_true.extend(labels.numpy())
        y_pred.extend(pred.cpu().numpy())

        for i in range(len(pred)):

            if pred[i] != labels[i]:

                img = images[i].cpu().squeeze().numpy()

                plt.imsave(
                    f"output/misclassified/{counter}_T{labels[i].item()}_P{pred[i].item()}.png",
                    img,
                    cmap="gray"
                )

                counter += 1


print(classification_report(
    y_true,
    y_pred,
    digits=4
))


with open(
    "output/classification_report.txt",
    "w",
    encoding="utf8"
) as f:

    f.write(
        classification_report(
            y_true,
            y_pred,
            digits=4
        )
    )


cm = confusion_matrix(
    y_true,
    y_pred
)

# Draw a confusion matrix with blue diagonal and red off-diagonal cells
fig, ax = plt.subplots(figsize=(8, 8))

mask = np.eye(cm.shape[0], dtype=bool)

# Normalize blue using the full matrix range, but normalize red using only off-diagonal errors.
blue_norm = plt.Normalize(vmin=cm.min(), vmax=cm.max())
red_values = cm[~mask]
red_max = float(red_values.max()) if red_values.size > 0 else 1.0
red_norm = plt.Normalize(vmin=0.0, vmax=max(red_max, 1.0))

cm_blue = plt.cm.Blues(blue_norm(cm))
cm_red = plt.cm.Reds(red_norm(cm))

# Build an RGB image: diagonal cells use blue, off-diagonal cells use red
image = np.zeros_like(cm_blue)
image[mask] = cm_blue[mask]
image[~mask] = cm_red[~mask]

ax.imshow(image, aspect='equal')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix (blue=correct, red=errors)')

# Add ticks and labels
labels = np.arange(cm.shape[0])
ax.set_xticks(labels)
ax.set_yticks(labels)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Annotate counts, with text color that contrasts against background
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text_color = 'white' if image[i, j, :3].mean() < 0.5 else 'black'
        ax.text(j, i, cm[i, j], ha='center', va='center', color=text_color)

plt.tight_layout()
plt.savefig(
    "output/confusion_matrix.png",
    dpi=300
)
plt.close()


accuracy = np.mean(
    np.array(y_true) == np.array(y_pred)
)

print(f"\nAccuracy : {accuracy*100:.2f}%")