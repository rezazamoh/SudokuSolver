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


model = SudokuCNN(num_classes=10).to(DEVICE)

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


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

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