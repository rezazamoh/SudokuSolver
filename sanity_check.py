import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SudokuCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("Running Sanity Check")
print("=" * 50)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(loader))

print("Images Shape :", images.shape)
print("Labels Shape :", labels.shape)
print("Labels :", labels.tolist())

model = SudokuCNN().to(DEVICE)

images = images.to(DEVICE)

with torch.no_grad():
    outputs = model(images)

print("Output Shape :", outputs.shape)

num_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total Parameters :", f"{num_params:,}")
print("Trainable Parameters :", f"{trainable_params:,}")

print("=" * 50)
print("Sanity Check Passed ✔")
print("=" * 50)