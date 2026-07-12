import cv2
import torch

from torchvision import transforms

from models.cnn import SudokuCNN

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SudokuCNN().to(device)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([

    transforms.ToPILImage(),
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

def predict_digit(cell):

    img = transform(cell)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(img)

        pred = torch.argmax(
            output,
            dim=1
        )

    return pred.item()