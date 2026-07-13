import cv2
import numpy as np

import torch
import torch.nn.functional as F
from models.cnn import SudokuCNN
from dataset_converter.preprocess import preprocess_image
from image_processing.extract_digit import extract_digit


class Predictor:

    def __init__(
        self,
        model_path,
        device=None
    ):

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = device

        self.model = SudokuCNN()

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    def _preprocess(self, image):

        # اگر تصویر رنگی بود
        if len(image.shape) == 3:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        digit = extract_digit(image)

        if digit is None:
            return None

        image = preprocess_image(digit)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # (1,32,32)
        image = np.expand_dims(
            image,
            axis=0
        )

        image = torch.from_numpy(image)

        # (1,1,32,32)
        image = image.unsqueeze(0)

        image = image.to(self.device)

        return image

    def predict(self, image):

        image = self._preprocess(image)

        if image is None:
            return (
                0,
                1.0,
                None
            )

        with torch.no_grad():

            outputs = self.model(image)

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        return (
            prediction.item(),
            confidence.item(),
            probabilities.squeeze().cpu().numpy()
        )