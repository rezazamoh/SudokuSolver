import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from models.cnn import SudokuCNN
from dataset_converter.preprocess import preprocess_image
from image_processing.extract_digit import extract_digit


class Predictor:

    def __init__(
        self,
        model_path,
        device=None,
        debug=False
    ):

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = device

        self.debug = debug
        self.debug_counter = 0
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

        digit, has_digit = extract_digit(image)

        if not has_digit:
            return None
        
        if digit is not None and digit.size < 50:
            digit = None
        
        if self.debug and digit is not None:
            os.makedirs("output/extracted_digits", exist_ok=True)

            cv2.imwrite(
                f"output/extracted_digits/{self.debug_counter:02d}.png",
                digit
            )

        self.debug_counter += 1

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