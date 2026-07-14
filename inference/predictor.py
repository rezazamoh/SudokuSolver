import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from debug_pipeline import save_pipeline
from models.cnn import SudokuCNN
from dataset_converter.preprocess import preprocess_image
from image_processing.extract_digit import extract_digit


class Predictor:

    def __init__(self, model_path, device=None, debug=False):
        # Select device automatically (GPU if available, else CPU)
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.debug = debug
        self.debug_counter = 0
        self.raw_digit = None
        self.processed = None
        # Define model architecture and load saved weights
        self.model = SudokuCNN()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image):
        # Convert colored image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Extract the digit region from the Sudoku cell
        digit, has_digit = extract_digit(gray)
        self.raw_digit = None if digit is None else digit.copy()

        if not has_digit or digit is None:
            return None

        # Ignore very small regions because they are probably noise
        if digit.size < 50:
            return None

        # Save raw extracted digit for debugging using original file name format
        if self.debug:
            os.makedirs("output/extracted_digits", exist_ok=True)
            cv2.imwrite(
                f"output/extracted_digits/{self.debug_counter}_raw.png",
                digit
            )

        # Convert the extracted digit to the same format used during training
        processed_digit = preprocess_image(digit)
        self.processed = image.copy()
        # Save processed digit for debugging using original file name format
        if self.debug:
            cv2.imwrite(
                f"output/extracted_digits/{self.debug_counter}_proc.png",
                processed_digit
            )

        self.debug_counter += 1

        # Normalize pixel values to the range [0, 1]
        processed_digit = processed_digit.astype(np.float32) / 255.0

        # Add channel dimension: (32, 32) -> (1, 32, 32)
        processed_digit = np.expand_dims(processed_digit, axis=0)
        tensor_img = torch.from_numpy(processed_digit)

        # Add batch dimension: (1, 32, 32) -> (1, 1, 32, 32)
        tensor_img = tensor_img.unsqueeze(0)
        tensor_img = tensor_img.to(self.device)

        return tensor_img

    def predict(self, image):
        # Preprocess the Sudoku cell image
        processed_img = self._preprocess(image)

        # If cell is empty, return class 0 (empty) with confidence 1.0
        if processed_img is None:
            probabilities = np.zeros(10, dtype=np.float32)
            probabilities[0] = 1.0
            return 0, 1.0, probabilities

        # Run inference using the neural network
        with torch.no_grad():
            outputs = self.model(processed_img)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)


        save_pipeline(
            raw_cell=image,
            digit=self.raw_digit,
            processed=self.processed,
            pred=prediction.item(),
            conf=confidence.item()
        )
        return (
            prediction.item(),
            confidence.item(),
            probabilities.squeeze().cpu().numpy()
        )
