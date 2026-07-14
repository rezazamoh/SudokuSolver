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
        self.model = SudokuCNN(num_classes=19)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _class_to_digit(predicted_class):
        if predicted_class == 0:
            return 0
        if 1 <= predicted_class <= 9:
            return predicted_class
        if 10 <= predicted_class <= 18:
            return predicted_class - 9
        raise ValueError(f"Invalid predicted class: {predicted_class}")

    @classmethod
    def class_to_digit(cls, predicted_class):
        return cls._class_to_digit(predicted_class)

    @staticmethod
    def _class_language(predicted_class):
        if 1 <= predicted_class <= 9:
            return "english"
        if 10 <= predicted_class <= 18:
            return "farsi"
        return None

    @staticmethod
    def _digit_language_class(digit, language):
        if digit == 0:
            return 0
        if language == "english":
            return digit
        return digit + 9

    @staticmethod
    def detect_board_language(all_probabilities):
        english_count = 0
        farsi_count = 0
        english_sum = 0.0
        farsi_sum = 0.0

        for probabilities in all_probabilities:
            if probabilities is None:
                continue

            best_class = int(np.argmax(probabilities))
            if best_class == 0:
                continue

            if 1 <= best_class <= 9:
                english_count += 1
            elif 10 <= best_class <= 18:
                farsi_count += 1

            english_sum += float(np.max(probabilities[1:10]))
            farsi_sum += float(np.max(probabilities[10:19]))

        if english_count + farsi_count == 0:
            return None

        if english_count != farsi_count:
            return "english" if english_count > farsi_count else "farsi"

        return "english" if english_sum >= farsi_sum else "farsi"

    @classmethod
    def language_corrected_prediction(cls, probabilities, board_language):
        if probabilities is None:
            return 0, 1.0

        raw_best = int(np.argmax(probabilities))
        if raw_best == 0 or board_language is None:
            return cls.class_to_digit(raw_best), float(probabilities[raw_best])

        raw_language = cls._class_language(raw_best)
        if raw_language == board_language:
            return cls.class_to_digit(raw_best), float(probabilities[raw_best])

        raw_digit = cls._class_to_digit(raw_best)
        board_class = cls._digit_language_class(raw_digit, board_language)

        if float(probabilities[board_class]) >= float(probabilities[raw_best]):
            return cls.class_to_digit(board_class), float(probabilities[board_class])

        return cls.class_to_digit(raw_best), float(probabilities[raw_best])

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
        self.processed = processed_digit.copy()

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

    def predict(self, image, board_language=None, save_debug_pipeline=True):
        # Preprocess the Sudoku cell image
        processed_img = self._preprocess(image)

        # If cell is empty, return digit 0 with confidence 1.0
        if processed_img is None:
            probabilities = np.zeros(19, dtype=np.float32)
            probabilities[0] = 1.0
            if save_debug_pipeline:
                save_pipeline(
                    raw_cell=image,
                    digit=self.raw_digit,
                    processed=self.processed,
                    pred=0,
                    conf=1.0,
                )
            return 0, 1.0, probabilities

        # Run inference using the neural network
        with torch.no_grad():
            outputs = self.model(processed_img)
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

        if board_language is not None:
            digit, confidence = self.language_corrected_prediction(
                probabilities, board_language
            )
        else:
            raw_best = int(np.argmax(probabilities))
            digit = self._class_to_digit(raw_best)
            confidence = float(probabilities[raw_best])

        if save_debug_pipeline:
            save_pipeline(
                raw_cell=image,
                digit=self.raw_digit,
                processed=self.processed,
                pred=digit,
                conf=confidence,
            )

        return digit, confidence, probabilities
