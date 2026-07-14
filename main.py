import cv2
import os
from image_processing.preprocess import preprocess
from image_processing.grid_detector import find_grid
from image_processing.perspective import warp
from image_processing.split_cells import split_cells
from inference.predictor import Predictor
from solver import solve_sudoku, is_board_valid


IMAGE_PATH = "images/4845408.jpg"
MODEL_PATH = "weights/best_model.pth"


def main():
    # Check whether the input image exists
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Input image was not found at: {IMAGE_PATH}")

    # Check whether the trained model weights exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights were not found at: {MODEL_PATH}. Train the model first."
        )

    # Initialize the digit predictor and enable debug output
    predictor = Predictor(MODEL_PATH, debug=True)

    # Read the original image
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise ValueError(f"Could not read image from: {IMAGE_PATH}")

    # Preprocess the image to get grayscale, blurred, and thresholded versions
    gray, blur, thresh = preprocess(image)

    # Detect the four corners of the Sudoku grid
    corners = find_grid(thresh)

    # Apply perspective transform to both the original and grayscale images
    board_color = warp(image, corners)
    board_gray = warp(gray, corners)

    # Save warped board images for debugging
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/board_color.png", board_color)
    cv2.imwrite("output/board_gray.png", board_gray)

    # Split the warped grayscale board into 81 individual cells
    cells = split_cells(board_gray)

    os.makedirs("output/cells", exist_ok=True)

    sudoku_grid = []
    print("Detecting digits from Sudoku cells...")

    for i in range(9):
        row_digits = []

        for j in range(9):
            cell_img = cells[i][j]

            # Save each raw cell image for debugging
            cv2.imwrite(f"output/cells/{i}_{j}.png", cell_img)

            # Predict the digit inside the current cell
            digit, confidence, _ = predictor.predict(cell_img)

            row_digits.append(digit)

        sudoku_grid.append(row_digits)

    print("\n--- Detected Sudoku Grid ---")
    for row in sudoku_grid:
        print(" ".join(map(str, row)))

    # Validate the detected grid before solving it
    if not is_board_valid(sudoku_grid):
        print("\n[Error] The detected Sudoku grid is invalid.")
        print("There is a conflict in a row, column, or 3x3 box.")
        print("Please check the input image quality or digit recognition model.")
        return

    print("\nSolving Sudoku...")
    grid_copy = [row[:] for row in sudoku_grid]

    if solve_sudoku(grid_copy):
        print("\n--- Solved Sudoku Grid ---")
        for row in grid_copy:
            print(" ".join(map(str, row)))
    else:
        print("\nNo solution was found for the detected Sudoku grid.")


if __name__ == "__main__":
    main()
