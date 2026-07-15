import os
import cv2
from image_processing.preprocess import preprocess
from image_processing.grid_detector import find_grid
from image_processing.perspective import warp
from image_processing.split_cells import split_cells
from image_processing.remove_grid_lsd import remove_grid_lsd
from inference.predictor import Predictor
from solver import solve_sudoku, is_board_valid


IMAGE_PATH = "images/4845408.jpg"
BEST_MODEL_PATH = "weights/best_model.pth"
ENGLISH_MODEL_PATH = "weights/10classesMINST.pth"
FARSI_MODEL_PATH = "weights/10classesHoda.pth"


def main():
    # Check required files
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Input image was not found at: {IMAGE_PATH}")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights were not found at: {BEST_MODEL_PATH}. Train the model first."
        )

    if not os.path.exists(ENGLISH_MODEL_PATH):
        raise FileNotFoundError(
            f"English model weights were not found at: {ENGLISH_MODEL_PATH}."
        )

    if not os.path.exists(FARSI_MODEL_PATH):
        raise FileNotFoundError(
            f"Farsi model weights were not found at: {FARSI_MODEL_PATH}."
        )

    # Create output folders
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/cells", exist_ok=True)

    # Load models (language detection + language-specific digit recognition)
    predictor = Predictor(BEST_MODEL_PATH, ENGLISH_MODEL_PATH, FARSI_MODEL_PATH, debug=True)

    # Read image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise ValueError(f"Could not read image from: {IMAGE_PATH}")

    # Save original image
    cv2.imwrite("output/original.png", image)

    # Preprocess image
    gray, blur, thresh = preprocess(image)
    cv2.imwrite("output/gray.png", gray)
    cv2.imwrite("output/blur.png", blur)
    cv2.imwrite("output/threshold.png", thresh)

    # Detect grid and warp board
    corners = find_grid(thresh)
    board_color = warp(image, corners)
    board_gray = warp(gray, corners)

    cv2.imwrite("output/board_color.png", board_color)
    cv2.imwrite("output/board_gray.png", board_gray)

    # Remove grid lines from the color board, then split into cells
    board_no_grid = remove_grid_lsd(board_color, debug=True)
    cv2.imwrite("output/board_no_grid.png", board_no_grid)

    cells = split_cells(board_no_grid)

    print("Detecting digits from Sudoku cells...")

    sudoku_grid = []
    all_probabilities = []

    for i in range(9):
        row_digits = []
        row_probabilities = []

        for j in range(9):
            cell_img = cells[i][j]

            # Save each cell for debugging
            cv2.imwrite(f"output/cells/{i}_{j}.png", cell_img)

            # Save one sample cell like your first script
            if i == 0 and j == 0:
                cv2.imwrite("output/test_predict.png", cell_img)

            # First call will auto-detect board language using 19-class model
            # Subsequent calls will use the detected language
            digit, confidence, probabilities = predictor.predict(
                cell_img, save_debug_pipeline=False
            )

            print(f"Cell ({i},{j}) -> {digit} ({confidence:.3f})")

            row_digits.append(digit)
            row_probabilities.append(probabilities)

        sudoku_grid.append(row_digits)
        all_probabilities.append(row_probabilities)

    print("\n--- Detected Sudoku Grid ---")
    for row in sudoku_grid:
        print(" ".join(map(str, row)))

    # Validate before solving
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
