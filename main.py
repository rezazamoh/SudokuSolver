import cv2
import os

from image_processing.preprocess import preprocess
from image_processing.grid_detector import find_grid
from image_processing.perspective import warp
from image_processing.split_cells import split_cells
from inference.predictor import Predictor
from solver import solve_sudoku

IMAGE_PATH = "images/sudoku.png"
MODEL_PATH = "weights/best_model.pth"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"No {IMAGE_PATH} exists.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No {MODEL_PATH} exists. Run train.py first.")

predictor = Predictor(MODEL_PATH)

image = cv2.imread(IMAGE_PATH)

gray, blur, thresh = preprocess(image)
corners = find_grid(thresh)
board = warp(image, corners)
cells = split_cells(board)

os.makedirs("output", exist_ok=True)
cv2.imwrite("output/gray.png", gray)
cv2.imwrite("output/threshold.png", thresh)
cv2.imwrite("output/board.png", board)

print("Predicting cells...")
sudoku_grid = []

for i in range(9):
    row_digits = []
    for j in range(9):
        digit, confidence, _ = predictor.predict(cells[i][j])
        row_digits.append(digit)
    sudoku_grid.append(row_digits)

print("\n--- Board detected by CNN ---")
for row in sudoku_grid:
    print(row)

print("\nSolving sudoku...")
grid_copy = [row[:] for row in sudoku_grid]

if solve_sudoku(grid_copy):
    print("\n--- Solved Sudoku ---")
    for row in grid_copy:
        print(row)
else:
    print("\nNo solution exists for the detected board. Please check detection accuracy.")
