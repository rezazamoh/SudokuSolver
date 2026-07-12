import cv2
import os
import numpy as np

from image_processing.preprocess import preprocess
from image_processing.grid_detector import find_grid
from image_processing.perspective import warp
from image_processing.split_cells import split_cells
from predict import predict_digit
from solver import solve_sudoku

image_path = "images/sudoku.png"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"No {image_path} exists.")

image = cv2.imread(image_path)

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
        cell_img = cells[i][j]
        digit = predict_digit(cell_img)
        row_digits.append(digit)
    sudoku_grid.append(row_digits)

print("\n--- Board detected by CNN ---")
for r in sudoku_grid:
    print(r)

print("\nSolving sudoku...")
grid_copy = [row[:] for row in sudoku_grid]

if solve_sudoku(grid_copy):
    print("\n--- Solved Sudoku ---")
    for r in grid_copy:
        print(r)
else:
    print("\nNo solution exists for the detected board. Please check detection accuracy.")
