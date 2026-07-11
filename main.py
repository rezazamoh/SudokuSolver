import cv2
import os

from image_processing.preprocess import preprocess
from image_processing.grid_detector import find_grid
from image_processing.perspective import warp
from image_processing.split_cells import split_cells


image=cv2.imread("images/sudoku.png")

gray,blur,thresh=preprocess(image)

corners=find_grid(thresh)

board=warp(image,corners)

cells=split_cells(board)

os.makedirs("output",exist_ok=True)

cv2.imwrite("output/gray.png",gray)

cv2.imwrite("output/threshold.png",thresh)

cv2.imwrite("output/board.png",board)

for i in range(9):
    for j in range(9):

        cv2.imwrite(
            f"output/cell_{i}_{j}.png",
            cells[i][j]
        )

print("Done")