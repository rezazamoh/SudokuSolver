import cv2
import numpy as np

def split_cells(image):
    cells = []
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate cell size
    cell_h = height // 9
    cell_w = width // 9
    
    # Define a margin to avoid grid lines (approx 10% of cell size)
    margin_h = int(cell_h * 0.1)
    margin_w = int(cell_w * 0.1)

    for i in range(9):
        row = []
        for j in range(9):
            # Calculate coordinates for each cell
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            
            # Crop the cell
            cell = image[y1:y2, x1:x2]
            
            # Apply margin to remove grid remnants from the edges
            # This is crucial for avoiding '6' vs '7' confusion caused by grid lines
            cell_clean = cell[margin_h:-margin_h, margin_w:-margin_w]
            
            row.append(cell_clean)
        cells.append(row)
        
    return cells
