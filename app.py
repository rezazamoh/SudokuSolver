import os

import cv2
import numpy as np
import streamlit as st

from image_processing.preprocess import preprocess
from image_processing.grid_detector import find_grid
from image_processing.perspective import warp
from image_processing.split_cells import split_cells
from inference.predictor import Predictor
from solver import solve_sudoku, is_board_valid

# Configure the Streamlit page
st.set_page_config(page_title="Smart Sudoku Solver", layout="wide")

st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Smart Sudoku Detection and Solver</h1>
        <p>Upload a Sudoku image so the system can process, detect, and solve it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Path to the trained model weights
MODEL_PATH = "weights/best_model.pth"


@st.cache_resource
def load_predictor(model_path):
    # Cache the predictor so the model is loaded only once
    if os.path.exists(model_path):
        return Predictor(model_path=model_path)
    return None


predictor = load_predictor(MODEL_PATH)

if predictor is None:
    st.warning(
        f"Model weights were not found at `{MODEL_PATH}`. "
        "Please train the model first or place the weights at this path."
    )

# Let the user upload an image
uploaded_file = st.file_uploader(
    "Choose an image (PNG, JPG, JPEG)...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # Convert the uploaded file into an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    debug_images = {}
    with st.spinner("Processing image and extracting Sudoku board..."):
        try:
            # Preprocess the input image
            gray, blur, thresh = preprocess(image)

            # Detect the Sudoku grid corners
            corners = find_grid(thresh)

            # Warp both color and grayscale versions of the board
            board_color = warp(image, corners)
            board_gray = warp(gray, corners)

            # Split the warped board into a 2D list of cells
            cells_2d = split_cells(board_gray)

            # Flatten the 9x9 grid into a flat list of 81 cells
            cells = [cell for row in cells_2d for cell in row]

            with col2:
                st.subheader("Extracted Board")
                st.image(
                    cv2.cvtColor(board_color, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

            # Show intermediate computer vision steps
            with st.expander("Show Image Processing Steps"):
                mid_col1, mid_col2 = st.columns(2)

                with mid_col1:
                    st.text("Thresholded Image")
                    st.image(thresh, use_container_width=True)

                with mid_col2:
                    st.text("Extracted Cells (9×9)")

                    if len(cells) == 81:

                        cols = st.columns(9)

                        for i, cell in enumerate(cells):

                            with cols[i % 9]:
                                st.caption(f"{i//9},{i%9}")
                                st.image(
                                    cell,
                                    clamp=True,
                                    use_container_width=True,
                                )

                            if (i + 1) % 9 == 0 and i != 80:
                                cols = st.columns(9)

            # Run digit recognition if the model is available
            if predictor is not None:
                grid = np.zeros((9, 9), dtype=int)
                progress_bar = st.progress(0, text="Recognizing digits...")

                for i in range(81):
                    row, col = divmod(i, 9)
                    cell = cells[i]

                    # Predictor handles digit extraction internally
                    pred_digit, confidence, probabilities = predictor.predict(cell)
                    grid[row][col] = pred_digit

                    progress_bar.progress((i + 1) / 81)

                progress_bar.empty()

                st.subheader("Detected Grid")
                grid_col1, grid_col2 = st.columns(2)

                with grid_col1:
                    st.write("Initial Matrix:")
                    st.dataframe(grid)

                # Copy the detected board before solving
                board_to_solve = grid.copy()

                if not is_board_valid(board_to_solve):
                    st.error(
                        "The detected grid is inconsistent. "
                        "There may have been an error in digit recognition."
                    )
                else:
                    with st.spinner("Solving Sudoku..."):
                        if solve_sudoku(board_to_solve):
                            with grid_col2:
                                st.write("Solved Grid:")
                                st.dataframe(board_to_solve)
                                st.success("Sudoku solved successfully.")
                        else:
                            st.error("No solution was found for this Sudoku.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
