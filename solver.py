def is_valid(board, row, col, num):
    # Check whether the number already exists in the same row or column
    for i in range(9):
        if board[row][i] == num and i != col:
            return False
        if board[i][col] == num and i != row:
            return False

    # Check whether the number already exists in the same 3x3 box
    start_row = 3 * (row // 3)
    start_col = 3 * (col // 3)

    for i in range(3):
        for j in range(3):
            r = start_row + i
            c = start_col + j

            if board[r][c] == num and (r != row or c != col):
                return False

    return True


def find_empty(board):
    # Find the first empty cell, represented by 0
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c

    return None


def solve_sudoku(board):
    # Find the next empty cell
    empty = find_empty(board)

    if not empty:
        return True  # The board is fully solved

    row, col = empty

    # Try numbers from 1 to 9
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            # Continue solving the rest of the board recursively
            if solve_sudoku(board):
                return True

            # Backtrack if the current number does not lead to a solution
            board[row][col] = 0

    return False


def is_board_valid(board):
    # Validate the detected board before trying to solve it
    for r in range(9):
        for c in range(9):
            val = board[r][c]

            if val != 0:
                # Temporarily remove the value so it does not compare with itself
                board[r][c] = 0
                valid = is_valid(board, r, c, val)
                board[r][c] = val

                if not valid:
                    return False

    return True
