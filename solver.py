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


def find_conflicts(board):
    """Find all cells that create conflicts in the Sudoku board."""
    conflicts = []
    
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val != 0:
                # Check for conflicts with other cells
                # Check row
                for i in range(9):
                    if board[r][i] == val and i != c:
                        if (r, c) not in conflicts:
                            conflicts.append((r, c))
                        if (r, i) not in conflicts:
                            conflicts.append((r, i))
                
                # Check column
                for i in range(9):
                    if board[i][c] == val and i != r:
                        if (r, c) not in conflicts:
                            conflicts.append((r, c))
                        if (i, c) not in conflicts:
                            conflicts.append((i, c))
                
                # Check 3x3 box
                start_row = 3 * (r // 3)
                start_col = 3 * (c // 3)
                for i in range(3):
                    for j in range(3):
                        ri = start_row + i
                        cj = start_col + j
                        if board[ri][cj] == val and (ri != r or cj != c):
                            if (r, c) not in conflicts:
                                conflicts.append((r, c))
                            if (ri, cj) not in conflicts:
                                conflicts.append((ri, cj))
    
    return conflicts


def try_fix_farsi_2_3_conflicts(board, all_probabilities, confidence_threshold=0.97):
    """
    Try to fix conflicts by swapping detected 2s and 3s if they have low confidence.
    This handles the Farsi digit recognition issue where 2 and 3 often look similar.
    
    Args:
        board: The detected Sudoku grid
        all_probabilities: 9x9 list of probability arrays for each cell
        confidence_threshold: Confidence threshold below which we attempt 2/3 swap
    
    Returns:
        A corrected board if fixable, None if still has conflicts
    """
    conflicts = find_conflicts(board)
    print(conflicts)
    if not conflicts:
        return board  # No conflicts
    
    # Try to fix conflicts by swapping 2s and 3s with low confidence
    fixed_board = [row[:] for row in board]
    
    for r, c in conflicts:
        val = fixed_board[r][c]
        
        # Only try to fix if it's a 2 or 3
        if val not in [2, 3]:
            continue
        
        # Check confidence of this cell
        if all_probabilities[r][c] is not None:
            confidence = float(all_probabilities[r][c][val])
            print(val)
            print(f'row: {r}, col: {c}, confidence: {confidence}')
            print(f'confidence_threshold: {confidence_threshold}')
            # If confidence is low, try swapping 2<->3
            if confidence < confidence_threshold:
                # Swap: if it's 2, change to 3; if it's 3, change to 2
                new_val = 3 if val == 2 else 2
                print(f'  -> Changing to {new_val}')
                fixed_board[r][c] = new_val
    
    for row in fixed_board:
                        print(" ".join(map(str, row)))
    # Check if the fixed board resolves conflicts
    if find_conflicts(fixed_board):
        return None  # Still has conflicts
    
    return fixed_board


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
