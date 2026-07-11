def split_cells(board):

    cells=[]

    step=50

    for i in range(9):

        row=[]

        for j in range(9):

            cell=board[
                i*step:(i+1)*step,
                j*step:(j+1)*step
            ]

            row.append(cell)

        cells.append(row)

    return cells