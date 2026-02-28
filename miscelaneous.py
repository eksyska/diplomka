import numpy as np

def export_matrix(matrix):
    """Exports matrix to a .txt file

    Args:
        matrix (2darray): matrix to export
    """

    with open("matrix_output.txt", "w") as f:
        for row in matrix:
            formatted_row = " ".join(f"{val.real:10.2f}" for val in row)
            f.write(formatted_row + "\n")

    print("Matrix saved to matrix_output.txt")