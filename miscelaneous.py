import numpy as np
import pandas as pd
import os

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

def csv_results(csv_path, L, N, J, U, gamma, dissipation, M, evals, z):
    """Append sector statistics to a CSV file, creating it if it doesn't exist."""

    row = {
        "L": L,
        "N": N,
        "J": J,
        "U": U,
        "gamma": gamma,
        "dissipation": dissipation,
        "M": M,
        "#evals": len(evals),
        "r_mean": round(np.abs(z).mean(), 4)
    }

    df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)