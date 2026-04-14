import numpy as np
import pandas as pd
import os
from models import *


def export_matrix(matrix, filename="matrix_output"):
    """Exports matrix to a .txt file

    Args:
        matrix (2darray): matrix to export
    """

    path = f"data/{filename}.txt"
    with open(path, "w") as f:

        for row in matrix:

            formatted_row = " ".join(f"{val:.2f}" for val in row)
            f.write(formatted_row + "\n")

    print(f"Matrix saved to {path}")

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
