import numpy as np
import pandas as pd
import os
from models import *
from basis_models import *


def print_L_basis(basis):
    """Prints Liouville space basis states (ket-bra)

    Args:
        basis (list of floats or SymStates): Hilbert space basis
    """

    for alpha in range(len(basis)**2):
        i = alpha % len(basis)
        j = alpha // len(basis)
        print(f"  alpha={alpha}: |{basis[i]}><{basis[j]}|")
        

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
