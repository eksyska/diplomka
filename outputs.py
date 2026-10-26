import numpy as np
import pandas as pd
import os
import ast
from models import *
from basis_models import *


def export_matrix(matrix, filename="matrix_output"):
    """Exports matrix to a .txt file

    Args:
        matrix (2darray): matrix to export
    """

    path = f"data/{filename}.txt"
    with open(path, "w") as f:

        for row in matrix:

            formatted_row = " ".join(f"{val.real:.1f}{val.imag:+.1f}j" for val in row)
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


###################################### EIGENVALUE STORAGE ######################################

def save_block_evals(path, block_evals, params):
    """Stores the eigenvalues of all blocks with their sector labels to one .npz file

    The eigenvalues of all blocks are concatenated, block_sizes and sectors keep which block
    every eigenvalue belongs to (the statistics is computed per block).
    The file is written under a temporary name and renamed afterwards, so an interrupted
    run never leaves a half-written .npz file that would look finished.

    Args:
        path (str): .npz file
        block_evals (dict): sector label -> eigenvalues
        params (dict): parameters stored with the eigenvalues (scalars or strings)
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"

    labels = list(block_evals.keys())

    with open(tmp, "wb") as fh:
        np.savez(fh,
                 evals=np.concatenate([block_evals[k] for k in labels]),
                 block_sizes=np.array([len(block_evals[k]) for k in labels]),
                 sectors=np.array([repr(k) for k in labels]),
                 **params)
        fh.flush()
        os.fsync(fh.fileno())

    os.replace(tmp, path)


def load_block_evals(path):
    """Loads a file stored by save_block_evals

    Returns:
        (dict, dict): sector label (tuple) -> eigenvalues, and the stored parameters
    """

    with np.load(path) as data:
        splits = np.cumsum(data["block_sizes"])[:-1]
        sectors = [ast.literal_eval(k) for k in data["sectors"].tolist()]
        block_evals = dict(zip(sectors, np.split(data["evals"], splits)))
        params = {k: data[k].item() for k in data.files if k not in ("evals", "block_sizes", "sectors")}

    return block_evals, params
