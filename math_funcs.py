import numpy as np

from collections.abc import Iterable
from math import gcd
import math

# empty objects for states
class _Sentinel:
    """Sentinel value meaning 'no filter applied'."""
    def __repr__(self): return "ALL"

_ALL = _Sentinel()
ALL = _Sentinel()


def fmt(z):
    return (
        f"{z.real:6.2f}"
        + (f"{z.imag:+6.2f}j" if not np.isclose(z.imag, 0) else "        ")
    )

# bra/ket strings
def fmt_ket(s):
    return "|" + ",".join(str(n) for n in s) + ">"
def fmt_bra(s):
    return "<" + ",".join(str(n) for n in s) + "|"


# lowest common multiple
def lcm(a, b):
    """Returns the lowest common multiple of a, b

    Args:
        a (int):
        b (int):

    Returns:
        int: the lowest common multiple of a, b
    """
    
    return a * b // gcd(a, b)

# commutator
def comm (A, B):
    """Computes commutator

    Args:
        A (Qobj or 2darray): matrix A
        B (Qobj or 2darray): matrix B

    Returns:
        Qobj or 2darray: commutator of A and B
    """
    return A @ B - B @ A


def clean_num_error(values, tol=1e-10):
    """Zero out real or imaginary parts of eigenvalues within numerical tolerance.

    Args:
        values (np.ndarray): complex eigenvalues
        tol (float): tolerance threshold

    Returns:
        np.ndarray: cleaned eigenvalues
    """

    if isinstance(values, Iterable):
        values = values.copy()
        values.real[np.abs(values.real) < tol] = 0.0
        values.imag[np.abs(values.imag) < tol] = 0.0
        
    else:
        if np.abs(values.real) < tol:
            values = 1j * values.imag
        if np.abs(values.imag) < tol:
            values = values.real

    return values


def compare_complex(arr1, arr2, precision=10):
    """
    Compares two complex arrays, their order doesn't matter.
    Used mainly to compare evals.

    Args:
    arr1 (np.array)
    arr2 (np.array)
    precision (int): Number of valid digits when rounding to compare. Defaults to 10.
    """
    if len(arr1) != len(arr2):
        return False
        
    idx1 = np.lexsort((arr1.imag, np.round(arr1.real, precision)))
    s1 = arr1[idx1]
    idx2 = np.lexsort((arr2.imag, np.round(arr2.real, precision)))
    s2 = arr2[idx2]
    
    is_eq = np.allclose(s1,s2)
    return is_eq


def get_complex_mismatches(arr1, arr2, rel_tol=1e-09, abs_tol=0.0):
    """
    Identifies elements that do not have a matching partner in the other array,
    accounting for floating-point noise.
    """
    # Copies to avoid mutating the original lists
    pool_a = list(arr1)
    pool_b = list(arr2)
    
    # We will track which items in A find a match in B
    mismatches_in_a = []
    
    # Iterate through A and try to 'consume' matches from B
    for val_a in pool_a:
        found_match = False
        for i, val_b in enumerate(pool_b):
            # Check if val_a and val_b are 'the same' within tolerance
            if math.isclose(val_a.real, val_b.real, rel_tol=rel_tol, abs_tol=abs_tol) and \
            math.isclose(val_a.imag, val_b.imag, rel_tol=rel_tol, abs_tol=abs_tol):
                pool_b.pop(i) # Match found, remove from B's pool
                found_match = True
                break
        
        if not found_match:
            mismatches_in_a.append(val_a)

    # Any items left in pool_b are the ones that had no match in A
    mismatches_in_b = pool_b
    
    return mismatches_in_a, mismatches_in_b


def find_blocks(matrices, tol=1e-10, sort_by_size=True):
    """Finds block-diagonal structure of one or several square matrices.

    Uses connected components of the (symmetrized) sparsity graph — finds the
    coarsest possible permutation into block-diagonal form, regardless of how
    the matrix was originally ordered. Useful e.g. to check whether a declared
    symmetry sector (kappa, pi, M) is actually irreducible, or still splits
    further into sub-blocks.

    Args:
        matrices (np.ndarray or dict): a single 2D array, or a dict mapping
            label -> 2D array (e.g. the `blocks` dict from bose_hubbard_L_blocks)
        tol (float): threshold below which an entry is treated as zero
        sort_by_size (bool): sort blocks largest-first within each matrix. Defaults to True.

    Returns:
        dict (single matrix) or dict of dict, keyed by the same labels as the
        input, each entry with:
            'n_blocks' (int): number of sub-blocks found
            'is_block_diagonal' (bool): True if n_blocks > 1 (i.e. the sector splits further)
            'blocks' (list of dict): one entry per sub-block, each with
                'indices' (np.ndarray): original row/col indices (within this matrix) in the sub-block
                'size' (int): sub-block dimension
                'block' (np.ndarray): the extracted submatrix
    """
    import scipy.sparse.csgraph as csg

    def _analyze(mat, label=None):
        mat = np.asarray(mat)
        n = mat.shape[0]
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"matrix must be square, got shape {mat.shape}" +
                              (f" for label {label}" if label is not None else ""))

        adjacency = (np.abs(mat) > tol) | (np.abs(mat.T) > tol)
        n_components, labels_arr = csg.connected_components(adjacency, directed=False)

        blocks = []
        for c in range(n_components):
            idx = np.where(labels_arr == c)[0]
            blocks.append({
                "indices": idx,
                "size": len(idx),
                "block": mat[np.ix_(idx, idx)],
            })

        if sort_by_size:
            blocks.sort(key=lambda b: -b["size"])

        tag = f" [{label}]" if label is not None else ""
        print(f"matrix size: {n}{tag}  ->  {n_components} block(s), sizes: "
              f"{sorted([b['size'] for b in blocks], reverse=True)}")

        return {
            "n_blocks": n_components,
            "is_block_diagonal": n_components > 1,
            "blocks": blocks,
        }

    if isinstance(matrices, dict):
        return {label: _analyze(mat, label=label) for label, mat in matrices.items()}

    elif isinstance(matrices, np.ndarray) and matrices.ndim == 2:
        return _analyze(matrices)

    else:
        # list/tuple/object-array of matrices, no labels
        return [_analyze(mat, label=i) for i, mat in enumerate(matrices)]