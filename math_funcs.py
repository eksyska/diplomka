import numpy as np

from collections.abc import Iterable
from math import gcd


def fmt_ket(s):
    return "|" + ",".join(str(n) for n in s) + ">"

def fmt_bra(s):
    return "<" + ",".join(str(n) for n in s) + "|"



def lcm(a, b):
    """Returns the lowest common multiple of a, b

    Args:
        a (int):
        b (int):

    Returns:
        int: the lowest common multiple of a, b
    """
    
    return a * b // gcd(a, b)

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