import numpy as np

from collections.abc import Iterable
from math import gcd
import math


class _Sentinel:
    """Sentinel value meaning 'no filter applied'."""
    def __repr__(self): return "ALL"

_ALL = _Sentinel()
ALL = _Sentinel()


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


def compare_complex(arr1, arr2, precision=9):
    """
    Scales well for large arrays. Rounds to 'precision' decimals 
    to neutralize floating point noise before sorting.
    """
    if len(arr1) != len(arr2):
        return False
    
    def normalize(c):
        # Round both parts to create a stable sort key
        return (round(c.real, precision), round(c.imag, precision))

    # Sort based on the rounded values
    sort_key = lambda c: normalize(c)
    
    s1 = sorted(arr1, key=sort_key)
    s2 = sorted(arr2, key=sort_key)

    
    # Compare rounded versions
    for c1, c2 in zip(s1, s2):
        if normalize(c1) != normalize(c2):

            miss_a, miss_b = get_complex_mismatches(s1, s2)
            
            if miss_b:
                print(f"# of mismatched elements: {len(miss_b)}")
                print("\n[!] Elements present in B but missing/extra relative to A:")
                for val in miss_b:
                    print(f"  - {val}")
            return False    

    return True

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