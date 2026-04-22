import numpy as np
import qutip as qt
import itertools

from scipy.sparse import lil_matrix

from math_funcs import *


###################################### BASIS BUILDING ######################################

class SymState:
    """instance of this class is an eigenstate of T, P operators
    """

    def __init__(self, states, coeffs, k, p=None):
        self.states = states
        self.coeffs = coeffs
        self.k = k
        self.p = p
        self.n = sum(states[0])

    def __str__(self):
        def fmt(c):
            if np.iscomplex(c):
                return f"{c.real:.2f}{c.imag:+.2f}j"
            return f"{float(c):.2f}"

        terms = " + ".join(
            f"{fmt(self.coeffs[i])} {self.states[i]}"
            for i in range(len(self.states))
        )
        return f"N={self.n} k={self.k} p={self.p} state = {terms}"
    
    __repr__ = __str__
    

def build_bose_basis(L, N, fixed_N=True, n_local_max=None):
    """Builds Bose basis in Fock space

    Args:
        L (int): number of sites
        N (int): total number of excitaions
        fixed_N (bool, optional): TRUE if all basis states have conserved # of excitations. Defaults to True.
        n_local_max (int): site cutoff. Default to None.

    Returns:
        list: basis states (int tuples)
    """

    if n_local_max is None:
        n_local_max = N 

    all_configs = itertools.product(range(n_local_max+1), repeat=L)

    if fixed_N:
        basis = [cfg for cfg in all_configs if sum(cfg) == N]

    else:
        basis = [cfg for cfg in all_configs if sum(cfg) <= N]

    return basis


def build_ket_orbits(basis_list):
    """Find all translation orbits of single Fock states.
 
    Args:
        basis_list (list of tuples): Fock basis states
 
    Returns:
        ket_orbits  (list of lists): each entry is the orbit [s, T(s), T²(s), ...] ordered from the seed (= lexicographically smallest element).
        ket_orbit_of (dict): state -> (orbit_index, position_within_orbit)
    """
    
    ket_orbits = []
 
    visited = set()
    for s in basis_list:

        if s in visited:
            continue

        orb = []
        cur = s
        while True:
            orb.append(cur)
            visited.add(cur)
            cur = translate(cur)
            if cur == s:
                break

        ket_orbits.append(orb)
 
    return ket_orbits


def build_sym_basis(basis_list):
    """Builds Fock basis from T, P operators eigenstates

    Args:
        basis_list (list of int tuples): oringinal Fock basis

    Returns:
        list of SymStates: new basis
    """

    L = len(basis_list[0]) #number of sites
    ket_orbits = build_ket_orbits(basis_list)

    sym_basis = []
    processed = set()
    for orb in ket_orbits:

        k_step = L // len(orb) if L % len(orb) == 0 else 1

        for k in range(0, L, k_step):

            s = []
            s_coeffs = []

            for l in range(len(orb)):
                #build state from orbits

                s.append(orb[l])
                s_coeffs.append( clean_num_error(np.exp(1j * 2 * np.pi * k/L * l)) )

            if k==0 or k==L/2:
                #handle parity

                inverted_s = [invert(vec) for vec in s]
                is_conj = set(map(tuple, inverted_s)) == set(map(tuple, s))

                if frozenset(map(tuple, s)) in processed:
                    #this state has already been processed
                    continue

                if is_conj:
                    #inverted state stays the same

                    s_coeffs = [1/np.sqrt(len(s_coeffs)) * c for c in s_coeffs]

                    if k == 0:
                        par = 1
                    else:
                        # k == L//2
                        p_state = invert(s[0])
                        p_pos = next(t for t, st in enumerate(s) if st == p_state)
                        phase = np.exp(2j * np.pi * k * p_pos / L)
                        par = int(np.round(phase.real))

                    sym_state = SymState(s, s_coeffs, k, p=par)
                    sym_basis.append(sym_state)

                else:
                    #inverted state is a different translation eigenstate
                    #build proper parity eigenstate

                    processed.add(frozenset(map(tuple, inverted_s)))
 
                    for s_p in (1,-1):
                        s_coeffs_new = s_coeffs + [s_p * c for c in s_coeffs]
                        s_coeffs_new = [1/np.sqrt(len(s_coeffs_new)) * c for c in s_coeffs_new]
                        sym_state = SymState(s + inverted_s, s_coeffs_new, k, p=s_p)
                        sym_basis.append(sym_state)

            else:    
                s_coeffs = [1/np.sqrt(len(s_coeffs)) * c for c in s_coeffs]  
                sym_state = SymState(s, s_coeffs, k, p=None)
                sym_basis.append(sym_state)

    
    for s in sym_basis:
        print(s)
    

    return sym_basis


###################################### SYMMETRY OPERATORS ######################################

def translate(state):
    """Translates a state

    Args:
        state (int tuple or list): input state

    Returns:
        tuple: translated state (int tuple)
    """

    state = tuple(state)
    return tuple(state[-1:] + state[:-1])

def translation_operator(basis):
    """Builds a translation operator

    Args:
        basis (list of int tuples): basis states

    Returns:
        Qobj: translation operator
    """

    dim = len(basis)
    T = lil_matrix((dim, dim), dtype=complex)

    # assign index i to every basis state
    state_index = {tuple(s): i for i, s in enumerate(basis)}

    # for indexes i, states s
    for i, s in enumerate(basis):
        j = state_index[translate(s)] # translate original state and find corresponding new index
        T[j, i] = 1.0 # <j|T|i> = 1 <=> T|i> = |j>

    return qt.Qobj(T)

def invert(state):
    """Reflects a state"""
    state = tuple(state)
    return tuple(state[::-1])

def parity_operator(basis):
    """Builds parity operator P: site i -> L-1-i."""

    dim = len(basis)
    # assign index i to every basis state
    state_index = {tuple(s): i for i, s in enumerate(basis)}

    P = lil_matrix((dim, dim), dtype=complex)
    for i, s in enumerate(basis):
        j = state_index[invert(s)]
        P[j, i] = 1.0

    return qt.Qobj(P)

def N_super(basis):
    """Builds super-particle number operator N = n⊗I - I⊗n^T in Liouville space.

    Eigenvalue of N on |Na><Nb| is Na - Nb.

    Args:
        basis (list): basis states

    Returns:
        Qobj: super-particle number superoperator
    """
    n_per_state = np.array([sum(s) for s in basis])
    dim = len(basis)

    # number operator as diagonal matrix
    n_op = qt.Qobj(np.diag(n_per_state.astype(complex)))

    # N = n⊗I - I⊗n^T
    I = qt.qeye(dim)
    N_super = qt.sprepost(n_op, I) - qt.sprepost(I, n_op.trans())

    return N_super