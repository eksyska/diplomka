import numpy as np
import math
import itertools
import qutip as qt

import qm_statistics as stat
import plot as plot

from scipy.sparse import lil_matrix
from scipy.sparse import dok_matrix

from miscelaneous import *



def bose_hubbard_lindbladian(L, N, J, U, gamma, c_ops_template, restrict_symmetry=False, k=1):
    """Builds Bose-Hubbard model Lindbladian, can be reduced to translation symmetry subspace

    Args:
        L (int): # of sites
        N (int): # of excitations
        J (float): 2-site interaction strength
        U (float): on-site interaction strength
        gamma (float): dissipation strength
        c_ops_template (array): list of dissipation coefficients on sites
        restrict_symmetry (bool, optional): TRUE if we want to restrict the Lindbladian to a translation subspace. Defaults to False.
        k (int, optional): momentum sector (if we restrict to a subspace). Defaults to 0.

    Returns:
        Qobj: Bose-Hubbard Lindbladian
    """

    basis_list = bose_basis(L, N, fixed_N=False)
    dim = len(basis_list)    

    # build all annihilation operators
    a_list = [build_a_i(i, basis_list) for i in range(L)]

    # construct Hamiltonian
    H = qt.Qobj(np.zeros((dim, dim), dtype=complex))

    # hopping term
    for i in range(L):
        j = (i + 1) % L
        H += -J * (a_list[i].dag() * a_list[j] + a_list[j].dag() * a_list[i])

    # on-site interaction
    for i in range(L):
        n_i = a_list[i].dag() * a_list[i]
        H += U/N * n_i * (n_i - 1)

    # jump operators for dissipation
    c_ops = []
    for i in range(L):
        c_ops.append(c_ops_template[i]*np.sqrt(gamma) * a_list[i])
    
    L_op = qt.liouvillian(H, c_ops)


    I_super = qt.qeye(L_op.shape[0])
    I_super.dims = L_op.dims
    L_op_shifted = L_op - (L_op.tr() / I_super.tr()) * I_super
    
    if not restrict_symmetry:
        return L_op
    
    T_op = translation_operator(basis_list)
    T_L = qt.sprepost(T_op, T_op.dag())
    evals, evecs = T_L.eigenstates()

    eval_sub = np.exp(1j * 2*np.pi * k / L) #select eigenvalue eccording to momentum sector k (has the form e^(i*2pi*k/L))
    tol = 1e-8

    indices = [i for i, ev in enumerate(evals) if abs(ev - eval_sub) < tol]

    P = qt.Qobj(np.column_stack([evecs[i].full().ravel() for i in indices]))
    L_red = qt.Qobj(P.dag() @ L_op.full() @ P)

    return L_red

def build_a_i(site, basis_list):
    """Builds annihilation operator on site

    Args:
        site (int): site index
        basis_list (int tuple list): list of basis states (int tuples)

    Returns:
        Qobj: annihilation operator
    """

    dim = len(basis_list)

    # map: tuple(state) -> index in basis_list
    state_index = {tuple(state): i for i, state in enumerate(basis_list)}

    data = dok_matrix((dim, dim), dtype=complex)
    for col, state in enumerate(basis_list):

        if state[site] > 0:
            new_state = list(state)
            new_state[site] -= 1
            row = state_index[tuple(new_state)]
            data[row, col] = np.sqrt(state[site])

    return qt.Qobj(data.tocsr())


def bose_hubbard_hamiltonian(L, N, J, U, restrict_symmetry=False, k=1):
    """Builds Bose-Hubbard model Hamiltonian, can be reduced to translation symmetry subspace

    Args:
        L (int): number of sites
        N (int): number of excitations
        J (float): 2-site interaction strength
        U (float): on-site interaction strength
        restrict_symmetry (bool, optional): TRUE if we want to restrict the Hamiltonian to a translation subspace. Defaults to False.
        k (int, optional): momentum sector (if we restrict to a subspace). Defaults to 0.

    Returns:
        Qobj: Bose-Hubbard Hamiltonian
    """

    # filter to total particle number = N
    basis_list = bose_basis(L, N, fixed_N=True)

    # basis: list of tuples with fixed N
    state_index = {tuple(s): i for i,s in enumerate(basis_list)}
    dim = len(basis_list)

    H = lil_matrix((dim, dim), dtype=complex)

    # hopping term
    for col, state in enumerate(basis_list):
        for i in range(L):
            j = (i + 1) % L

            if state[j] > 0:
                new_s = list(state)
                new_s[i] += 1
                new_s[j] -= 1
                new_s = tuple(new_s)

                row = state_index[new_s]
                amp = -J * np.sqrt((state[i] + 1) * state[j])
                H[row, col] += amp
                H[col, row] += amp

    # interaction term
    for i, state in enumerate(basis_list):
        diag = sum(n*(n-1) for n in state)
        H[i, i] += U/N * diag

    H = qt.Qobj(H)

    if not restrict_symmetry:
        return H
    
    T_op = translation_operator(basis_list)
    evals, evecs = T_op.eigenstates()

    eval_sub = np.exp(1j * 2*np.pi * k / L) #select eigenvalue according to momentum sector k (has the form e^(i*2pi*k/L))
    tol = 1e-8

    indices = [i for i, ev in enumerate(evals) if abs(ev - eval_sub) < tol]

    P = qt.Qobj(np.column_stack([evecs[i].full().ravel() for i in indices]))
    H_red = qt.Qobj(P.dag() @ H.full() @ P)

    return H_red


def bose_basis(L, N, fixed_N=True):
    """Builds Bose basis in Fock space

    Args:
        L (int): number of sites
        N (int): total number of excitaions
        fixed_N (bool, optional): TRUE if all basis states have conserved # of excitations. Defaults to True.

    Returns:
        list: basis states (int tuples)
    """

    n_local_max = N
    all_configs = itertools.product(range(n_local_max+1), repeat=L)

    if fixed_N:
        basis = [cfg for cfg in all_configs if sum(cfg) == N]

    else:
        basis = [cfg for cfg in all_configs if sum(cfg) <= N]

    return basis


def translate(state):
    """Translates a state

    Args:
        state (int tuple or list): input state

    Returns:
        tuple: translated state (int tuple)
    """

    state = tuple(state)
    return (state[-1],) + state[:-1]


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


def comm (A, B):
    """Computes commutator

    Args:
        A (Qobj or 2darray): matrix A
        B (Qobj or 2darray): matrix B

    Returns:
        Qobj or 2darray: commutator of A and B
    """
    return A @ B - B @ A


################################ DEPRECATED ################################

def bose_hubbard_lindbladian_deprecated(L=3, N=2, J=1.0, U=1.0, gamma=0.1, restrict_symmetry=False):

    basis_list = bose_basis(L, N, fixed_N=False)
    idx_map = {cfg: i for i, cfg in enumerate(basis_list)}

    # local Hilbert space dimension (cutoff N)
    d = N + 1
    dim = d ** L

    # bosonic operators for one site
    a  = qt.destroy(d)
    n  = a.dag() * a

    # Build tensor operators for each site
    a_list, n_list = [], []
    for j in range(L):
        op_list = [qt.qeye(d) for _ in range(L)]
        op_list[j] = a
        a_list.append(qt.tensor(op_list)) 

        op_list[j] = n
        n_list.append(qt.tensor(op_list)) 

    # Bose-Hubbard Hamiltonian
    H = 0

    # hopping (closed boundary)
    for i in range(L):
        j = (i + 1) % L
        H += -J * (a_list[i].dag() * a_list[j] + a_list[j].dag() * a_list[i])

    # onsite interaction
    for j in range(L):
        H += U / N * n_list[j] * (n_list[j] - qt.qeye(H.dims[0]))
    
    # collapse operators (loss at each site)
    #c_ops = [np.sqrt(gamma) * a_j for a_j in a_list]

    c_ops = [np.sqrt(gamma) * a_list[0], 2*np.sqrt(gamma) * a_list[1]]

    # Build Liouvillian
    L_op = qt.liouvillian(H, c_ops)

    if not restrict_symmetry:
        return L_op
