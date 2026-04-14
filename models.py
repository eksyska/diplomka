import numpy as np
import qutip as qt
import itertools

from scipy.sparse import lil_matrix, dok_matrix
from collections import defaultdict

from math_funcs import *



def bose_hubbard_lindbladian(L, N, J, U, gamma, dissipation_type, c_ops_template, n_local_max=None, restrict_symmetry=False):
    """Builds Bose-Hubbard model Lindbladian, can be reduced to translation symmetry subspace

    Args:
        L (int): number of sites
        N (int): number of excitations
        J (float): 2-site interaction strength
        U (float): on-site interaction strength
        gamma (array float): dissipation strength
        c_ops_template (array float): list of dissipation coefficients on sites
        n_local_max (int): site cuttoff. Defaults to None.
        restrict_symmetry (bool, optional): TRUE if we want to restrict the Lindbladian to a translation subspace. Defaults to False.
        k (int, optional): momentum sector (if we restrict to a subspace). Defaults to 0.

    Returns:
        Qobj: Bose-Hubbard Lindbladian
    """

    print(f"[*] building lindbladian for L={L}, N={N}, J={J}, U={U}, gamma={gamma}")

    basis_list = bose_basis(L, N, fixed_N=False, n_local_max=n_local_max)
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

        if dissipation_type == 'DEPHASING':
            c_ops.append(c_ops_template[i]*np.sqrt(gamma[0]) * a_list[i].dag() * a_list[i])

        elif dissipation_type == 'LOSS':
            c_ops.append(c_ops_template[i]*np.sqrt(gamma[0]) * a_list[i])

        elif dissipation_type == 'PUMPLOSS':
            c_ops.append(c_ops_template[i]*np.sqrt(gamma[0]) * a_list[i])
            c_ops.append(c_ops_template[i]*np.sqrt(gamma[1]) * a_list[i].dag())

    L_op = qt.liouvillian(H, c_ops)


    #check commutator L_op, N_op (should be 0)
    """
    I_super = qt.qeye(L_op.shape[0])
    I_super.dims = L_op.dims
    L_op_shifted = L_op - (L_op.tr() / I_super.tr()) * I_super

    N_super_op = N_super(basis_list)

    comm_LN = comm(L_op_shifted.full(), N_super_op.full())
    print(f"||[L, N]|| = {np.linalg.norm(comm_LN):.2e}")
    """

    if not restrict_symmetry:
        return L_op
    
    T_op = translation_operator(basis_list)
    T_L = qt.sprepost(T_op, T_op.dag())
    t_evals, t_evecs = T_L.eigenstates()
    tol = 1e-8

    P_op = parity_operator(basis_list)
    P_L_full = qt.sprepost(P_op, P_op.dag()).full()
    parity_k = {0} | ({L // 2} if L % 2 == 0 else set())

    L_op_full = L_op.full()

    L_op_sectors = []
    for k in range(L):
        eval_sub = np.exp(1j * 2 * np.pi * k / L)
        t_indices = [i for i, ev in enumerate(t_evals) if abs(ev - eval_sub) < tol]
        P_k = np.column_stack([t_evecs[i].full().ravel() for i in t_indices])

        if k in parity_k:
            P_L_k = np.conj(P_k).T @ P_L_full @ P_k
            p_evals, p_evecs = np.linalg.eigh(P_L_k)
            for p in [+1, -1]:
                p_indices = [i for i, ev in enumerate(p_evals) if abs(ev - p) < tol]
                if not p_indices:
                    continue
                P_kp = P_k @ p_evecs[:, p_indices]
                L_op_sectors.append(qt.Qobj(np.conj(P_kp).T @ L_op_full @ P_kp))
        else:
            L_op_sectors.append(qt.Qobj(np.conj(P_k).T @ L_op_full @ P_k))

    return L_op_sectors

def lindblad_eigenvalues_by_M(L_op, L, N, n_local_max, M_chosen=None):
    """Compute Lindbladian eigenvalues separately for each M = N_a - N_b sector.
    M is the super-particle number: eigenvalue of N = n⊗I - I⊗n^T.
    This is an exact weak symmetry of the Lindbladian: [L, N] = 0.
    
    Args:
        L_op (Qobj): Bose-Hubbard lindbladian
        L (int): number of sites
        N (int): number of excitations
        n_local_max (int): site cuttoff
        M_chosen (int): sector to compute evals in. Defaults to None (all sectors computed)

    Returns:
        dict: {M: eigenvalues} for M = -N, ..., N
    """

    basis_list = bose_basis(L, N, fixed_N=False, n_local_max=n_local_max)
    n_per_state = np.array([sum(s) for s in basis_list])
    dim = len(basis_list)

    L_mat = L_op.full()

    sector_evals = {}

    if M_chosen == None:
        M_range = range(-N, N + 1)
    else:
        M_range = [M_chosen]

    for M in M_range:

        print(f"[*] computing evals for M={M}")

        # indices a*dim+b to store rho_ab where N_a-N_b=M (defines the block)
        liou_idx = np.array([a * dim + b for a, Na in enumerate(n_per_state) for b, Nb in enumerate(n_per_state) if Na - Nb == M])

        #blocks <2 cannot produce spacing ratios
        if len(liou_idx) < 2:
            continue

        #extract the block M and diagonalize it independently
        block = L_mat[np.ix_(liou_idx, liou_idx)]
        sector_evals[M] = np.linalg.eigvals(block)

    return sector_evals

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
    basis_list = bose_basis(L, N, fixed_N=True, n_local_max=None)

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


def bose_basis(L, N, fixed_N=True, n_local_max=None):
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

def reflect(state):
    """Reflects a state"""
    state = tuple(state)
    return tuple(reversed(state))

def parity_operator(basis):
    """Builds parity operator P: site i -> L-1-i."""

    dim = len(basis)
    # assign index i to every basis state
    state_index = {tuple(s): i for i, s in enumerate(basis)}

    P = lil_matrix((dim, dim), dtype=complex)
    for i, s in enumerate(basis):
        j = state_index[reflect(s)]
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

def comm (A, B):
    """Computes commutator

    Args:
        A (Qobj or 2darray): matrix A
        B (Qobj or 2darray): matrix B

    Returns:
        Qobj or 2darray: commutator of A and B
    """
    return A @ B - B @ A


