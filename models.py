import numpy as np
import qutip as qt
import itertools

from scipy.sparse import lil_matrix, dok_matrix
from collections import defaultdict

from math_funcs import *

###################################### LINDBLADIAN BUILDING ######################################

class Lindbladian():
    """Lindbladian object
    """

    def __init__(self, L_op, L, N, J, U, gamma, dissipation_type, c_ops_template, basis, n_local_max=None, is_symmetric=False):

        self.L_op = L_op
        self.L = L
        self.N = N
        self.J = J
        self.U = U
        self.gamma = gamma
        self.dissip = dissipation_type
        self.c_templ = c_ops_template
        self.n_local_max = n_local_max if n_local_max!=None else N
        self.is_symmetric = is_symmetric
        self.basis = basis


def bose_hubbard_lindbladian(L, N, J, U, gamma, dissipation_type, c_ops_template, n_local_max=None, is_symmetric=False):
    """Builds Bose-Hubbard model Lindbladian, can be reduced to translation symmetry subspace"""

    print(f"[*] building lindbladian for L={L}, N={N}, J={J}, U={U}, gamma={gamma}, symmetric={is_symmetric}")

    basis = build_bose_basis(L, N, fixed_N=False, n_local_max=n_local_max)
    dim = len(basis)

    if not is_symmetric:
        a_list = [build_a_i(i, basis) for i in range(L)]

    else:
        #rebuild basis list into translation and parity basis states
        basis = build_sym_basis(basis)
        a_list = [build_a_i_sym(i, basis) for i in range(L)]      

    H = qt.Qobj(np.zeros((dim, dim), dtype=complex))

    for i in range(L):
        j = (i + 1) % L
        H += -J * (a_list[i].dag() * a_list[j] + a_list[j].dag() * a_list[i])

    for i in range(L):
        n_i = a_list[i].dag() * a_list[i]
        H += U/N * n_i * (n_i - 1)

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

    #clean numerical errors
    tol = 1e-10
    data = L_op.full().copy()
    data.real[np.abs(data.real) < tol] = 0.0
    data.imag[np.abs(data.imag) < tol] = 0.0
    L_op = qt.Qobj(data, dims=L_op.dims, superrep=L_op.superrep)

    lind = Lindbladian(L_op, L, N, J, U, gamma, dissipation_type, c_ops_template, basis, n_local_max, is_symmetric)

    return lind



def build_a_i(site, basis):
    """Builds annihilation operator on site

    Args:
        site (int): site index
        basis (list of int tuples): list of basis states

    Returns:
        Qobj: annihilation operator
    """

    dim = len(basis)

    # map: tuple(state) -> index in basis
    state_index = {tuple(state): i for i, state in enumerate(basis)}

    data = dok_matrix((dim, dim), dtype=complex)
    for col, state in enumerate(basis):

        if state[site] > 0:
            new_state = list(state)
            new_state[site] -= 1
            row = state_index[tuple(new_state)]
            data[row, col] = np.sqrt(state[site])

    return qt.Qobj(data.tocsr())

def build_a_i_sym(site, sym_basis):
    """Builds annihilation operator on a site in the symmetric basis

    Args:
        site (int): site index
        sym_basis (list of SymStates): list of symmetric basis states

    Returns:
        Qobj: annihilation operator
    """

    dim = len(sym_basis)
    
    # maps Fock tuple -> list of (basis_index, coefficient)
    # allows us to find which SymStates contain a specific Fock state
    fock_to_sym = {}
    for idx, sym_state in enumerate(sym_basis):

        for fock, coeff in zip(sym_state.states, sym_state.coeffs):

            fock_tuple = tuple(fock)
            if fock_tuple not in fock_to_sym:
                fock_to_sym[fock_tuple] = []
            fock_to_sym[fock_tuple].append((idx, coeff))

    data = dok_matrix((dim, dim), dtype=complex)

    # iterate over every state in the basis (the "initial" state / column)
    for col_idx, state_beta in enumerate(sym_basis):
        # apply a_i to every Fock component of the SymState
        for fock_beta, c_beta in zip(state_beta.states, state_beta.coeffs):

            n_i = fock_beta[site]
            
            # no particles at this site
            if n_i <= 0:
                continue
            
            new_fock = list(fock_beta)
            new_fock[site] -= 1
            new_fock_tuple = tuple(new_fock)
            
            # project the result onto the "final" states (the rows)
            # find all SymStates that contain this new Fock state
            matches = fock_to_sym.get(new_fock_tuple, [])
            
            for row_idx, c_alpha in matches:

                val = np.conj(c_alpha) * c_beta * np.sqrt(n_i)
                data[row_idx, col_idx] += val # += because multiple Fock states in beta might map to the same SymState alpha.

    return qt.Qobj(data.tocsr())


def liouvillian_blocks(lind) -> dict:

    L_op = lind.L_op
    L = lind.L
    sym_basis = lind.basis

    dim_H = len(sym_basis)
    all_n = [s.n for s in sym_basis]

    block_indices = defaultdict(list)

    for i, si in enumerate(sym_basis):

        for j, sj in enumerate(sym_basis):

            kappa = (si.k - sj.k) % L
            pi    = (si.p * sj.p) if (si.p is not None and sj.p is not None) else None
            M     = all_n[i] - all_n[j]
            block_indices[(kappa, pi, M)].append(i + j * dim_H)

    L_dense = L_op.full()

    blocks = {label: (indices, L_dense[np.ix_(indices, indices)]) for label, indices in block_indices.items()}

    return blocks



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

    """
    for s in sym_basis:
        print(s)
    """

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


###################################### HAMILTONIAN ONLY ######################################

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
    basis_list = build_bose_basis(L, N, fixed_N=True, n_local_max=None)

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