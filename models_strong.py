import numpy as np
import qutip as qt
import itertools

from scipy.sparse import lil_matrix, dok_matrix
from collections import defaultdict

from math_funcs import *
from math_funcs import _ALL


def get_block_indices(basis, all_n, k_L=_ALL, k_R=_ALL, p_L=_ALL, p_R=_ALL, M=_ALL):
    """Get Liouville space indices for a block defined by sector filters

    Used as a helper function instead of typing the conditions repeatedly

    Args:
        basis (list): SymState or tuple basis
        all_n (list): particle numbers of the basis states
        k_L, k_R (int): ket/bra translation sector. Default to _ALL (for all sectors)
        p_L, p_R (int): ket/bra parity sector. Default to _ALL (for all sectors)
        M (int): particle number sector. Default to _ALL (for all sectors)

    Returns:
        np.ndarray: Liouville space indices
    """

    dim = len(basis)
    is_sym = isinstance(basis[0], SymState)

    masked_arr = np.array([
        i + j * dim
        for i, si in enumerate(basis)
        for j, sj in enumerate(basis)
        if (not is_sym or k_L is _ALL or si.k == k_L)
        and (not is_sym or k_R is _ALL or sj.k == k_R)
        and (not is_sym or p_L is _ALL or si.p == p_L)
        and (not is_sym or p_R is _ALL or sj.p == p_R)
        and (M is _ALL or all_n[i] - all_n[j] == M)
    ])
    return masked_arr

def bose_hubbard_L_blocks(L, N, J, U, gamma, dissipation_type, c_ops_template, n_local_max=None, is_symmetric=False,
                                    k_L_list=None, k_R_list=None, p_L_list=None, p_R_list=None, M_list=None):
    """Build Lindbladian directly as a dict of blocks

    Args:
        L (int): number of sites
        N (int): number of excitations
        J (float): jump coefficient
        U (float): energy coefficient
        gamma (tuple of floats): dissipation rates
        dissipation_type (string): DEPHASING / LOSS / PUMPLOSS
        c_ops_template (tuple of floats): per-site dissipation weights
        n_local_max (int): local Hilbert space cutoff. Defaults to None
        is_symmetric (bool): True if symmetric dissipation on all sites. Defaults to False
        k_L_list (list of ints): ket translation sectors to compute. Defaults to None (for all sectors)
        k_R_list (list of ints): bra translation sectors to compute. Defaults to None (for all sectors)
        p_L_list (list of ints): ket parity sectors to compute. Defaults to None (for all sectors)
        p_R_list (list of ints): bra parity sectors to compute. Defaults to None (for all sectors)
        M_list (list of ints): particle number sectors to compute. Defaults to None (for all sectors)

    Returns:
        dict: label -> np.ndarray block matrix
    """

    print(f"[*] building lindbladian blocks L={L}, N={N}, J={J}, U={U}, gamma={gamma}")

    basis = build_bose_basis(L, N, fixed_N=False, n_local_max=n_local_max)
    if is_symmetric:
        basis = build_sym_basis(basis)

    dim = len(basis)
    all_n = [s.n if isinstance(s, SymState) else sum(s) for s in basis]

    #build H and c_ops once in the (sym) basis
    a_list = [build_a_i_sym(i, basis) for i in range(L)] if is_symmetric else [build_a_i(i, basis) for i in range(L)]
    H, c_ops = build_H_and_cops(a_list, L, N, J, U, gamma, dissipation_type, c_ops_template, dim)
    H_op = H.full()
    c_ops = [c.full() for c in c_ops]

    #determine sector combinations to compute
    if is_symmetric:
        ks  = k_L_list or list(range(L))
        krs = k_R_list or list(range(L))
        pls = p_L_list or [1, -1, None]
        prs = p_R_list or [1, -1, None]
    else:
        ks = krs = pls = prs = [_ALL]

    ms = M_list or list(range(-N, N + 1))

    blocks = {}
    for k_l, k_r, p_l, p_r, M in itertools.product(ks, krs, pls, prs, ms):

        indices = get_block_indices(basis, all_n, k_L=k_l, k_R=k_r, p_L=p_l, p_R=p_r, M=M)
        if len(indices) == 0:
            continue

        label = (k_l, k_r, p_l, p_r, M) if is_symmetric else M
        print(f"[*] block {label}: size {len(indices)}")

        block = build_L_block_direct(H_op, c_ops, indices)
        blocks[label] = block
    
    return blocks


def csr_from_evals(block_evals, csr_func, k_L=_ALL, k_R=_ALL, p_L=_ALL, p_R=_ALL, M=_ALL):
    """Compute complex spacing ratios separately per block and pool results

    Usually the block eigenvalues are already filtered so there is no need to filter again

    Args:
        block_evals (dict): label -> eigenvalues
        csr_func (callable): function that takes eigenvalues and returns complex spacing ratios
        k_L (int or list): ket translation filter. Defaults to _ALL (for all sectors)
        k_R (int or list): bra translation filter. Defaults to _ALL (for all sectors)
        p_L (int or list): ket parity filter. Defaults to _ALL (for all sectors)
        p_R (int or list): bra parity filter. Defaults to _ALL (for all sectors)
        M (int or list): particle number filter. Defaults to _ALL (for all sectors)

    Returns:
        np.ndarray: pooled complex spacing ratios
    """

    all_ratios = []

    for label, evals in block_evals.items():
        if isinstance(label, tuple) and len(label) == 5:
            kl, kr, pl, pr, m = label
        else:
            kl, kr, pl, pr, m = _ALL, _ALL, _ALL, _ALL, label

        if (k_L is _ALL or kl == k_L) \
        and (k_R is _ALL or kr == k_R) \
        and (p_L is _ALL or pl == p_L) \
        and (p_R is _ALL or pr == p_R) \
        and (M is _ALL or m == M):

            if len(evals) < 3:
                continue

            result = csr_func(evals)
            ratios = (result[0] if isinstance(result, tuple) else result).ravel()
            all_ratios.append(ratios)

    if not all_ratios:
        raise ValueError(f"no blocks matched k_L={k_L}, k_R={k_R}, p_L={p_L}, p_R={p_R}, M={M}")

    pooled_ratios = np.concatenate(all_ratios)
    return pooled_ratios

def pool_evals(block_evals, k_L=_ALL, k_R=_ALL, p_L=_ALL, p_R=_ALL, M=_ALL):
    """Concatenate eigenvalues from blocks matching the given sector filters

    Usually the block eigenvalues are already filtered so there is no need to filter again

    Args:
        block_evals (dict): label -> eigenvalues
        k_L (int or list): ket translation filter. Defaults to _ALL (for all sectors)
        k_R (int or list): bra translation filter. Defaults to _ALL (for all sectors)
        p_L (int or list): ket parity filter. Defaults to _ALL (for all sectors)
        p_R (int or list): bra parity filter. Defaults to _ALL (for all sectors)
        M (int or list): particle number filter. Defaults to _ALL (for all sectors)

    Returns:
        np.ndarray: concatenated eigenvalues
    """

    evals = []
    for label, v in block_evals.items():
        if isinstance(label, tuple) and len(label) == 5:
            kl, kr, pl, pr, m = label
        else:
            kl, kr, pl, pr, m = _ALL, _ALL, _ALL, _ALL, label

        if (k_L is _ALL or kl == k_L) \
        and (k_R is _ALL or kr == k_R) \
        and (p_L is _ALL or pl == p_L) \
        and (p_R is _ALL or pr == p_R) \
        and (M is _ALL or m == M):
            evals.append(v)

    if not evals:
        raise ValueError(f"no blocks matched k_L={k_L}, k_R={k_R}, p_L={p_L}, p_R={p_R}, M={M}")
    
    pooled_evals = np.concatenate(evals)
    return pooled_evals

def liouvillian_blocks(lind) -> dict:

    L_op = lind.L_op
    L = lind.L
    sym_basis = lind.basis

    dim_H = len(sym_basis)
    all_n = [s.n for s in sym_basis]

    block_indices = defaultdict(list)

    for i, si in enumerate(sym_basis):

        for j, sj in enumerate(sym_basis):

            label = (si.k, sj.k, si.p, sj.p, all_n[i] - all_n[j])
            block_indices[label].append(i + j * dim_H)

    L_dense = L_op.full()

    blocks = {label: (indices, L_dense[np.ix_(indices, indices)]) for label, indices in block_indices.items()}

    return blocks





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



def build_H_and_cops(a_list, L, N, J, U, gamma, dissipation, c_ops_template, dim):
    """Builds Hamiltonian and dissipation operators

    Args:
        a_list (list of Qobjs): annihilation operators
        L (int): number of sites
        N (int): number of excitations
        J (float): jump coefficient
        U (float): energy coefficient
        gamma (tuple of floats): dissipation rates
        dissipation_type (string): DEPHASING / LOSS / PUMPLOSS
        c_ops_template (tuple of floats): per-site dissipation weights
        dim (int): Hamiltonian dimenstion

    Returns:
        tuple of Qobjs: Hamiltonian and list of dissipation operators
    """
    
    H = qt.Qobj(np.zeros((dim, dim), dtype=complex))

    for i in range(L):
        j = (i + 1) % L
        H += -J * (a_list[i].dag() * a_list[j] + a_list[j].dag() * a_list[i])

    for i in range(L):
        n_i = a_list[i].dag() * a_list[i]
        H += U / N * n_i * (n_i - 1)

    c_ops = []
    for i in range(L):
        if dissipation == 'DEPHASING':
            c_ops.append(c_ops_template[i] * np.sqrt(gamma[0]) * a_list[i].dag() * a_list[i])
        elif dissipation == 'LOSS':
            c_ops.append(c_ops_template[i] * np.sqrt(gamma[0]) * a_list[i])
        elif dissipation == 'PUMPLOSS':
            c_ops.append(c_ops_template[i] * np.sqrt(gamma[0]) * a_list[i])
            c_ops.append(c_ops_template[i] * np.sqrt(gamma[1]) * a_list[i].dag())

    return H, c_ops

def build_L_block_direct(H_op, c_ops, block_indices):
    """Builds a Liouvillian block directly only in the specified symmetry sector

    Args:
        H_op (np.ndarray): Hamiltonian
        c_ops (list): dissipation operators
        block_indices (np.ndarray): Liouville indices in the block

    Returns:
        np.ndarray: block submatrix
    """

    dim = H_op.shape[0]

    #ket(i) and bra(j) indices for each Liouville index
    i_idx = block_indices % dim  
    j_idx = block_indices // dim  

    delta_j = (j_idx[:,None] == j_idx[None,:]) #delta_j[p,q] = 1 if bra indices match: j_p == j_q
    delta_i = (i_idx[:,None] == i_idx[None,:]) #delta_i[p,q] = 1 if ket indices match: i_p == i_q

    #first Lindbladian term (commutator)
    # -i H rho: <i|-iH|k><l|j> = -i*H[i,k] * delta_{l,j}
    # +i rho H: <i|k><l|+iH|j> = +i*delta_{i,k} * H[l,j]
    block = -1j * (H_op[np.ix_(i_idx, i_idx)] * delta_j - H_op[np.ix_(j_idx, j_idx)].T * delta_i)

    #second Lindbladian term (dissipation)
    for c in c_ops:
        cd = c.conj().T
        cdc = cd @ c

        block += c[np.ix_(i_idx, i_idx)] * c.conj()[np.ix_(j_idx, j_idx)]
        block -= 0.5 * cdc[np.ix_(i_idx, i_idx)] * delta_j
        block -= 0.5 * cdc[np.ix_(j_idx, j_idx)].T * delta_i

    return block

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
        Qobj: annihilation operator in the symmetric basis
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


def bose_hubbard_L_full(L, N, J, U, gamma, dissipation_type, c_ops_template, n_local_max=None, is_symmetric=False):
    """Builds full Bose-Hubbard model Lindbladian utilizing QuTip's liouvillian function
    
    Args:
        L (int): number of sites
        N (int): number of excitations
        J (float): jump coefficient
        U (float): energy coefficient
        gamma (tuple of floats): dissipation rates
        dissipation_type (string): DEPHASING / LOSS / PUMPLOSS
        c_ops_template (tuple of floats): per-site dissipation weights
        n_local_max (int): local Hilbert space cutoff. Defaults to None
        is_symmetric (bool): True if symmetric dissipation on all sites. Defaults to False

    Returns:
        Lindbladian: Lindbladian
    """
    
    print(f"[*] building lindbladian for L={L}, N={N}, J={J}, U={U}, gamma={gamma}, symmetric={is_symmetric}")

    basis = build_bose_basis(L, N, fixed_N=False, n_local_max=n_local_max)
    if is_symmetric:
        #rebuild basis list into translation and parity basis states
        basis = build_sym_basis(basis)

    dim = len(basis)
    
    a_list = [build_a_i_sym(i, basis) for i in range(L)] if is_symmetric else [build_a_i(i, basis) for i in range(L)] 
    H, c_ops = build_H_and_cops(a_list, L, N, J, U, gamma, dissipation_type, c_ops_template, dim)
    
    L_op = qt.liouvillian(H, c_ops)

    lind = Lindbladian(L_op, L, N, J, U, gamma, dissipation_type, c_ops_template, basis, n_local_max, is_symmetric)

    return lind


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
        self.dim = L_op.shape[0]

    def center_shift(self):
        """Shifts Lindbladian such that its eigenvalues are now centered around 0 (zero trace)

        Returns:
            Lindbladian: shifted Lindbladian
        """
        
        shift = self.L_op.tr() / self.dim
        data = self.L_op.full() - shift * np.eye(self.dim)
        self.L_op = qt.Qobj(data, dims=self.L_op.dims, superrep=self.L_op.superrep)
        return self