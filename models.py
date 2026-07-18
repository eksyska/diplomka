import numpy as np
import qutip as qt
import scipy.sparse as sp

from collections import defaultdict

from math_funcs import *
from basis_models import *
from math_funcs import _ALL



class BoseHubbard:
    """Bose-Hubbard model with set parameter values
    """

    def __init__(self, L, N, J, U, dissipation, gamma, n_local_max, M_list=[], kappa_list=[], pi_list=[] ):
        self.L = L
        self.N = N
        self.J = J
        self.U = U
        self.dissipation = dissipation
        self.gamma = gamma
        self.n_local_max = n_local_max
        self.M_list = M_list
        self.kappa_list = kappa_list
        self.pi_list = pi_list


    def build_H(self, fock_basis):
        """Builds Bose-Hubbard Hamiltonian in Fock basis

        Args:
            fock_basis (list of int tuples): Fock basis states

        Returns:
            np.2darray: Hamiltonian matrix in Fock basis
        """

        L = self.L
        J = self.J
        U = self.U
        N = self.N

        dim = len(fock_basis)
        
        # initialize empty Hamiltonian matrix
        H = sp.csr_matrix((dim, dim), dtype=complex)
        
        # two-site interaction
        for i in range(L):
            j = (i + 1) % L  # periodic boundary
            a_i = get_a_i(i, fock_basis)
            a_j = get_a_i(j, fock_basis)
            
            # -J * (a_i^dagger a_j + a_j^dagger a_i)
            H += -J * (a_i.conjugate().transpose() @ a_j + a_j.conjugate().transpose() @ a_i)
            
        # on-site interaction U / N * n*(n-1))
        diag_vals = [U / N * sum(n * (n - 1) for n in state) for state in fock_basis]
        H = H + sp.diags(diag_vals, dtype=complex, format='csr')
            
        return H
    

    def build_jump_ops(self, fock_basis):
        """Builds jump operators in Fock basis

        Args:
            fock_basis (list of int tuples): Fock basis states

        Returns:
            list of np.2darrays: list of jump operators on all sites
        """

        if self.dissipation == "LOSS":
            jump_ops = [get_a_i(i, fock_basis) * np.sqrt(self.gamma[0]) for i in range(self.L)]

        elif self.dissipation == "PUMPLOSS":
            a_i_list = [get_a_i(i, fock_basis) for i in range(self.L)]
            jump_ops = [a_i * np.sqrt(self.gamma[0]) + a_i.conjugate().transpose() * np.sqrt(self.gamma[1]) for a_i in a_i_list]

        return jump_ops

    
    def build_L_blocks(self, fock_basis, sym_basis_L):
        """
        Builds Liouvillian blocks in symmetry subspaces labeled M, kappa, pi.

        Args:
            fock_basis (list of int tuples): Fock basis states
            sym_basis_L (list of SymStateL): Liouvillian translation and parity symmetric basis states
            H_fock (np.2darray): Hamiltonian in Fock basis
            jump_ops_fock (np.2darray): jump operators in Fock basis
        """

        M_list = self.M_list if self.M_list else np.arange(-self.N, self.N+1, 1)
        kappa_list = self.kappa_list if self.kappa_list else np.arange(0, self.L+1, 1)
        pi_list = self.pi_list if self.pi_list else None
        M_set, kappa_set = set(M_list), set(kappa_list)
        pi_set = set(pi_list) if pi_list is not None else None

        sectors = {}

        # select wanted sectors
        for ss_L in sym_basis_L:

            if ss_L.M not in M_set or ss_L.kappa not in kappa_set:
                continue
            if pi_set is not None and ss_L.pi not in pi_set:
                continue
            sectors.setdefault((ss_L.M, ss_L.kappa, ss_L.pi), []).append(ss_L)

        dim = len(fock_basis)
        fock_to_idx = {state: i for i, state in enumerate(fock_basis)}

        H_fock = self.build_H(fock_basis)
        jump_ops_fock = self.build_jump_ops(fock_basis)
        L_full = build_full_liouvillian(H_fock, jump_ops_fock) # built once

        blocks = {}

        # build selected blocks
        for sector_key, sector_states in sectors.items():

            M, kappa, pi = sector_key
            size = len(sector_states)
            print(f"Building sector [M={M}, kappa={kappa}, pi={pi}], #states: {size}")

            RHO = sp.hstack([ss.to_sparse(fock_to_idx, dim) for ss in sector_states], format='csc')
            LRHO = L_full @ RHO
            blocks[sector_key] = (RHO.conjugate().transpose() @ LRHO).toarray()

        return blocks


###################################### LINDBLADIAN BUILDING ######################################    


def build_full_liouvillian(H, jump_ops):
    """Builds sparse dim^2 Liouvillian superoperator (column-major density matrix convention).

    Args:
        H (np.2darray): Hamiltonian matrix
        jump_ops (list of np.2darrays): jump operators matrices

    Returns:
        scipy.sparse.csr_matrix: Liouvillian matrix
    """

    dim = H.shape[0]
    I = sp.eye(dim, dtype=complex, format='csr')
    Hc = H.tocsr()

    L = -1j * (sp.kron(I, Hc, format='csr') - sp.kron(Hc.transpose(), I, format='csr'))

    for L_j in jump_ops:

        L_j = L_j.tocsr()
        L_j_dag = L_j.conjugate().transpose()
        L_j_dag_L_j = (L_j_dag @ L_j).tocsr()
        L += sp.kron(L_j.conjugate(), L_j, format='csr')
        L -= 0.5 * (sp.kron(I, L_j_dag_L_j, format='csr') + sp.kron(L_j_dag_L_j.transpose(), I, format='csr'))

    return L.tocsr()


def get_a_i(site, fock_basis):
    """Builds annihilation operator in Fock basis for given site

    Args:
        site (int): Bose-Hubbard site
        fock_basis (list of int tuples): Fock basis states

    Returns:
        np.2darray: annihilation operator matrix in Fock basis
    """

    dim = len(fock_basis)
    state_to_idx = {state: i for i, state in enumerate(fock_basis)}
    
    # sparse matrix initialization
    row, col, data = [], [], []
    
    for idx, state in enumerate(fock_basis):

        n = state[site] # initial number of excitations on the site

        if n > 0:
            # lower number of excitations on given site
            new_state = list(state)
            new_state[site] -= 1
            new_state = tuple(new_state)
            
            if new_state in state_to_idx:
                # find Fock basis element that describes the new state
                target_idx = state_to_idx[new_state]
                row.append(target_idx)
                col.append(idx)
                data.append(np.sqrt(n))
                
    return sp.csr_matrix((data, (row, col)), shape=(dim, dim), dtype=complex)
    

def ss_to_density_matrix(sym_state_l, fock_basis):
    """
    Expands symmetrical Liovillian state to density matrix in Fock basis.
    
    Args:
        sym_state_l (SymStateL): translation and parity symmetrical Liouvillian state
        fock_basis (list of int tuples): Fock basis states

    Returns:
        np.2darray: density matrix in Fock basis
    """

    dim = len(fock_basis)
    fock_to_idx = {state: i for i, state in enumerate(fock_basis)}
    
    # empty density matrix
    rho = np.zeros((dim, dim), dtype=complex)
    
    # iterate through individual Liouvillian states inside the symmetrical Liouvillian state
    for state_l, coeff_l in zip(sym_state_l.states, sym_state_l.coeffs):
        
        # build ket vector in Fock basis
        ket_vec = np.zeros(dim, dtype=complex)
        for f_state, coeff_k in zip(state_l.ket.fock_states, state_l.ket.coeffs):
            ket_vec[fock_to_idx[f_state]] += coeff_k
            
        # build bra vector in Fock basis
        bra_vec = np.zeros(dim, dtype=complex)
        for f_state, coeff_b in zip(state_l.bra.fock_states, state_l.bra.coeffs):
            bra_vec[fock_to_idx[f_state]] += np.conj(coeff_b)
            
        # outer product |ket><bra| (bra_vec is already conjugated)
        rho += coeff_l * np.outer(ket_vec, bra_vec)
        
    # normalization
    norm = np.sqrt(np.sum(np.conj(rho) * rho))
    if norm > 1e-12:
        rho /= norm
        
    return rho


###################################### OLD CODE (MAINLY FULL LINDBLADIAN FOR COMPARSIONS) ######################################   


class Lindbladian():
    """Lindbladian object"""

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
    
def print_L_basis(basis):
    """Prints Liouville space basis states (ket-bra)

    Args:
        basis (list of floats or SymStates): Hilbert space basis
    """

    for alpha in range(len(basis)**2):
        i = alpha % len(basis)
        j = alpha // len(basis)
        print(f"  alpha={alpha}: |{basis[i]}><{basis[j]}|")


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

    data = sp.dok_matrix((dim, dim), dtype=complex)
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

    data = sp.dok_matrix((dim, dim), dtype=complex)

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



def liouvillian_blocks(lind):
    L_op = lind.L_op
    L = lind.L
    sym_basis = lind.basis
    dim_H = len(sym_basis)
    all_n = [s.n for s in sym_basis]

    block_indices = defaultdict(list)
    for i, si in enumerate(sym_basis):
        for j, sj in enumerate(sym_basis):
            kappa = (si.k - sj.k) % L
            pi = True if (si.p == sj.p) else None
            M = all_n[i] - all_n[j]
            block_indices[(kappa, pi, M)].append(i + j * dim_H)

    L_dense = L_op.full()
    return {
        label: (indices, L_dense[np.ix_(indices, indices)])
        for label, indices in block_indices.items()
    }

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