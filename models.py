import numpy as np
import scipy.sparse as sp

from math_funcs import *
from basis_models import *



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
            jump_ops = []
            for a_i in a_i_list:
                jump_ops.append(a_i * np.sqrt(self.gamma[0]))
                jump_ops.append(a_i.conjugate().transpose() * np.sqrt(self.gamma[1]))

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

    # row-major: vec_r(AXB) = (A ⊗ B^T) vec_r(X)
    # -i(HX - XH): A=H,B=I for HX -> H⊗I ; A=I,B=H for XH -> I⊗H^T
    L = -1j * (sp.kron(Hc, I, format='csr') - sp.kron(I, Hc.transpose(), format='csr'))

    for L_j in jump_ops:

        L_j = L_j.tocsr()
        L_j_dag = L_j.conjugate().transpose()
        L_j_dag_L_j = (L_j_dag @ L_j).tocsr()

        # L X L^dag: A=L_j, B=L_j^dag -> L_j ⊗ (L_j^dag)^T = L_j ⊗ conj(L_j)
        L += sp.kron(L_j, L_j.conjugate(), format='csr')

        # -0.5(L^dag L X + X L^dag L): A=L^dag L,B=I -> (L^dag L)⊗I ; A=I,B=L^dag L -> I⊗(L^dag L)^T
        L -= 0.5 * (sp.kron(L_j_dag_L_j, I, format='csr') + sp.kron(I, L_j_dag_L_j.transpose(), format='csr'))

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