import numpy as np
import scipy.sparse as sp

from math_funcs import *
from basis_models import *


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

        # L_j ⊗ L_j*
        L += sp.kron(L_j, L_j.conjugate(), format='csr')

        # -0.5 [ (L^dag L)⊗I + I⊗(L^dag L)^T ]
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


def get_bilinear(i, j, fock_basis, state_to_idx=None):
    """Builds a_i^dag a_j in fixed-N Fock basis.
    
    Args:
        i (int): site index (creation operator)
        j (int): site index (annihilation operator)
        fock_basis (list of int tuples): Fock basis states
        state_to_idx (dict): int -> tuple of ints (dictionary of Fock basis states)

    Returns:
        sp.csr_matrix: matrix bilinear term    
    """

    if state_to_idx is None:
        state_to_idx = {s: k for k, s in enumerate(fock_basis)}

    dim = len(fock_basis)
    row, col, data = [], [], []

    for idx, s in enumerate(fock_basis):

        if s[j] == 0:
            continue
        if i == j:
            row.append(idx); col.append(idx); data.append(float(s[i]))
            continue

        t = list(s)
        t[j] -= 1
        t[i] += 1
        t = tuple(t)

        if t in state_to_idx:  # fails only if the per-site cutoff truncates the basis
            row.append(state_to_idx[t]); col.append(idx)
            data.append(np.sqrt(s[j]) * np.sqrt(s[i] + 1))

    return sp.csr_matrix((data, (row, col)), shape=(dim, dim), dtype=complex)


def ss_to_density_matrix(sym_state_l, fock_basis):
    """Expands symmetrical Liovillian state to density matrix in Fock basis.
    
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
