import itertools
import numpy as np
from math import comb
from scipy.sparse import dok_matrix, csc_matrix, identity, kron, diags
from scipy.sparse.linalg import eigs
from scipy.special import factorial

# ---------- Basis (fixed total N, L sites) ----------

def build_basis(N, L, full_fock=True):
    """Constructs a basis for Bose-Hubbard system

    Args:
        N (int): # of bosons
        L (int): # of sites
        full_fock (bool): True if particle number is nj <= N, False if it is only nj = N

    Returns:
        numpy.ndarray: 2d int array - basis vectors stored in 2d matrix (rows are vectors, column indices label sites)
    """

    basis = np.empty((0, 3), dtype=int)

    for nj in reversed(range(0, N+1)):

        #not full fock -> only the basis for N particles is generated
        if not full_fock and nj < N:
            break

        dim_j = int(factorial(nj + L - 1) / (factorial(nj) * factorial(L-1) ))

        states = np.zeros((dim_j, L), dtype=int)

        #init
        states[0, 0] = nj
        pivot = 0  #pivot index

        for i in range(1, states.shape[0]):

            states[i, :L-1] = states[i-1, :L-1]
            states[i, pivot] -= 1
            states[i, pivot+1] += 1 + states[i-1, L-1]

            if pivot >= L-2:
                if np.any(states[i, :L-1]):
                    pivot = np.nonzero(states[i, :L-1])[0][-1]
            else:
                pivot += 1

        basis = np.vstack([basis, states])

    return basis


#---------------------------------------------------
# Operator actions
#---------------------------------------------------
def apply_annihilation(state, site):
    """Apply b_site on basis state (tuple/array of occupations).

    Args:
        state (1d array (len L)): 
        site (int): 

    Returns:
        tuple:
            numpy.ndarray: 1d int array - contains new state
            float: normalization coefficient
    """
    if state[site] == 0:
        return None, 0.0
    
    new_state = state.copy()
    new_state[site] -= 1
    coeff = np.sqrt(state[site]) #normalization coefficicent

    return new_state, coeff

def apply_creation(state, site):
    """Apply b_site dagger on basis state.

    Args:
        state (1d array (len L)): 
        site (int):

    Returns:
        tuple:
            numpy.ndarray: 1d int array - contains new state
            float: normalization coefficient
    """
    new_state = state.copy()
    new_state[site] += 1
    coeff = np.sqrt(state[site] + 1)

    return new_state, coeff

#---------------------------------------------------
# Hamiltonian builder
#---------------------------------------------------
def build_hamiltonian(basis, N, L, t=1.0, U=1.0, mu=0.0):

    dim = len(basis)
    H = dok_matrix((dim, dim), dtype=np.complex128)
    
    # Map states to indices
    state_index = {tuple(state): i for i, state in enumerate(basis)}
    
    for i, state in enumerate(basis):
        # On-site interaction + chemical potential
        for j in range(L):
            n = state[j]
            H[i, i] += 0.5 * U * n * (n-1) - mu * n
        
        # Hopping terms
        for j in range(L-1):
            # b_j^\dagger b_{j+1}
            if state[j+1] > 0:
                new_state, coeff1 = apply_annihilation(state, j+1)
                new_state, coeff2 = apply_creation(new_state, j)
                idx = state_index[tuple(new_state)]
                H[idx, i] += -t * coeff1 * coeff2
            
            # b_{j+1}^\dagger b_j
            if state[j] > 0:
                new_state, coeff1 = apply_annihilation(state, j)
                new_state, coeff2 = apply_creation(new_state, j+1)
                idx = state_index[tuple(new_state)]
                H[idx, i] += -t * coeff1 * coeff2
    
    return H.tocsc()

#---------------------------------------------------
# Lindbladian builder
#---------------------------------------------------
def build_lindbladian(basis, N, L, H, gamma=0.1):

    dim = len(basis)
    H = H.tocsc()
    
    # Identity
    I = csc_matrix(np.eye(dim))
    
    # Vectorized Liouvillian: L = -i (H ⊗ I - I ⊗ H^T) + sum_j γ (L_j ⊗ L_j* - 0.5(L_j†L_j ⊗ I + I ⊗ (L_j†L_j)^T))
    # We'll only implement simple local loss jumps: L_j = b_j
    
    # Build annihilation operators
    state_index = {tuple(state): i for i, state in enumerate(basis)}
    L_ops = []
    
    for j in range(L):
        L_j = dok_matrix((dim, dim), dtype=np.complex128)
        for i, state in enumerate(basis):
            if state[j] > 0:
                new_state, coeff = apply_annihilation(state, j)
                idx = state_index[tuple(new_state)]
                L_j[idx, i] = coeff
        L_ops.append(L_j.tocsc())
    
    # Hamiltonian part
    Lmat = -1j * (np.kron(H.toarray(), np.eye(dim)) - np.kron(np.eye(dim), H.toarray().T))
    
    # Dissipators
    for L_j in L_ops:
        Lj = L_j.toarray()
        Lj_dagLj = Lj.conj().T @ Lj
        Lmat += gamma * (np.kron(Lj, Lj.conj()) -
                         0.5 * (np.kron(Lj_dagLj, np.eye(dim)) +
                                np.kron(np.eye(dim), Lj_dagLj.T)))
    
    return csc_matrix(Lmat)

#---------------------------------------------------
# Example usage
#---------------------------------------------------
if __name__ == "__main__":
    N, L = 3, 3   # 3 bosons, 3 sites
    basis = build_basis(N, L)
    print("Basis:\n", basis)   # dim x L matrix
    
    H = build_hamiltonian(basis, N, L, t=1.0, U=1.0, mu=0.0)
    print("Hamiltonian dim:", H.shape)
    
    Lmat = build_lindbladian(basis, N, L, H, gamma=0.1)
    print("Lindbladian dim:", Lmat.shape)
    
    # Diagonalize a few eigenvalues of Lindbladian
    vals, vecs = eigs(Lmat, k=4, which='SR')
    print("Lindbladian eigenvalues:\n", vals)