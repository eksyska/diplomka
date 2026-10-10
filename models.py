import numpy as np
import scipy.sparse as sp

from math_funcs import *
from basis_models import *

###################################### PARAMETER DIAGNOSTICS ######################################

def uniform_fixed_points(g, det_tilde, kappa, f):
    """Uniform classical fixed points n = |z|^2
        g^2 n^3 - 2 g det_tilde n^2 + (det_tilde^2 + kappa^2/4) n - f^2 = 0

    Returns:
        np.ndarray: real roots, sorted (1 root = monostable, 3 roots = bistable)
    """

    roots = np.roots([g**2, -2 * g * det_tilde, det_tilde**2 + kappa**2 / 4, -f**2])
    return np.sort(roots[np.abs(roots.imag) < 1e-9].real)


def cutoff_report(L, N, J, U, driving, gamma, n_cut=None):
    """Checks whether the Fock cutoff can hold the state the parameters ask for

    The drive is scaled as f*sqrt(N) and g = U*N is held fixed, so the classical density
    n = |z|^2 corresponds to a mean site occupation <n_i> = N * n. The basis, however, is
    cut at sum_i n_i <= n_cut. If L * N * n exceeds n_cut, the computed spectrum is not the faithful spectrum of the model.

    Args:
        L, N, J, U, driving: as in BoseHubbard
        gamma (tuple of floats): dissipation RATES
        n_cut (int, optional): total-boson cutoff of the basis. Defaults to N.
    """

    f = driving[0]
    det = driving[1]

    if n_cut is None:
        n_cut = N

    b = 2 if L > 2 else L - 1  # Number of bonds of a periodic chain

    g = 2.0 * U                     
    det_tilde = det + b * J + U / N
    kappa = gamma[0] - (gamma[1] if len(gamma) > 1 else 0.0)  # Warning! This kappa is not the same as the kappa in the symmetry sector labels, which is a quasimomentum index.

    n_cl = uniform_fixed_points(g, det_tilde, kappa, f)
    n_site = N * n_cl.max()
    n_total = L * n_site
    n_pump = (gamma[1] / kappa) if (len(gamma) > 1 and kappa > 0) else 0.0

    bistable = (g * det_tilde > 0) and (abs(det_tilde) > np.sqrt(3) / 2 * kappa)
    ok = (n_total < 0.5 * n_cut) and (kappa > 0)

    print(f"[params] g = {g:.4g}, Delta_tilde = {det_tilde:.4g}, kappa = {kappa:.4g}, f = {f:.4g}")
    branches = "3 branches, f is inside the hysteresis window" if len(n_cl) == 3 else "single branch"
    region = "the region admits bistability" if bistable else "monostable region (|Delta_tilde| <= sqrt(3)/2 kappa or g*Delta_tilde < 0)"
    print(f"[params] uniform classical fixed points n = {np.array2string(n_cl, precision=4)}"
            f"  ({branches}; {region})")
    print(f"[cutoff] <n_i> = N*n = {n_site:.3f}  ->  sum_i <n_i> = {n_total:.3f}   vs   cutoff sum_i n_i <= {n_cut}")
    print(f"[cutoff] incoherent pump background gamma_p/kappa = {n_pump:.3f} per site "
            f"({L * n_pump:.3f} in total)")
    if kappa <= 0:
        print("[cutoff] !! kappa <= 0: gain exceeds loss, there is no normalisable steady state")
    elif not ok:
        print(f"[cutoff] !! the state does not fit in the basis: raise n_cut well above "
                f"{n_total:.0f}, or lower f / raise kappa. The spectrum is a cutoff artefact.")
    else:
        print("[cutoff] ok: the classical state fits comfortably inside the basis")

    return


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
