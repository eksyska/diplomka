import numpy as np
import scipy.sparse as sp

from math_funcs import *
from basis_models import *


class BoseHubbard:
    """Bose-Hubbard model with set parameter values
    """

    def __init__(self, L, N, J, U, config, driving, gamma, n_local_max,
                N_pairs=None, M_list=None, kappa_list=None, pi_list=None, n_cut=None ): 
        """
        Args:
            N (int): the semiclassical parameter, 1/hbar_eff. It sets the scaling of the
                Hamiltonian (U/N, f*sqrt(N)) and nothing else.
            n_cut (int, optional): total-boson cutoff of the Fock basis, sum_i n_i <= n_cut.
                Kept separate from N: the mean occupation the parameters ask for <n_i> = N * |z|^2. See cutoff_report.
            n_local_max (int): per-site cutoff of the Fock basis.
        """
        self.L = L
        self.N = N
        self.n_cut = N if n_cut is None else n_cut
        self.J = J
        self.U = U
        self.driving = driving
        self.config = config
        self.gamma = gamma

        self.n_local_max = n_local_max
        self.N_pairs = N_pairs
        self.M_list = M_list
        self.kappa_list = kappa_list
        self.pi_list = pi_list

    def build_basis(self, fixed_N=False):
        """
        Builds the Fock basis. Uses self.n_cut as the total-boson cutoff and self.n_local_max as the per-site cutoff.
        """

        return build_bose_basis(self.L, self.N, self.n_cut, fixed_N=fixed_N, n_local_max=self.n_local_max)

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
        config = self.config

        dim = len(fock_basis)
        
        # initialize empty Hamiltonian matrix
        H = sp.csr_matrix((dim, dim), dtype=complex)

        # build all a_i once
        a_list = [get_a_i(i, fock_basis) for i in range(L)]

        # hopping: -J * (a_i^dagger a_j + a_j^dagger a_i)
        # L >= 3 has L bonds; L = 2 has a single bond (i -> i+1 would count it twice), L = 1 has none
        n_bonds = L if L > 2 else L - 1

        for i in range(n_bonds):
            j = (i + 1) % L  # periodic boundary
            a_i, a_j = a_list[i], a_list[j]
            H += -J * (a_i.conjugate().transpose() @ a_j + a_j.conjugate().transpose() @ a_i)

        # on-site interaction U / N * n*(n-1))
        diag_vals = [U / N * sum(n * (n - 1) for n in state) for state in fock_basis]

        H = H + sp.diags(diag_vals, dtype=complex, format='csr')

        if config.has_driving:

            f = self.driving[0]
            det = self.driving[1]

            # driving: f * sqrt(N) * (a_i + a_i^dagger)
            for i in range(L):
                a_i = a_list[i]
                H += f * np.sqrt(N) * (a_i + a_i.conjugate().transpose())

            # driving: detuning: - det * a_i^dagger a_
            diag_det = [- det * sum(n for n in state) for state in fock_basis]

            H = H + sp.diags(diag_det, dtype=complex, format='csr')
            
        return H
    

    def build_jump_ops(self, fock_basis):
        """Builds jump operators in Fock basis

        Implements
        sum_i ( g_loss D[a_i] + g_pump D[a_i^dag] + g_deph D[n_i] + g_circ/N D[a_i^dag a_j] + g_bpl/N D[c_ij]) rho.
        The rates are read from self.gamma in this order: gamma = (g_loss, g_pump, g_deph, g_circ, g_bpl)

        Returns:
            list of np.2darrays: list of jump operators on all sites
        """

        gamma = list(self.gamma)
        for idx in self.config.zero_gamma_idx:
            gamma[idx] = 0

        jump_ops = []
        a_i_list = [get_a_i(i, fock_basis) for i in range(self.L)]
        a_i_dag_list = [a_i_list[i].conjugate().transpose() for i in range(self.L)]

        for i in range(self.L):

            if gamma[0] != 0.0: #loss
                jump_ops.append(a_i_list[i] * np.sqrt(gamma[0]))

            if gamma[1] != 0.0: #pumping
                jump_ops.append(a_i_dag_list[i] * np.sqrt(gamma[1]))

            if gamma[2] != 0.0: #dephasing
                jump_ops.append((a_i_dag_list[i] @ a_i_list[i]) * np.sqrt(gamma[2]))

            if gamma[3] != 0.0: #directed circulation plus
                j = (i + 1) % self.L 
                jump_ops.append((a_i_dag_list[j] @ a_i_list[i]) * np.sqrt(gamma[3]/self.N))

            if gamma[4] != 0.0: #directed circulation plus
                j = (i + 1) % self.L 
                jump_ops.append((a_i_dag_list[i] @ a_i_list[j]) * np.sqrt(gamma[4]/self.N))

            if gamma[5] != 0.0: #bond phase locking
                j = (i + 1) % self.L 
                jump_ops.append((a_i_dag_list[i] + a_i_dag_list[j]) @ (a_i_list[i] - a_i_list[j]) * np.sqrt(gamma[5]/self.N))

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

        config = self.config

        def given(x):
            return x is not None and len(x) > 0

        kappa_list = self.kappa_list if given(self.kappa_list) else np.arange(self.L)
        kappa_set = set(kappa_list)

        if config.has_parity:
            pi_set = set(self.pi_list) if given(self.pi_list) else None
        else:
            if given(self.pi_list):
                raise ValueError("pi_list given, but this config has no parity symmetry")
            pi_set = None

        if config.N_weak:
            M_list = self.M_list if given(self.M_list) else np.arange(-self.n_cut, self.n_cut + 1)
            M_set = set(M_list)

        N_pair_set = None
        if config.N_strong and given(self.N_pairs):
            N_pair_set = {(int(a), int(b)) for a, b in self.N_pairs}

        sectors = {}

        # select wanted sectors
        for ss_L in sym_basis_L:

            if ss_L.kappa not in kappa_set:
                continue

            if pi_set is not None and ss_L.pi not in pi_set:
                continue

            if config.N_weak and ss_L.M not in M_set:
                continue

            if N_pair_set is not None and (ss_L.N_i, ss_L.N_j) not in N_pair_set:
                continue

            key = []
            if config.N_weak:
                key.append(ss_L.M)
            elif config.N_strong:
                key.extend((ss_L.N_i, ss_L.N_j))
            key.append(ss_L.kappa)
            if config.has_parity:
                key.append(ss_L.pi)

            sectors.setdefault(tuple(key), []).append(ss_L)

        dim = len(fock_basis)
        fock_to_idx = {state: i for i, state in enumerate(fock_basis)}

        H_fock = self.build_H(fock_basis)
        jump_ops_fock = self.build_jump_ops(fock_basis)
        L_full = build_full_liouvillian(H_fock, jump_ops_fock) # built once

        blocks = {}

        # build selected blocks
        for sector_key, sector_states in sectors.items():

            print(f"Building sector {sector_key}, #states: {len(sector_states)}")
            

            RHO = sp.hstack([ss.to_sparse(fock_to_idx, dim) for ss in sector_states], format='csc')
            LRHO = L_full @ RHO
            blocks[sector_key] = (RHO.conjugate().transpose() @ LRHO).toarray()

        return blocks

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
