import numpy as np
import scipy.sparse as sp

from math_funcs import *
from basis_models import *
from models import *


class BoseHubbard:
    """Bose-Hubbard model with set parameter values
    """

    def __init__(self, L, N, J, U, config, driving, gamma, n_local_max,
                N_pairs=None, M_list=None, kappa_list=None, pi_list=None, n_cut=None): 
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
        """Builds the Fock basis. Uses self.n_cut as the total-boson cutoff and self.n_local_max as the per-site cutoff.
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

            if gamma[4] != 0.0: #directed circulation minus
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

        if config.N_strong:
            return self._build_L_blocks_fixed_N(fock_basis, sym_basis_L)

        def given(x):
            return x is not None and len(x) > 0

        kappa_list = self.kappa_list if given(self.kappa_list) else np.arange(self.L)
        kappa_set = set(kappa_list)

        if config.has_parity:
            pi_set = set(self.pi_list) if given(self.pi_list) else None
        else:
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


    def build_basis(self, fixed_N=False):
        """Chooses basis building approach: all N basis/fixed N basis
        
        Args:
            fixed_N (bool): Defaults to False
        """

        if self.config.N_strong:
            return self.build_basis_by_N()
        
        return build_bose_basis(self.L, self.N, self.n_cut, fixed_N=fixed_N, n_local_max=self.n_local_max)


    def _needed_N(self):
        """Proccesses N pairs needed in case of strong symmetry

        Returns:
            list of ints: required N pairs to compute
        """

        # N_pairs passed as parameter -> sort them
        if self.N_pairs is not None and len(self.N_pairs) > 0:
            return sorted({int(n) for pair in self.N_pairs for n in pair})

        # else all N pairs
        return list(range(self.n_cut + 1))

    def build_basis_by_N(self):
        """Builds multiple bases for all n needed for the model

        Returns:
            dict: n -> basis
        """

        ns = self._needed_N()

        if self.n_local_max is not None and self.n_local_max < max(ns):
            print(f"[basis] warning: n_local_max={self.n_local_max} < max N={max(ns)}, "
                  f"fixed-N bases are truncated (same truncation as the full-basis path)")
            
        bases = {}
        for n in ns:

            b = build_fixed_N_basis(self.L, n, self.n_local_max)
            if b:
                bases[n] = b

        return bases


    def build_sym_basis(self, fock_basis):
        """Liouville symmetry basis matching the chosen mode.
        
        Args:
            fock_basis (list of int tuples): Fock basis states
        
        Returns:
            Liouville symmetry basis matching the chosen mode
        """

        if self.config.N_strong:
            return build_sym_L_basis_fixed_N(fock_basis, self.N_pairs)
        
        return build_sym_L_basis(fock_basis, use_parity=self.config.has_parity)


    def build_H_fixed_N(self, basis, state_to_idx=None):
        """Builds H on one fixed-N basis (built from bilinears a_i^dag a_j)
        
        Args:
            basis (list of int tuples): Fock basis for fixed N
            state_to_idx (dict): int -> tuple of ints (dictionary of Fock basis states). Defaults to None
                    
        Returns:
            np.csr_matrix: matrix Hamiltonian representation"""

        L, J, U, N = self.L, self.J, self.U, self.N

        if state_to_idx is None:
            state_to_idx = {s: k for k, s in enumerate(basis)}

        dim = len(basis)

        H = sp.csr_matrix((dim, dim), dtype=complex)

        n_bonds = L if L > 2 else L - 1
        for i in range(n_bonds):
            j = (i + 1) % L
            H += -J * (get_bilinear(i, j, basis, state_to_idx) + get_bilinear(j, i, basis, state_to_idx))

        diag_vals = [U / N * sum(n * (n - 1) for n in s) for s in basis]

        return H + sp.diags(diag_vals, dtype=complex, format='csr')
    

    def build_jump_ops_fixed_N(self, basis, state_to_idx=None):
        """Builds N-conserving jump operators for one fixed-N basis
        
        Args:
            basis (list of int tuples): Fock basis for fixed N
            state_to_idx (dict): int -> tuple of ints (dictionary of Fock basis states)
            
        Returns:
            list of np.2darrays: list of jump operators on all sites
        """
        
        L, N = self.L, self.N
        if state_to_idx is None:
            state_to_idx = {s: k for k, s in enumerate(basis)}

        gamma = list(self.gamma)
        for idx in self.config.zero_gamma_idx:
            gamma[idx] = 0

        B = lambda a, b: get_bilinear(a, b, basis, state_to_idx)  # a_a^dag a_b

        jump_ops = []
        for i in range(L):

            j = (i + 1) % L

            if gamma[2] != 0:  # dephasing
                jump_ops.append(B(i, i) * np.sqrt(gamma[2]))

            if gamma[3] != 0:  # directed circulation plus
                jump_ops.append(B(j, i) * np.sqrt(gamma[3] / N))

            if gamma[4] != 0:  # directed circulation minus
                jump_ops.append(B(i, j) * np.sqrt(gamma[4] / N))

            if gamma[5] != 0:  # bond phase locking
                jump_ops.append((B(i, i) - B(i, j) + B(j, i) - B(j, j)) * np.sqrt(gamma[5] / N))

        return jump_ops


    def _build_L_blocks_fixed_N(self, bases_by_N, sym_basis_L):
        """Builds sector Liouvillians for (N_i, N_j, kappa)"""

        config = self.config

        def given(x):
            return x is not None and len(x) > 0

        kappa_set = set(self.kappa_list if given(self.kappa_list) else np.arange(self.L))
        N_pair_set = {(int(a), int(b)) for a, b in self.N_pairs} if given(self.N_pairs) else None

        # select sectors
        sectors = {}
        for ss_L in sym_basis_L:
            if ss_L.kappa not in kappa_set:
                continue
            if N_pair_set is not None and (ss_L.N_i, ss_L.N_j) not in N_pair_set:
                continue
            sectors.setdefault((ss_L.N_i, ss_L.N_j, ss_L.kappa), []).append(ss_L)

        # operators and index maps, once per N
        cache = {}

        def ops_for(n):

            if n not in cache:
                basis = bases_by_N[n]
                idx = {s: k for k, s in enumerate(basis)}
                cache[n] = (self.build_H_fixed_N(basis, idx),
                            self.build_jump_ops_fixed_N(basis, idx), idx, len(basis))
                
            return cache[n]

        # group by (N_i, N_j): the sector Liouvillian does not depend on kappa
        by_pair = {}
        for key in sectors:
            by_pair.setdefault(key[:2], []).append(key)

        blocks = {}
        for (n_i, n_j), keys in by_pair.items():

            H_i, C_i, idx_i, d_i = ops_for(n_i)
            H_j, C_j, idx_j, d_j = ops_for(n_j)
            I_i = sp.identity(d_i, dtype=complex, format='csr')
            I_j = sp.identity(d_j, dtype=complex, format='csr')

            L_sec = -1j * (sp.kron(H_i, I_j) - sp.kron(I_i, H_j.transpose()))
            for c_i, c_j in zip(C_i, C_j):
                cdc_i = (c_i.conjugate().transpose() @ c_i).tocsr()
                cdc_j = (c_j.conjugate().transpose() @ c_j).tocsr()
                L_sec = L_sec + sp.kron(c_i, c_j.conjugate())
                L_sec = L_sec - 0.5 * (sp.kron(cdc_i, I_j) + sp.kron(I_i, cdc_j.transpose()))
            L_sec = L_sec.tocsr()

            for key in keys:
                states = sectors[key]
                print(f"Building sector {key}, #states: {len(states)}")
                RHO = sp.hstack([ss.to_sparse(idx_i, d_i, idx_j, d_j) for ss in states], format='csc')
                blocks[key] = (RHO.conjugate().transpose() @ (L_sec @ RHO)).toarray()

        return blocks