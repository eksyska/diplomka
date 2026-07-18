import numpy as np
import scipy.sparse as sp
import qutip as qt
import itertools

from math_funcs import *



###################################### STATE CLASSES ######################################

class SymState:
    """Eigenstate of translation (and optionally parity)
    """

    def __init__(self, fock_states, coeffs, k, p=None):
        self.fock_states = tuple(fock_states)
        self.coeffs = np.asarray(coeffs, dtype=complex)
        self.k = k
        self.p = p
        self.n = sum(self.fock_states[0])
    
    def __repr__(self):
        string = ""
        for f_s in self.fock_states:
            string = f"{string} {str(f_s)}"
        return f"SymState(k={self.k}, p={self.p}): {string}"


class StateL:
    """Represents a state on Liouville space
    """

    def __init__(self, ket, bra):
        """
        Args:
            self.ket (SymState)
            self.bra (SymState)
        """
        self.ket = ket
        self.bra = bra
    
    @property
    def L(self):
        return len(self.ket.fock_states[0])

    @property
    def kappa(self):
        return (self.ket.k - self.bra.k) % self.L
    
    @property
    def M(self):
        return sum(self.ket.fock_states[0]) - sum(self.bra.fock_states[0])

    def __repr__(self):
        return f"\nStateL(kappa={self.kappa}):\n ket: {self.ket} \n bra: {self.bra})"


class SymStateL:

    def __init__(self, states, coeffs, kappa, pi=None):
        self.states = tuple(states)
        self.coeffs = np.asarray(coeffs, dtype=complex)
        self.kappa = kappa
        self.pi = pi
        self.M = states[0].M

    def __str__(self):
        return (
            f"\n\nSYM STATE L (kappa={self.kappa}, pi={self.pi}):\n{self.states}"
        )
    
    __repr__ = __str__

    def to_sparse(self, fock_to_idx, dim):
        """Vectorizes a SymStateL directly as a sparse dim^2 column

        Args:
            fock_to_idx (dict): maps a Fock basis state (tuple) to its index in fock_basis
            dim (int): dimension of the Fock basis

        Returns:
            scipy.sparse.csc_matrix: vectorized SymStateL (sparse dim^2 column)
        """

        rows, data = [], []
        for state_l, coeff_l in zip(self.states, self.coeffs):

            ket, bra = state_l.ket, state_l.bra

            for f_k, c_k in zip(ket.fock_states, ket.coeffs):

                i = fock_to_idx[f_k]
                ket_val = coeff_l * c_k

                for f_b, c_b in zip(bra.fock_states, bra.coeffs):

                    j = fock_to_idx[f_b]
                    rows.append(i + j * dim)
                    data.append(ket_val * np.conj(c_b))

        v = sp.coo_matrix((data, (rows, [0]*len(rows))), shape=(dim*dim, 1), dtype=complex).tocsc()

        # normalize vector
        norm = np.sqrt(np.sum(np.abs(v.data)**2))
        if norm > 1e-12:
            v = v / norm
            
        return v


###################################### BASIS BUILDERS ######################################   

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


def build_ket_orbits(fock_basis):
    """Find all translation orbits of single Fock states.
 
    Args:
        fock_basis (list of tuples): Fock basis states
 
    Returns:
        ket_orbits (list of lists): each entry is the orbit [s, T(s), T²(s), ...]
        ket_orbit_of (dict): state -> (orbit_index, position_within_orbit)
    """
    ket_orbits = []
    visited = set()

    for s in fock_basis:

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

        # set ordering of the orbit (by rotating the list, largest state is first)
        max_idx = max(range(len(orb)), key=lambda i: orb[i])
        orb = orb[max_idx:] + orb[:max_idx]
        ket_orbits.append(orb)

    return ket_orbits


def build_translation_basis(fock_basis):
    """Builds translation invariant basis in Fock space

    Args:
        fock_basis (list of tuples): Fock basis states

    Returns:
        list of SymStates: translation invariant basis states in Fock space
    """

    L = len(fock_basis[0])
    orbits = build_ket_orbits(fock_basis)

    basis = []

    for orbit in orbits:

        period = len(orbit)

        for m in range(period):

            k = m * L // period

            coeffs = np.exp(2j * np.pi * k * np.arange(period) / L) / np.sqrt(period)
            basis.append(SymState(orbit, coeffs, k))

    return basis


def build_sym_L_basis(fock_basis):
    """Builds translation and parity invariant basis in Lioville space

    Args:
        fock_basis (list of tuples): Fock basis states

    Returns:
        list of SymStateL: translation and parity invariant basis states in Lioville space
    """

    L = len(fock_basis[0])
    t_basis = build_translation_basis(fock_basis)

    t_lookup = {_key(ts): ts for ts in t_basis}

    def parity_partner(sym_state):
        """Find the parity partner in translation invariant basis

        Args: 
            sym_state (SymState): translation invariant state to invert
            t_basis (list of SymStates): translation invariant basis states on Fock space
            L (int): number of sites

        Returns:
            partner (Symstate): parity partner to the original state
            phase (float): phase changed by inverting
        """

        inv_states = [invert(s) for s in sym_state.fock_states]
        target_key = (frozenset(inv_states), (-sym_state.k) % L)
        partner = t_lookup[target_key]

        naive_map = dict(zip(inv_states, sym_state.coeffs))
        partner_map = dict(zip(partner.fock_states, partner.coeffs))
        ref = partner.fock_states[0]
        phase = naive_map[ref] / partner_map[ref]

        return partner, phase

    sym_statesL = []
    seen = set() 

    for ket in t_basis:

        ket_key = _key(ket)

        for bra in t_basis:

            bra_key = _key(bra)
            pair_key = (ket_key, bra_key)
            if pair_key in seen:
                continue

            kappa = (ket.k - bra.k) % L
            s_l = StateL(ket, bra)

            if kappa != 0:
                sym_statesL.append(SymStateL((s_l,), (1,), kappa, pi=None))
                seen.add(pair_key)
                continue

            # kappa == 0

            # find inverted states among translation symmetric states
            ket_partner, ket_phase = parity_partner(ket)
            bra_partner, bra_phase = parity_partner(bra)
            inv_key = (_key(ket_partner), _key(bra_partner))
            total_phase = ket_phase * np.conj(bra_phase)

            if inv_key == pair_key:
                # parity maps the state on itself

                p = 1 if total_phase.real > 0 else -1
                sym_statesL.append(SymStateL((s_l,), (1,), 0, pi=p))
                seen.add(pair_key)

            else:
                # parity maps the state on a different translation state

                inv = StateL(ket_partner, bra_partner)
                for p in (-1, 1):
                    coeffs = [1/np.sqrt(2), p * total_phase / np.sqrt(2)]
                    sym_statesL.append(SymStateL([s_l, inv], coeffs, 0, pi=p))
                seen.add(pair_key)
                seen.add(inv_key)

    return sym_statesL


def _key(ss):
    """Hashable key to compare Symstates
    
    Args:
        ss (SymState)
    
    Returns:
        tuple (fock_states, k)
    """
    return (frozenset(ss.fock_states), ss.k)


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


def invert(state):
    """Reflects a state"""
    state = tuple(state)
    return tuple(state[::-1])

