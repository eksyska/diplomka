import numpy as np
import qutip as qt
import itertools

from scipy.sparse import lil_matrix

from math_funcs import *

from math_funcs import _ALL


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

    def __eq__(self, other):

        if not isinstance(other, SymState):
            return NotImplemented

        is_eq = ( (set(self.fock_states) == set(other.fock_states)) and (self.k == other.k) )

        return is_eq
    
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

    def __eq__(self, other):

        if not isinstance(other, StateL):
            return NotImplemented

        return (
            self.ket == other.ket and self.bra == other.bra
        )

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

    # make all combinations of translation invariant ket and bra states
    statesL = [StateL(ket, bra) for ket in t_basis for bra in t_basis]

    sym_statesL = []
    processed = []

    for s_l in statesL:

        if any(s_l == s for s in processed):
            continue

        if s_l.kappa != 0:

            sym_statesL.append(SymStateL((s_l,), (1,), s_l.kappa, pi=None))
            processed.append(s_l)
            continue

        # kappa == 0 -> handle parity

        # compute inverted ket and bra states
        ket_partner, ket_phase = parity_partner(s_l.ket, t_basis, L)
        bra_partner, bra_phase = parity_partner(s_l.bra, t_basis, L)
        inv = StateL(ket_partner, bra_partner)
        total_phase = ket_phase * np.conj(bra_phase)

        if inv == s_l:
            # inverted state is the same
            p = 1 if total_phase.real > 0 else -1
            sym_statesL.append(SymStateL((s_l,), (1,), 0, pi=p))
            processed.append(s_l)
        
        else:
            # inverted state is a different state in the translation invariant basis -> compute linear combination
            for p in (-1, 1):
                coeffs = [1/np.sqrt(2), p * total_phase / np.sqrt(2)]
                sym_statesL.append(SymStateL([s_l, inv], coeffs, 0, pi=p))
            processed.append(s_l)
            processed.append(inv)

    return sym_statesL


def parity_partner(sym_state, t_basis, L):
    """Find the parity partner in translation invariant basis

    Args: 
        sym_state (SymState): translation invariant state to invert
        t_basis (list of SymStates): translation invariant basis states on Fock space
        L (int): number of sites

    Returns:
        partner (Symstate): parity partner to the original state
        phase (float): phase changed by inverting
    """

    def find_in_t_basis(fock_set, k, t_basis):
        for ts in t_basis:
            if ts.k == k and set(ts.fock_states) == fock_set:
                return ts
        return None

    inv_states = [invert(s) for s in sym_state.fock_states]

    # we look for the partner in this set
    target_set = set(inv_states)

    target_k = (-sym_state.k) % L
    partner = find_in_t_basis(target_set, target_k, t_basis)

    naive_map = dict(zip(inv_states, sym_state.coeffs))
    partner_map = dict(zip(partner.fock_states, partner.coeffs))
    ref = partner.fock_states[0]
    phase = naive_map[ref] / partner_map[ref]

    return partner, phase


def build_sym_basis(fock_basis):
    """Builds Fock basis from T, P operators eigenstates

    Args:
        fock_basis (list of int tuples): oringinal Fock basis

    Returns:
        list of SymStates: new basis
    """

    L = len(fock_basis[0]) #number of sites
    ket_orbits = build_ket_orbits(fock_basis)

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

                    par = 1
                    """
                    if k == 0:
                        
                    else:
                        # k == L//2
                        p_state = invert(s[0])
                        p_pos = next(t for t, st in enumerate(s) if st == p_state)
                        phase = np.exp(2j * np.pi * k * p_pos / L)
                        par = int(np.round(phase.real))
                    """
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