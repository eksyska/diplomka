import numpy as np
import matplotlib.pyplot as plt
from models import *
from qm_statistics import *
from plot import *
from models import *
from outputs import *
from math_funcs import *
from math_funcs import _ALL

################################ MODELS ################################

def bose_hubbard_lindbladian_deprecated(L=3, N=2, J=1.0, U=1.0, gamma=0.1, restrict_symmetry=False):

    basis_list = build_bose_basis(L, N, fixed_N=False)
    idx_map = {cfg: i for i, cfg in enumerate(basis_list)}

    # local Hilbert space dimension (cutoff N)
    d = N + 1
    dim = d ** L

    # bosonic operators for one site
    a  = qt.destroy(d)
    n  = a.dag() * a

    # Build tensor operators for each site
    a_list, n_list = [], []
    for j in range(L):
        op_list = [qt.qeye(d) for _ in range(L)]
        op_list[j] = a
        a_list.append(qt.tensor(op_list)) 

        op_list[j] = n
        n_list.append(qt.tensor(op_list)) 

    # Bose-Hubbard Hamiltonian
    H = 0

    # hopping (closed boundary)
    for i in range(L):
        j = (i + 1) % L
        H += -J * (a_list[i].dag() * a_list[j] + a_list[j].dag() * a_list[i])

    # onsite interaction
    for j in range(L):
        H += U / N * n_list[j] * (n_list[j] - qt.qeye(H.dims[0]))
    
    # collapse operators (loss at each site)
    #c_ops = [np.sqrt(gamma) * a_j for a_j in a_list]

    c_ops = [np.sqrt(gamma) * a_list[0], 2*np.sqrt(gamma) * a_list[1]]

    # Build Liouvillian
    L_op = qt.liouvillian(H, c_ops)

    if not restrict_symmetry:
        return L_op
        
#wrong splitting into blocks
def evals_by_sector (L_op, L, N, n_local_max=None):

    basis_list = build_bose_basis(L, N, fixed_N=False, n_local_max=n_local_max)
    dim = len(basis_list)
    n_per_state = np.array([sum(s) for s in basis_list])
    sector_evals = {}
    
    for Na in range(N + 1):
        for Nb in range(N + 1):
            # indices in Liouville space corresponding to this (Na, Nb) block
            # rho is indexed as rho[a*dim + b], so block (Na,Nb) = rows/cols where
            # a has Na particles and b has Nb particles
            idx_a = np.where(n_per_state == Na)[0]
            idx_b = np.where(n_per_state == Nb)[0]
            
            # Liouville space indices for this block
            liou_idx = np.array([a * dim + b for a in idx_a for b in idx_b])
            
            if len(liou_idx) == 0:
                continue
            
            block = L_op[np.ix_(liou_idx, liou_idx)]
            evals = np.linalg.eigvals(block)
            sector_evals[(Na, Nb)] = evals
    
    return sector_evals

################################ BLOCK SPLITTING ################################


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