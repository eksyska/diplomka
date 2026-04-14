import numpy as np
import matplotlib.pyplot as plt
from models import *
from qm_statistics import *
from plot import *
from models import *
from outputs import *

################################ MODELS ################################

def bose_hubbard_lindbladian_deprecated(L=3, N=2, J=1.0, U=1.0, gamma=0.1, restrict_symmetry=False):

    basis_list = bose_basis(L, N, fixed_N=False)
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
    
def evals_by_sector (L_op, L, N, n_local_max=None):

    basis_list = bose_basis(L, N, fixed_N=False, n_local_max=n_local_max)
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

################################ PLOT ################################

def plot_complex_ratios_by_NaNb(sector_evals, N, is_single_color=True, sectors=None):
    """Plots complex spacing ratio statistics colored by (Na, Nb) sector.
    Computes z on the full spectrum, then colors by sector.

    Args:
        sector_evals (dict): {(Na, Nb): eigenvalues} from lindblad_eigenvalues_by_NaNb
        N (int): max particle number
        real (bool): if True, plot only |z| histogram for real spectra
    """
    cmap = plt.cm.get_cmap('tab20', (N + 1) ** 2)

    # concatenate all eigenvalues and track sector labels
    all_evals = []
    valid_sectors = []

    for idx, ((Na, Nb), evals) in enumerate(sector_evals.items()):
        if len(evals) < 3:
            continue
        all_evals.append(evals)
        valid_sectors.append((idx, Na, Nb, len(evals)))

    all_evals = np.concatenate(all_evals)

    # compute z on the full spectrum
    z, _ = complex_spacing_ratios(all_evals)

    # split z back by sector
    sector_z = {}
    start = 0
    for idx, Na, Nb, n in valid_sectors:
        sector_z[(Na, Nb)] = z[start:start + n]
        start += n

    all_r = []
    for idx, Na, Nb, n in valid_sectors:
        r_sec = np.abs(sector_z[(Na, Nb)])
        all_r.append(r_sec)
        print(f"({Na},{Nb}): ⟨r⟩ = {r_sec.mean():.4f}")
    print(f"total ⟨r⟩ = {np.concatenate(all_r).mean():.4f}  (GinUE: 0.739, Poisson: 0.500)")

    if sectors is not None:
        valid_sectors = [(idx, Na, Nb, n) for idx, Na, Nb, n in valid_sectors if (Na, Nb) in sectors]

    def get_color(idx):
        return 'C0' if is_single_color else cmap(idx)

    sec_colors = [get_color(idx) for idx, Na, Nb, n in valid_sectors]
    sec_labels = [f'({Na},{Nb})' for idx, Na, Nb, n in valid_sectors]

    # legend handles
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cmap(idx), markersize=5, label=f'({Na},{Nb})')
               for idx, Na, Nb, n in valid_sectors]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    # scatter colored by sector
    start = 0
    for idx, Na, Nb, n in valid_sectors:
        z_sec = z[start:start + n]
        ax[0].scatter(z_sec.real, z_sec.imag, s=1, alpha=0.5, color=get_color(idx))
        start += n

    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax[0].add_artist(circle)
    ax[0].set_aspect('equal')
    ax[0].set_title("Complex spacing ratios")
    if not is_single_color:
        ax[0].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncols=2)

    all_abs = [np.abs(sector_z[(Na, Nb)]) for _, Na, Nb, _ in valid_sectors]
    all_arg = [np.angle(sector_z[(Na, Nb)]) for _, Na, Nb, _ in valid_sectors]

    """
    for abs_vals, arg_vals, color, label in zip(all_abs, all_arg, sec_colors, sec_labels):
        ax[1].hist(abs_vals, bins=50, density=False, alpha=0.4, color=color, label=label)
        ax[2].hist(arg_vals, bins=50, density=False, alpha=0.4, color=color, label=label)"""

    ax[1].hist(all_abs, bins=50, density=False, histtype='barstacked', alpha=0.7, color=sec_colors, label=sec_labels)
    ax[1].set_xlabel("|z|")
    ax[1].set_ylabel("count")
    ax[1].set_title("Histogram of |z|")

    ax[2].hist(all_arg, bins=50, density=False, histtype='barstacked', alpha=0.7, color=sec_colors, label=sec_labels)
    ax[2].set_xlabel("arg(z)")
    ax[2].set_ylabel("count")
    ax[2].set_title("Histogram of arg(z)")

    plt.tight_layout()
    plt.savefig("spacings_Na_Nb.png")

def plot_spectrum_by_NaNb(sector_evals, N, is_single_color=True):
    """Plot Lindbladian eigenvalues in complex plane colored by (Na, Nb) sector.

    Args:
        sector_evals (dict): {(Na, Nb): eigenvalues} from lindblad_eigenvalues_by_NaNb
        N (int): max particle number (for colormap range)
    """
    fig, ax = plt.subplots(figsize=(7,5))

    cmap = plt.cm.get_cmap('tab20', (N + 1) ** 2)

    def get_color(idx):
        return 'C0' if is_single_color else cmap(idx)

    for idx, ((Na, Nb), evals) in enumerate(sector_evals.items()):
        ax.scatter(evals.real, evals.imag,
                   s=4, alpha=0.7,
                   color=get_color(idx),
                   label=f'({Na},{Nb})')

    plt.axhline(0, color='black', alpha=0.5, linewidth=0.8)
    plt.axvline(0, color='black', alpha=0.5, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title('Lindbladian spectrum by (Na, Nb) sector')
    if not is_single_color:
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncols=2)

    plt.tight_layout()
    plt.show()
