import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree, KDTree
from scipy import stats, spatial
import mpmath as mp



########################## COMPLEX SPACING RATIOS ##########################

def complex_spacing_ratios(evals):
    """Compute complex spacing ratios for a list/array of complex eigenvalues.

    Args:
        evals (complex float array): eigenvalues

    Returns:
        float: array of complex spacing ratios
    """
    
    print(f"[*] computing complex spacing ratios")

    #evals, n_removed = remove_near_duplicates(evals)
    #print("removed:", n_removed)
    evals = np.asarray(evals)
    pts = np.column_stack([evals.real, evals.imag])
    tree = cKDTree(pts)
    distances, indices = tree.query(pts, k=3)  # k=1 is self, k=2 NN, k=3 NNN

    r1 = evals[indices[:, 1]] - evals
    r2 = evals[indices[:, 2]] - evals
    r1 = clean_num_error(r1)
    r2 = clean_num_error(r2)

    z = r1 / r2
    valid_mask = np.isfinite(z)
    z = z[valid_mask]
    r = np.abs(z)

    return z, r

def sector_stats (sector_evals, stats_func=complex_spacing_ratios):
    """Compute complex spacing ratios (or different statistics) within each M sector and pool results.

    Args:
        sector_evals (dict): {M: eigenvalues} split into sectors defined by N superoperator eigenvalues
        stats_func (callable): function that accepts evals and returns (z, r). Defaults to complex_spacing_ratios.

    Returns:
        np.ndarray: pooled complex spacing ratios z
    """
    all_z = []
    for M, evals in sector_evals.items():

        #blocks <2 cannot produce spacing ratios
        if len(evals) < 3:
            continue

        z, r = stats_func(evals)
        all_z.append(z)
        print(f"M={M:+d}: dim={len(evals)}, ⟨r⟩={r.mean():.4f}") #sector M statistics

    z_pooled = np.concatenate(all_z)
    print(f"Pooled ⟨r⟩ = {np.abs(z_pooled).mean():.4f}  (GinUE: 0.739, Poisson: 0.500)") #total statistics

    return z_pooled

########################## SPACING STATISTICS ##########################

def level_spacings(evals, unfolding=False, degree=5):

    evals = np.sort(evals)

    if unfolding:
        evals = cut_edges(evals, 0.05)
        evals = polynomial_unfolding(evals, degree)

    # stretch the spectrum to get an exact average level density 1
    #evals = (evals - evals[0]) / (evals[-1] - evals[0]) * len(evals)

    spacings = np.diff(evals)
    #spacings = spacings[spacings > 1e-10]
    #spacings /= np.mean(spacings)  # makes ⟨s⟩ = 1

    return spacings

def normalized_nn_spacings(evals, kde_bw=None, remove_edge_frac=0.1, min_dist=1e-12):
    """Compute normalized nearest-neighbour spacings for complex eigenvalues."""

    evals = np.asarray(evals, dtype=np.complex128)
    pts = np.column_stack((evals.real, evals.imag))
    M = len(pts)

    # estimate local 2D density
    rho = stats.gaussian_kde(pts.T, bw_method=kde_bw)(pts.T)/(2*np.pi)

    # remove outer points
    r = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    cutoff = np.quantile(r, 1 - remove_edge_frac)
    keep = np.where(r <= cutoff)[0] if M > 5 else np.arange(M)
    pts, rho = pts[keep], rho[keep]

    # compute nearest-neighbour distances
    dists, _ = spatial.cKDTree(pts).query(pts, k=2)
    spacings_raw = dists[:, 1]

    # remove degenerate or almost-zero distances
    valid_indices = spacings_raw > min_dist
    spacings_raw = spacings_raw[valid_indices]
    rho = rho[valid_indices]

    # normalize by density and mean
    spacings_norm = spacings_raw * np.sqrt(rho)
    spacings_norm /= spacings_norm.mean()

    return spacings_norm
    
def r_statistic(evals):
    evals = np.sort(evals)
    s = np.diff(evals)
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    return np.mean(r)


def polynomial_unfolding(evals, degree, figure=True):
    """Unfolds the spectrum (smooth density given by a polynomial fit)

    Args:
        evals (float array): eigenvalues (sorted)
        degree (int): degree of the polynomial

    Returns:
        float-array: unfolded spectrum
    """
    
    p = np.polynomial.Polynomial.fit(evals, range(len(evals)), degree)

    if figure:
        plt.plot(evals, range(len(evals)), label="Data")
        plt.plot(*p.linspace(), label=f"Polynomial of degree {degree}")
        plt.title("Polynomial Unfolding")
        plt.legend()
        plt.show()

    return p(evals)

def cut_edges(evals, edge_frac):
    """Removes eigenvalues from the spectrum edge

    Args:
        evals (float array): eigenvalues
        edge_frac (float): fraction 0.0-0.5 of eigenvalues to remove from each side of the spectrum

    Returns:
        float array: cut spectrum
    """

    num_states = int(np.floor(len(evals)*edge_frac))
    return evals[num_states : -num_states]

########################## USEFUL FUNCS ##########################

def clean_num_error(values, tol=1e-10):
    """Zero out real or imaginary parts of eigenvalues within numerical tolerance.

    Args:
        values (np.ndarray): complex eigenvalues
        tol (float): tolerance threshold

    Returns:
        np.ndarray: cleaned eigenvalues
    """
    values = values.copy()
    values.real[np.abs(values.real) < tol] = 0.0
    values.imag[np.abs(values.imag) < tol] = 0.0
    return values

def remove_near_duplicates(evals, eps=1e-8):
    """Remove eigenvalues that are closer than eps in the complex plane."""
    evals = np.asarray(evals, dtype=np.complex128)
    pts = np.column_stack([evals.real, evals.imag])
    tree = cKDTree(pts)
    pairs = tree.query_pairs(eps)
    
    to_remove = set()
    for i, j in pairs:
        # remove one of each nearly identical pair (e.g., keep lower index)
        to_remove.add(j if np.abs(evals[j]) >= np.abs(evals[i]) else i)
    
    mask = np.ones(len(evals), dtype=bool)
    mask[list(to_remove)] = False
    cleaned = evals[mask]
    return cleaned, len(to_remove)

def poisson(s):
    return np.exp(-s)

def poisson2d(s):
    return np.pi/2 * s * np.exp(-np.pi * s**4)

def wigner(s):
    return np.pi/2 * s * np.exp(-np.pi/4 * s**2)

def p_ginUE(s, K=50, J=50):
    """computes Ginimbre ansamble from p_GIN (s) = product_k(gamma_func(1+l, s^2)/k!) sum_j((2s^2j+1 e^-s^2)/gamma_func(1+j, s^2))

    Args:
        s (_type_): _description_
        K (int, optional): _description_. Defaults to 50.
        J (int, optional): _description_. Defaults to 50.

    Returns:
        _type_: _description_
    """

    mp.mp.dps = 50

    s = mp.mpf(s)
    s2 = s**2

    # infinite product (truncated)
    prod = mp.mpf(1)
    for k in range(1, K + 1):
        prod *= mp.gammainc(1 + k, s2, mp.inf) / mp.factorial(k)

    # infinite sum (truncated)
    summ = mp.mpf(0)
    for j in range(1, J + 1):
        summ += (2 * s**(2*j + 1) * mp.e**(-s2)) / mp.gammainc(1 + j, s2, mp.inf)

    return prod * summ

