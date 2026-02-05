import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy import stats, spatial
import mpmath as mp


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

########################## COMPLEX SPACING RATIOS ##########################

def complex_spacing_ratios(evals):
    """Compute complex spacing ratios for a list/array of complex eigenvalues.

    Args:
        evals (complex float array): eigenvalues

    Returns:
        float: array of complex spacing ratios
    """

    #evals, n_removed = remove_near_duplicates(evals)
    #print("removed:", n_removed)
    evals = np.asarray(evals)
    pts = np.column_stack([evals.real, evals.imag])

    nn = NearestNeighbors(n_neighbors=3).fit(pts)

    distances, indices = nn.kneighbors(pts)

    r1 = evals[indices[:, 1]] - evals
    r2 = evals[indices[:, 2]] - evals
    return r1 / r2

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




########################## USEFUL FUNCS ##########################

def poisson(s):
    return np.exp(-s)

def poisson2d(s):
    return np.pi/2 * s * np.exp(-np.pi * s**4)

def wigner(s):
    return np.pi/2 * s * np.exp(-np.pi/4 * s**2)








