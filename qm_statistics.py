import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy import stats, spatial


#COMPLEX SPACING RATIO

def complex_spacing_ratios(evals):
    """
    Compute complex spacing ratios for a list/array of complex eigenvalues.
    Returns complex ratios z_i = (r1 / r2).
    """

    evals, n_removed = remove_near_duplicates(evals, eps=1e-14)
    evals = np.asarray(evals)
    pts = np.column_stack([evals.real, evals.imag])

    nn = NearestNeighbors(n_neighbors=3).fit(pts)

    distances, indices = nn.kneighbors(pts)

    r1 = evals[indices[:, 1]] - evals
    r2 = evals[indices[:, 2]] - evals
    return r1 / r2

def remove_near_duplicates(evals, eps=1e-12):
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


# SPACING STATISTICS

def normalized_nn_spacings(eigs, kde_bw=None, remove_edge_frac=0.1, min_dist=1e-12):
    """Compute normalized nearest-neighbour spacings for complex eigenvalues."""
    eigs = np.asarray(eigs, dtype=np.complex128)
    pts = np.column_stack((eigs.real, eigs.imag))
    M = len(pts)

    # Estimate local 2D density
    rho = stats.gaussian_kde(pts.T, bw_method=kde_bw)(pts.T)

    # Optionally remove outer points
    r = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    cutoff = np.quantile(r, 1 - remove_edge_frac)
    keep = np.where(r <= cutoff)[0] if M > 5 else np.arange(M)
    pts, rho = pts[keep], rho[keep]

    # Compute nearest-neighbour distances
    dists, _ = spatial.cKDTree(pts).query(pts, k=2)
    s_raw = dists[:, 1]

    # Remove degenerate or almost-zero distances
    valid = s_raw > min_dist
    s_raw = s_raw[valid]
    rho = rho[valid]

    # Normalize by density and mean
    s_norm = s_raw * np.sqrt(rho)
    s_norm /= s_norm.mean()

    return s_norm, rho, keep, s_raw

# USEFUL FUNCS

def poisson(s):
    return np.exp(-s)

def poisson2d(s):
    return 2 * np.pi * s * np.exp(-np.pi * s**2)

def wigner(s):
    return 2 * np.pi * s * np.exp(-np.pi * s**2)

