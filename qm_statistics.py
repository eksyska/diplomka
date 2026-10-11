import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree, KDTree
from scipy import stats, spatial
import mpmath as mp

from math_funcs import *
from math_funcs import _ALL
from models import *

########################## BLOCK SPLITTING ##########################

def evals_from_blocks(blocks):
    """Diagonalize prebuilt blocks

    Args:
        blocks (dict): label -> np.ndarray block matrix

    Returns:
        dict: label -> np.ndarray eigenvalues
    """
    
    block_evals = {label: np.linalg.eigvals(block) for label, block in blocks.items()}
    return block_evals

def pool_evals(block_evals, kappa=_ALL, M=_ALL):
    """Concatenate eigenvalues from blocks matching the given sector filters

    Usually the block eigenvalues are already filtered so there is no need to filter again

    Args:
        block_evals (dict): label -> eigenvalues
        kappa (int or list or _Sentinel): ket-bra translation match filter. Defaults to _ALL (for all sectors)
        M (int or list or _Sentinel): particle number filter. Defaults to _ALL (for all sectors)

    Returns:
        np.ndarray: concatenated eigenvalues
    """

    evals = []

    for label, v in block_evals.items():

        if isinstance(label, tuple) and len(label) == 3:
            k, p, m = label
        else:
            k, p, m = _ALL, _ALL, label

        if (kappa is _ALL or k == kappa) and (M is _ALL or m == M):
            evals.append(v)

    if not evals:
        raise ValueError(f"no blocks matched kappa={kappa}, M={M}")
    
    return np.concatenate(evals)

def csr_from_evals(block_evals, csr_func, n_slices=None, slice_by="abs", slice_mode="quantile", slice_per_block=False):
    """Compute complex spacing ratios separately per block and pool results

    Args:
        block_evals (dict): label -> eigenvalues
        csr_func (callable): function that takes eigenvalues and returns complex spacing ratios,
            for slicing it has to accept return_centers=True and return (ratios, centers)
        n_slices (int, optional): number of eigenvalue slices (3-5 is reasonable). Defaults to None (no slicing)
        slice_by (str, optional): eigenvalue size measure, "abs" for |λ|, "real" for |Re λ|, "imag" for |Im λ|. Defaults to "abs"
        slice_mode (str, optional): "quantile" for slices with equal number of ratios, "linear" for slices of equal width. Defaults to "quantile"
        slice_per_block (bool, optional): slice each block separately and pool the slices. Defaults to False

    Returns:
        np.ndarray: pooled complex spacing ratios (if n_slices is None)
        (np.ndarray, list): pooled ratios and list of slice dicts (see slice_csr) (if n_slices is given)
    """

    all_ratios = []
    all_centers = []
    block_slices = []

    for evals in block_evals.values():

        if len(evals) < 3:
            continue

        if n_slices is None:
            result = csr_func(evals)
            ratios = (result[0] if isinstance(result, tuple) else result).ravel()
        else:
            ratios, centers = csr_func(evals, return_centers=True)
            ratios = ratios.ravel()
            all_centers.append(centers.ravel())

            if slice_per_block:
                block_slices.append(slice_csr(ratios, centers.ravel(), n_slices, slice_by=slice_by, slice_mode=slice_mode))

        all_ratios.append(ratios)

    if not all_ratios:
        raise ValueError("no blocks with at least 3 eigenvalues")

    pooled = np.concatenate(all_ratios)

    if n_slices is None:
        print(f"pooled <|r|> = {np.abs(pooled).mean():.4f}")
        print(f"pooled -<cos(arg(z))> = {-np.cos(np.angle(pooled)).mean():.4f}")
        return pooled

    if slice_per_block:
        slices = merge_slices(block_slices)
    else:
        centers = np.concatenate(all_centers)
        slices = slice_csr(pooled, centers, n_slices, slice_by=slice_by, slice_mode=slice_mode)

    mode = slice_mode + (" per block" if slice_per_block else "")
    print_csr_slices(pooled, slices, slice_by=slice_by, slice_mode=mode)
    return pooled, slices


_SLICE_KEYS = {
    "abs":  (lambda x: np.abs(x),      "|lam|"),
    "real": (lambda x: np.abs(x.real), "|Re lam|"),
    "imag": (lambda x: np.abs(x.imag), "|Im lam|"),
}

# reference values (<|r|>, -<cos θ>) of the complex spacing ratios
CSR_REF = {
    "GinUE":   (0.7381, 0.2405),
    "Poisson": (2/3, 0.0),
}

def slice_csr(z, centers, n_slices, slice_by="abs", slice_mode="quantile"):
    """Split complex spacing ratios into slices by the size of the eigenvalue they belong to

    Args:
        z (complex np.ndarray): complex spacing ratios
        centers (complex np.ndarray): eigenvalue for which each ratio was computed (same length as z)
        n_slices (int): number of slices
        slice_by (str, optional): "abs", "real" or "imag" (see csr_from_evals). Defaults to "abs"
        slice_mode (str, optional): "quantile" or "linear" (see csr_from_evals). Defaults to "quantile"

    Returns:
        list: one dict per slice with keys "range" (lo, hi), "z", "n", "mean_abs", "mean_cos"
    """

    key = _SLICE_KEYS[slice_by][0](np.asarray(centers))

    if slice_mode == "quantile":
        edges = np.quantile(key, np.linspace(0, 1, n_slices + 1))
    elif slice_mode == "linear":
        edges = np.linspace(key.min(), key.max(), n_slices + 1)
    else:
        raise ValueError(f"unknown slice_mode '{slice_mode}'")

    # slice index 0 .. n_slices-1, the last edge is included in the last slice
    idx = np.searchsorted(edges[1:-1], key, side="right")

    return [_slice_dict(z[idx == i], (edges[i], edges[i+1])) for i in range(n_slices)]

def merge_slices(slice_lists):
    """Pool slices from several sectors / runs, the i-th slices of all lists are joined together

    Useful when each sector was sliced separately (by its own edges), e.g. by separate
    csr_from_evals(..., n_slices=n) calls for different kappa or M

    Args:
        slice_lists (list): lists of slice dicts (from slice_csr / csr_from_evals), all with the same number of slices

    Returns:
        list: pooled slice dicts, "range" is the envelope of the pooled slice ranges (so neighbouring slices can overlap)
    """

    n_slices = {len(sl) for sl in slice_lists}
    if len(n_slices) != 1:
        raise ValueError(f"all slice lists must have the same number of slices, got {sorted(n_slices)}")

    merged = []
    for parts in zip(*slice_lists):
        z_i = np.concatenate([s["z"] for s in parts])
        rng = (min(s["range"][0] for s in parts), max(s["range"][1] for s in parts))
        merged.append(_slice_dict(z_i, rng))

    return merged

def _slice_dict(z, rng):
    return {
        "range": rng,
        "z": z,
        "n": len(z),
        "mean_abs": np.abs(z).mean() if len(z) else np.nan,
        "mean_cos": -np.cos(np.angle(z)).mean() if len(z) else np.nan,
    }

def print_csr_slices(z, slices, slice_by="abs", slice_mode="quantile"):
    """Print a table of <|r|> and -<cos th> per slice together with the total and reference values"""

    label = _SLICE_KEYS[slice_by][1]
    print(f"complex spacing ratios sliced by {label} ({slice_mode}, {len(slices)} slices)")
    print(f"{'slice':>7} {label + ' range':>22} {'n':>8} {'<|r|>':>8} {'-<cos th>':>9}")

    for i, s in enumerate(slices):
        lo, hi = s["range"]
        rng = f"[{lo:.3f}, {hi:.3f}]"
        print(f"{i+1:>7} {rng:>22} {s['n']:>8} {s['mean_abs']:>8.4f} {s['mean_cos']:>9.4f}")

    print(f"{'total':>7} {'':>22} {len(z):>8} {np.abs(z).mean():>8.4f} {-np.cos(np.angle(z)).mean():>9.4f}")

    for name, (r_ref, c_ref) in CSR_REF.items():
        print(f"{name:>7} {'(reference)':>22} {'':>8} {r_ref:>8.4f} {c_ref:>9.4f}")


########################## COMPLEX SPACING RATIOS ##########################

def complex_spacing_ratios(evals, return_centers=False):
    """Compute complex spacing ratios for a list/array of complex eigenvalues

    Args:
        evals (float np.array): Lindbladian eigenvalues
        return_centers (bool, optional): also return the eigenvalue each ratio was computed for. Defaults to False

    Returns:
        float np.array: complex spacing ratios
        (float np.array, complex np.array): ratios and their eigenvalues (if return_centers)
    """

    evals = np.asarray(evals)
    pts = np.column_stack([evals.real, evals.imag])
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=3, workers=-1)  # k=1 is self, k=2 NN, k=3 NNN

    r1 = evals[indices[:, 1]] - evals
    r2 = evals[indices[:, 2]] - evals
    r1 = clean_num_error(r1)

    z = r1 / r2
    valid_mask = np.isfinite(z)
    z = z[valid_mask]

    if return_centers:
        return z, evals[valid_mask]

    return z



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

