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


def pool_evals(block_evals):
    """Concatenate eigenvalues from blocks

    Args:
        block_evals (dict): label -> eigenvalues

    Returns:
        np.ndarray: concatenated eigenvalues
    """

    # dict -> list -> concatenate
    return np.concatenate(list(block_evals.values()))


def csr_from_evals(block_evals, csr_func, n_slices=None, width=None, step=None, slice_by="abs", min_count=200, frac=None):
    """Compute complex spacing ratios separately per block and pool results

    The ratios are always computed on the full block spectrum, the selection by frac and the slicing / windowing
    are done afterwards according to the size of the eigenvalue each ratio was computed for

    Args:
        block_evals (dict): label -> eigenvalues
        csr_func (callable): function that takes eigenvalues and returns complex spacing ratios,
            for slicing it has to accept return_centers=True and return (ratios, centers)
        n_slices (int, optional): number of eigenvalue slices of equal width. Defaults to None (no slicing)
        width (float or str, optional): width of the sliding window, "auto" to choose it from min_count (see window_csr). Defaults to None (no window)
        step (float, optional): step of the sliding window center. Defaults to None (width/4)
        slice_by (str, optional): eigenvalue size measure, "abs" for |λ|, "real" for |Re λ|. Defaults to "abs"
        min_count (int, optional): required number of ratios in a window, windows with fewer give np.nan. Defaults to 200
        frac (float, optional): keep only the ratios of the smallest frac (0-1] of eigenvalues by slice_by size
            (over all blocks together), applied before slicing / windowing. Defaults to None (all ratios)

    Returns:
        np.ndarray: pooled complex spacing ratios (if neither n_slices nor width is given)
        (np.ndarray, list): pooled ratios and list of slice dicts (see slice_csr) (if n_slices is given)
        (np.ndarray, dict): pooled ratios and window dict (see window_csr) (if width is given)
    """

    if n_slices is not None and width is not None:
        raise ValueError("give either n_slices or width, not both")

    sliced = n_slices is not None or width is not None
    need_centers = sliced or frac is not None

    all_ratios = []
    all_centers = []

    for evals in block_evals.values():

        if len(evals) < 3:
            continue

        if not need_centers:
            # compute statistics for all evals in a block together
            result = csr_func(evals)
            ratios = (result[0] if isinstance(result, tuple) else result).ravel()

        else:
            # keep the eigenvalue of each ratio for the selection / slicing
            ratios, centers = csr_func(evals, return_centers=True)
            ratios = ratios.ravel()
            all_centers.append(centers.ravel())

        all_ratios.append(ratios)

    if not all_ratios:
        raise ValueError("no blocks with at least 3 eigenvalues")

    pooled = np.concatenate(all_ratios)

    if frac is not None:
        centers = np.concatenate(all_centers)
        key = _SLICE_KEYS[slice_by][0](centers)
        cutoff = np.quantile(key, frac)
        keep = key <= cutoff
        print(f"keeping ratios of the smallest {frac:.0%} of {_SLICE_KEYS[slice_by][1]} "
              f"(<= {cutoff:.4g}): {keep.sum()} of {len(pooled)}")
        pooled, all_centers = pooled[keep], [centers[keep]]

    if not sliced:

        print(f"pooled <|r|> = {np.abs(pooled).mean():.4f}")
        print(f"pooled -<cos(arg(z))> = {-np.cos(np.angle(pooled)).mean():.4f}")

        return pooled

    centers = np.concatenate(all_centers)

    if n_slices is not None:
        slices = slice_csr(pooled, centers, n_slices, slice_by=slice_by)
        print_csr_slices(pooled, slices, slice_by=slice_by)
        return pooled, slices

    window = window_csr(pooled, centers, width, step=step, slice_by=slice_by, min_count=min_count)
    print(f"sliding window over {_SLICE_KEYS[slice_by][1]}: width = {window['width']:.4g}, step = {window['step']:.4g}, "
          f"{len(window['center'])} windows, {np.sum(window['n'] >= min_count)} with >= {min_count} ratios")
    print(f"pooled <|r|> = {np.abs(pooled).mean():.4f}")
    print(f"pooled -<cos(arg(z))> = {-np.cos(np.angle(pooled)).mean():.4f}")
    return pooled, window


_SLICE_KEYS = {
    "abs":  (lambda x: np.abs(x),      "|lam|"),
    "real": (lambda x: np.abs(x.real), "|Re lam|"),
}

# reference values (<|r|>, -<cos θ>) of the complex spacing ratios
CSR_REF = {
    "GinUE":   (0.7381, 0.2405),
    "Poisson": (2/3, 0.0),
}


def slice_csr(z, centers, n_slices, slice_by="abs"):
    """Split complex spacing ratios into slices by the size of the eigenvalue they belong to,
    the interval (min, max) of eigenvalue sizes is divided into n_slices slices of equal width

    Args:
        z (complex np.ndarray): complex spacing ratios
        centers (complex np.ndarray): eigenvalue for which each ratio was computed (same length as z)
        n_slices (int): number of eigenvalue slices
        slice_by (str, optional): eigenvalue size measure, "abs" for |λ|, "real" for |Re λ|. Defaults to "abs"

    Returns:
        list: one dict per slice with keys "range" (lo, hi), "z", "n", "mean_abs", "mean_cos"
    """

    key = _SLICE_KEYS[slice_by][0](np.asarray(centers))
    edges = np.linspace(key.min(), key.max(), n_slices + 1)

    # slice index 0 .. n_slices-1, the last edge is included in the last slice
    idx = np.searchsorted(edges[1:-1], key, side="right")

    return [_slice_dict(z[idx == i], (edges[i], edges[i+1])) for i in range(n_slices)]


def window_csr(z, centers, width, step=None, slice_by="abs", min_count=200, edge_frac=0.05):
    """Statistics of complex spacing ratios in a sliding window of eigenvalue sizes

    The window [c - width/2, c + width/2] is moved with its center c from the smallest
    to the largest eigenvalue size, in each position <|r|> and -<cos θ> are computed
    from the ratios whose eigenvalue lies in the window

    Args:
        z (complex np.ndarray): complex spacing ratios
        centers (complex np.ndarray): eigenvalue for which each ratio was computed (same length as z)
        width (float or str): window width, "auto" for the smallest width with at least min_count
            ratios in every window of the spectrum bulk (see auto_window_width)
        step (float, optional): step of the window center. Defaults to None (width/4)
        slice_by (str, optional): eigenvalue size measure, "abs" for |λ|, "real" for |Re λ|. Defaults to "abs"
        min_count (int, optional): windows with fewer ratios give np.nan. Defaults to 200
        edge_frac (float, optional): fraction of ratios at the spectrum edges ignored by width="auto". Defaults to 0.05

    Returns:
        dict: "center" window centers, "n" number of ratios, "mean_abs" <|r|>, "mean_cos" -<cos θ> (arrays),
            "width", "step", "slice_by"
    """

    key = _SLICE_KEYS[slice_by][0](np.asarray(centers))
    order = np.argsort(key)
    key = key[order]

    if width == "auto":
        width = auto_window_width(key, min_count, edge_frac=edge_frac)

    if step is None:
        step = width / 4

    # cumulative sums with a leading 0, sum over sorted indices [lo, hi) = cs[hi] - cs[lo]
    cs_abs = np.concatenate([[0.0], np.cumsum(np.abs(z[order]))])
    cs_cos = np.concatenate([[0.0], np.cumsum(-np.cos(np.angle(z[order])))])

    mid = np.arange(key[0], key[-1] + step/2, step)
    lo = np.searchsorted(key, mid - width/2, side="left")
    hi = np.searchsorted(key, mid + width/2, side="right")
    n = hi - lo

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_abs = (cs_abs[hi] - cs_abs[lo]) / n
        mean_cos = (cs_cos[hi] - cs_cos[lo]) / n

    mean_abs[n < min_count] = np.nan
    mean_cos[n < min_count] = np.nan

    return {"center": mid, "n": n, "mean_abs": mean_abs, "mean_cos": mean_cos,
            "width": width, "step": step, "slice_by": slice_by}


def auto_window_width(key, min_count, edge_frac=0.05, n_test=500):
    """Smallest window width such that every window centered in the spectrum bulk contains at least min_count ratios

    The bulk is the range between the edge_frac/2 and 1 - edge_frac/2 quantiles of the eigenvalue sizes,
    windows centered at the sparse spectrum edges are not required to be full (they get np.nan in window_csr)

    Args:
        key (np.ndarray): sorted eigenvalue sizes of the ratios
        min_count (int): required number of ratios in a window
        edge_frac (float, optional): fraction of ratios at the spectrum edges that is ignored. Defaults to 0.05
        n_test (int, optional): # of tested window centers. Defaults to 500

    Returns:
        float: window width
    """

    if len(key) < min_count:
        raise ValueError(f"only {len(key)} ratios in total, less than min_count = {min_count}")

    mid = np.linspace(*np.quantile(key, [edge_frac/2, 1 - edge_frac/2]), n_test)

    def fewest(w):
        lo = np.searchsorted(key, mid - w/2, side="left")
        hi = np.searchsorted(key, mid + w/2, side="right")
        return (hi - lo).min()

    # the count only grows with the width -> bisection, a window of twice the full range always contains everything
    w_lo, w_hi = 0.0, 2 * (key[-1] - key[0])
    for _ in range(50):
        w = (w_lo + w_hi) / 2
        if fewest(w) >= min_count:
            w_hi = w
        else:
            w_lo = w

    return w_hi


def _slice_dict(z, rng):
    """Builds dictionary of complex spacing ratios of one slice together with their statistics

    Args:
        z (complex np.ndarray): complex spacing ratios
        rng (tuple): (lo, hi) eigenvalue size range of the slice

    Returns:
        dict: "range" (lo, hi), "z" ratios, "n" number of ratios, "mean_abs" <|r|>, "mean_cos" -<cos(arg z)> (both np.nan for an empty slice)
    """

    return {
        "range": rng,
        "z": z,
        "n": len(z),
        "mean_abs": np.abs(z).mean() if len(z) else np.nan,
        "mean_cos": -np.cos(np.angle(z)).mean() if len(z) else np.nan,
    }


def print_csr_slices(z, slices, slice_by="abs"):
    """Print a table of <|r|> and -<cos th> per slice together with the total and reference values

    Args:
        z (complex np.ndarray): all complex spacing ratios
        slices (list): slice dicts
        slice_by (str, optional): eigenvalue size measure, "abs" for |λ|, "real" for |Re λ|. Defaults to "abs"
    """

    label = _SLICE_KEYS[slice_by][1]
    print(f"complex spacing ratios sliced by {label} ({len(slices)} slices of equal width)")
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

