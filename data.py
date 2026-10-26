import os
import io
import csv
import glob
import time
import contextlib
import numpy as np

from qm_statistics import *
from BoseHubbard import BoseHubbard
from config import *


"""
Scan of the complex spacing ratio statistics <|r|> and -<cos θ> over the (U, g_circ_p) plane.

Every grid point is appended to a CSV file right after it is computed, so a crashed or interrupted
run can be restarted with the same settings and it continues where it stopped (finished points are skipped).

With save_evals = True the eigenvalues of all blocks of every grid point are also stored, one .npz file
per point in the folder evals_dir. The statistics of any part of the spectrum (e.g. the eigenvalues closest
to zero) can then be computed afterwards without recomputing the scan, see csr_stats and plot_scan_selected.
"""

################## THE IMPORTANT SETTINGS ##################

L = 4
N = 5

n_cut = N
n_local_max = n_cut

J = 1

f = 0.7 #driving
det = 1 #detuning

g_loss = 0.9 # loss
g_pump = 0.2 # pumping  /  kappa = g_loss - g_pump
g_deph = 0.0 # dephasing
g_circ_m = 0.0 # directed circulation minus
g_bpl = 0.3 # bond phase locking

config_name = "circulation"
config = CONFIGS[config_name] #config encodes symmetries, driving and nullified gammas

# scanned grid
U_values = np.linspace(-25, 0, 101)
g_circ_p_values = np.linspace(0, 5, 81)

out_path = f"data/scan_U_gcircp_{config_name}_L{L}_N{N}_J{J}_gm{g_circ_m}.csv"

save_evals = True # store the eigenvalues of every grid point (needed for plot_scan_selected)
evals_dir = os.path.splitext(out_path)[0] + "_evals"


############################################################

FIELDS = ["U", "g_circ_p", "r_mean", "cos_mean", "n_ratios", "time",
          "L", "N", "n_cut", "J", "f", "det", "g_loss", "g_pump", "g_deph", "g_circ_m", "g_bpl", "config"]


def _key(U, g):
    return (round(float(U), 10), round(float(g), 10))


def load_done(path):
    """Grid points already stored in the CSV file

    Returns:
        set: (U, g_circ_p) keys of finished points
    """

    if not os.path.exists(path):
        return set()

    with open(path, newline="") as fh:
        return {_key(row["U"], row["g_circ_p"]) for row in csv.DictReader(fh)}


def append_row(path, row):
    """Appends one row to the CSV file (with header if the file is new) and forces it to disk"""

    new_file = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


###################################### EIGENVALUE STORAGE ######################################

def _evals_file(folder, U, g):
    # repr of a float round-trips exactly, the key is parsed back from the name in load_done_evals
    return os.path.join(folder, f"U={float(U)!r}_g={float(g)!r}.npz")


def save_point_evals(folder, U, g, block_evals, params):
    """Stores the eigenvalues of all blocks of one grid point

    The file is written atomically, see save_block_evals.

    Args:
        folder (str): folder of the scan
        U, g (float): grid point
        block_evals (dict): sector label -> eigenvalues
        params (dict): model parameters stored with the eigenvalues
    """

    save_block_evals(_evals_file(folder, U, g), block_evals, {"U": float(U), "g_circ_p": float(g), **params})


def load_point_evals(path):
    """Loads one grid point stored by save_point_evals

    Returns:
        (dict, dict): sector label (tuple) -> eigenvalues, and the stored parameters (incl. U, g_circ_p)
    """

    return load_block_evals(path)


def load_done_evals(folder):
    """Grid points whose eigenvalues are already stored

    Returns:
        set: (U, g_circ_p) keys of finished points
    """

    done = set()
    for path in glob.glob(os.path.join(folder, "U=*_g=*.npz")):
        U, g = os.path.basename(path)[2:-4].split("_g=")
        done.add(_key(U, g))

    return done


###################################### STATISTICS ######################################

def compute_point(bh, fock_basis, L_basis):
    """Eigenvalues of all Liouvillian blocks for the current parameters of bh

    Returns:
        dict: sector label -> eigenvalues
    """

    # the block building prints every sector, keep the scan output readable
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = bh.build_L_blocks(fock_basis, L_basis)

    return evals_from_blocks(blocks)


def csr_stats(block_evals, select=None, min_count=1):
    """Pooled <|r|> and -<cos θ> of the complex spacing ratios of all blocks of one grid point

    The ratios are always computed on the full spectrum of each block (a cut spectrum would give
    the eigenvalues at the cut wrong nearest neighbours), the selection is applied afterwards
    to the eigenvalue each ratio was computed for.

    Args:
        block_evals (dict): sector label -> eigenvalues
        select (callable, optional): select(centers, all_evals) -> bool mask over centers, where centers
            are the eigenvalues of the ratios and all_evals the pooled eigenvalues of all blocks of the point
            (see smallest_abs, abs_below, re_window). Defaults to None (all ratios)
        min_count (int, optional): fewer selected ratios give np.nan. Defaults to 1

    Returns:
        (float, float, int): <|r|>, -<cos θ>, number of selected ratios
    """

    z_all, centers_all = [], []

    for ev in block_evals.values():

        if len(ev) < 3:
            continue

        z, centers = complex_spacing_ratios(ev, return_centers=True)
        z_all.append(z.ravel())
        centers_all.append(centers.ravel())

    z = np.concatenate(z_all)

    if select is not None:
        mask = select(np.concatenate(centers_all), pool_evals(block_evals))
        z = z[mask]

    if len(z) < max(min_count, 1):
        return np.nan, np.nan, len(z)

    return np.abs(z).mean(), -np.cos(np.angle(z)).mean(), len(z)


# The selections below work on the pooled spectrum of a grid point (all blocks together), so one
# threshold in |λ| is shared by all blocks and every block contributes its own slow modes.
# A fraction per block would instead mix different |λ| ranges of blocks of different size.
#
# The steady state λ = 0 (one per block containing it) is removed from the centers by exclude_zero,
# it stays in the spectrum as a neighbour of the other eigenvalues. Exactly degenerate eigenvalues are
# handled inside complex_spacing_ratios (a ratio with zero NNN distance is dropped, a zero NN distance gives z = 0).

ZERO_TOL = 1e-8


def smallest_abs(p=0.2, exclude_zero=True):
    """Selection of the fraction p of the eigenvalues of a grid point with the smallest |λ|

    A fixed fraction is comparable across the grid even though the spectrum scale changes with U and the rates.
    """

    def select(centers, all_evals):
        mask = np.abs(centers) <= np.quantile(np.abs(all_evals), p)
        if exclude_zero:
            mask &= np.abs(centers) > ZERO_TOL
        return mask

    select.label = f"{p:.0%} smallest |λ|"
    return select


def abs_below(R, exclude_zero=True):
    """Selection of the eigenvalues with |λ| < R

    Warning: the spectrum scale changes with U and the rates, so a fixed R selects
    different parts of the spectrum in different grid points.
    """

    def select(centers, all_evals):
        mask = np.abs(centers) < R
        if exclude_zero:
            mask &= np.abs(centers) > ZERO_TOL
        return mask

    select.label = f"|λ| < {R:g}"
    return select


def re_window(lo, hi):
    """Selection of the eigenvalues with lo <= |Re λ| <= hi (same fixed-scale warning as abs_below)"""

    def select(centers, all_evals):
        return (np.abs(centers.real) >= lo) & (np.abs(centers.real) <= hi)

    select.label = f"{lo:g} ≤ |Re λ| ≤ {hi:g}"
    return select


###################################### SCAN ######################################

def run_scan():

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done_csv = load_done(out_path)
    done = load_done_evals(evals_dir) if save_evals else done_csv

    grid = [(U, g) for U in U_values for g in g_circ_p_values]
    todo = [(U, g) for U, g in grid if _key(U, g) not in done]
    print(f"{evals_dir if save_evals else out_path}: {len(grid) - len(todo)}/{len(grid)} points already done, "
          f"{len(todo)} to compute")

    if not todo:
        return

    params = {"L": L, "N": N, "n_cut": n_cut, "J": J, "f": f, "det": det,
              "g_loss": g_loss, "g_pump": g_pump, "g_deph": g_deph, "g_circ_m": g_circ_m, "g_bpl": g_bpl,
              "config": config_name}

    # the basis does not depend on U and gamma -> built once, only the parameters of bh change
    bh = BoseHubbard(L, N, J, U_values[0], config, (f, det),
                     (g_loss, g_pump, g_deph, g_circ_p_values[0], g_circ_m, g_bpl),
                     n_local_max=n_local_max, n_cut=n_cut,
                     kappa_list=[0, 1, 2], pi_list=[], N_pairs=[(N, N)])

    fock_basis = bh.build_basis(fixed_N=False)
    L_basis = bh.build_sym_basis(fock_basis)

    for i, (U, g) in enumerate(todo, 1):

        bh.U = U
        bh.gamma = (g_loss, g_pump, g_deph, g, g_circ_m, g_bpl)

        t0 = time.time()
        block_evals = None
        try:
            block_evals = compute_point(bh, fock_basis, L_basis)
            r_mean, cos_mean, n_ratios = csr_stats(block_evals)
        except Exception as e:
            # store the failed point as NaN so the scan goes on (delete the row to recompute it)
            print(f"  !! U = {U:.4g}, g_circ_p = {g:.4g} failed: {e!r}")
            r_mean, cos_mean, n_ratios = np.nan, np.nan, 0
        dt = time.time() - t0

        # points of an older run without eigenvalues are already in the CSV, their row is kept
        if _key(U, g) not in done_csv:
            append_row(out_path, {"U": U, "g_circ_p": g, "r_mean": r_mean, "cos_mean": cos_mean,
                                  "n_ratios": n_ratios, "time": round(dt, 3), **params})

        # written last: the point counts as done only when both the CSV row and the eigenvalues exist
        if save_evals and block_evals is not None:
            save_point_evals(evals_dir, U, g, block_evals, params)

        print(f"[{i}/{len(todo)}] U = {U:.4g}, g_circ_p = {g:.4g}: <|r|> = {r_mean:.4f}, "
              f"-<cos th> = {cos_mean:.4f}  ({n_ratios} ratios, {dt:.1f} s)")


###################################### LOADING ######################################

def _to_grid(points):
    """Arranges (U, g_circ_p, <|r|>, -<cos θ>) tuples into 2D arrays, missing points are np.nan"""

    U = np.unique([p[0] for p in points])
    g = np.unique([p[1] for p in points])
    R = np.full((len(U), len(g)), np.nan)
    C = np.full((len(U), len(g)), np.nan)

    for u, gg, r, c in points:
        i = np.searchsorted(U, u)
        j = np.searchsorted(g, gg)
        R[i, j] = r
        C[i, j] = c

    return U, g, R, C


def load_grid(path):
    """Loads the scan into 2D arrays for plotting (e.g. plt.pcolormesh(U, g, R.T))

    Returns:
        (np.ndarray, np.ndarray, np.2darray, np.2darray): U values, g_circ_p values,
            <|r|> and -<cos θ> with shape (len(U), len(g_circ_p)), missing points are np.nan
    """

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    return _to_grid([(float(r["U"]), float(r["g_circ_p"]), float(r["r_mean"]), float(r["cos_mean"])) for r in rows])


def load_grid_selected(folder, select=None, min_count=200):
    """Recomputes <|r|> and -<cos θ> of every stored grid point from a selected part of its spectrum

    Args:
        folder (str): eigenvalue folder of the scan (evals_dir)
        select (callable, optional): selection, see csr_stats. Defaults to None (all eigenvalues)
        min_count (int, optional): grid points with fewer selected ratios are np.nan. Defaults to 200

    Returns:
        (np.ndarray, np.ndarray, np.2darray, np.2darray): as load_grid
    """

    points = []
    for path in glob.glob(os.path.join(folder, "U=*_g=*.npz")):
        block_evals, params = load_point_evals(path)
        r, c, _ = csr_stats(block_evals, select=select, min_count=min_count)
        points.append((params["U"], params["g_circ_p"], r, c))

    if not points:
        raise ValueError(f"no stored eigenvalues in {folder}")

    return _to_grid(points)


def scan_label(path):
    """Title of a scan (CSV file or eigenvalue folder) built from its stored parameters"""

    if os.path.isdir(path):
        _, row = load_point_evals(glob.glob(os.path.join(path, "U=*_g=*.npz"))[0])
    else:
        with open(path, newline="") as fh:
            row = next(csv.DictReader(fh))

    return f"L = {row['L']}, N = {row['N']}"


###################################### PLOTTING ######################################

def plot_grids(grids, titles, suptitle=None, show=True, save_path=None):
    """Heat maps of <|r|> (top row) and -<cos θ> (bottom row) over the (U, g_circ_p) plane, one column per grid

    The color scale is diverging around the Poisson value (white): red towards GinUE (chaos),
    blue below Poisson (level clustering). The limits are shared by all columns of a row,
    the reference values are marked in the colorbars.

    Args:
        grids (list): (U, g_circ_p, <|r|>, -<cos θ>) tuples from load_grid / load_grid_selected
        titles (list of str): column titles
        suptitle (str, optional): figure title. Defaults to None
        show (bool, optional): Defaults to True
        save_path (str, optional): file to save the figure to. Defaults to None (not saved)
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(2, len(grids), figsize=(4.5 * len(grids) + 1, 7), squeeze=False,
                           sharex=True, sharey=True, layout="constrained")

    for k, ylabel in enumerate(["<|r|>", "-<cos θ>"]):

        r_gin, r_poi = CSR_REF["GinUE"][k], CSR_REF["Poisson"][k]
        vals = np.concatenate([grid[2 + k].ravel() for grid in grids])
        vals = vals[np.isfinite(vals)]
        norm = TwoSlopeNorm(vcenter=r_poi,
                            vmin=min(vals.min(initial=r_poi), r_poi - 1e-3),
                            vmax=max(vals.max(initial=r_gin), r_gin))

        for col, ((U, g, R, C), title) in enumerate(zip(grids, titles)):
            im = ax[k, col].pcolormesh(U, g, (R, C)[k].T, shading="nearest", cmap="RdBu_r", norm=norm)
            if k == 0:
                ax[k, col].set_title(title)

        cb = fig.colorbar(im, ax=ax[k, :], label=ylabel)
        # reference values as lines in the colorbar, labeled on its left side
        for (name, ref), ls in zip(CSR_REF.items(), ["--", ":"]):
            cb.ax.axhline(ref[k], color="k", ls=ls, lw=1.2)
            cb.ax.text(-0.15, ref[k], name, transform=cb.ax.get_yaxis_transform(),
                       va="center", ha="right", fontsize=7)

    for a in ax[-1, :]:
        a.set_xlabel("U")
    for a in ax[:, 0]:
        a.set_ylabel("g_circ_p")

    if suptitle is not None:
        fig.suptitle(suptitle)

    if save_path is not None:
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()

    return fig


def plot_scan(paths, show=True, save_path=None):
    """Heat maps of the full-spectrum statistics stored in scan CSV files, see plot_grids

    Args:
        paths (list of str): scan CSV files
    """

    return plot_grids([load_grid(p) for p in paths], [scan_label(p) for p in paths],
                      show=show, save_path=save_path)


def plot_scan_selected(folders, select, min_count=200, show=True, save_path=None):
    """Heat maps of the statistics of a selected part of the spectra stored in eigenvalue folders, see plot_grids

    Args:
        folders (list of str): eigenvalue folders of the scans (evals_dir)
        select (callable): selection, e.g. smallest_abs(0.2), see csr_stats
        min_count (int, optional): grid points with fewer selected ratios are blank. Defaults to 200
    """

    grids = [load_grid_selected(d, select=select, min_count=min_count) for d in folders]
    label = getattr(select, "label", "selected eigenvalues")

    return plot_grids(grids, [scan_label(d) for d in folders], suptitle=f"{label} (min. {min_count} ratios)",
                      show=show, save_path=save_path)


#run_scan()
"""
plot_scan([
    "data/scan_U_gcircp_circulation_L3_N12_J1_gm0.0.csv",
    "data/scan_U_gcircp_circulation_L4_N6_J1_gm0.0.csv",
])"""

# scans with stored eigenvalues (save_evals = True)

evals_dirs = [
    "data/scan_U_gcircp_circulation_L3_N11_J1_gm0.0_evals",
]

frac = 0.2 # fraction of the eigenvalues with the smallest |λ| used for the statistics

# full spectrum of the same scans for comparison
#plot_scan([d.removesuffix("_evals") + ".csv" for d in evals_dirs])

plot_scan_selected(evals_dirs, smallest_abs(frac))
