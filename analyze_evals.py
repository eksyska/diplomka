import os
from qm_statistics import *
from plot import *
from outputs import *


"""
Complex spacing ratio statistics of eigenvalues stored by test.py (data/evals/*.npz), the classic
statistic and the sliding window version.

With compare_refs = True the same statistics are computed for GinUE (chaotic) and Poisson (regular)
spectra, so the curves of the data can be compared with the ideal ones, including their noise and
their behaviour at the spectrum edges.
"""

################## THE IMPORTANT SETTINGS ##################


# U = -1  gamma = 3.0
# U = -5  gamma = 3.5
# U = -10 gamma = 0.5
# U = -15 gamma = 2.0
# U = -20 gamma = 1.0
evals_files = [
    "data/evals/circulation_L3_N21_ncut21_J1_U-1_f0.7_det1_gamma(0.0, 0.0, 0.0, 3.0, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-5_f0.7_det1_gamma(0.0, 0.0, 0.0, 3.5, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-10_f0.7_det1_gamma(0.0, 0.0, 0.0, 0.5, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-15_f0.7_det1_gamma(0.0, 0.0, 0.0, 2.0, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-20_f0.7_det1_gamma(0.0, 0.0, 0.0, 1.0, 0.0, 0.3).npz",
]

frac = 1 # classic statistic: fraction of the eigenvalues with the smallest |λ| (1 = whole spectrum)

# sliding window
width = "auto" # window width, "auto" = smallest width with at least min_count ratios in every window of the bulk
step = None # step of the window center, None = width/4
slice_by = "abs" # "abs" for |λ|, "real" for |Re λ|, "rank" for the rank of |λ| (same # of ratios in every window)
min_count = 1000 # windows with fewer ratios are np.nan

show_figs = True # False = figures are saved to figs/fig_dir instead of shown
fig_dir = "evals_analysis"
show_slider = True # interactive figure per data file with a slider over the window position (only with show_figs)

# reference spectra
compare_refs = False # GinUE and Poisson spectra through the same statistics
ref_size = 2000 # size of one GinUE matrix = # of Poisson points of one realization
n_realizations = 30 # ref_size * n_realizations should be comparable to the # of eigenvalues of one data file
seed = 1
ref_dir = "data/evals_ref"

############################################################

if not show_figs:
    os.makedirs(os.path.join("figs", fig_dir), exist_ok=True)


###################################### SPECTRA ######################################

spectra = [] # (label, block_evals, is_reference)

for path in evals_files:

    block_evals, _ = load_block_evals(path)
    label = os.path.splitext(os.path.basename(path))[0]

    print(f"{label}: {len(block_evals)} blocks {list(block_evals)}, sizes {[len(ev) for ev in block_evals.values()]}")
    spectra.append((label, block_evals, False))

if compare_refs:

    # GinUE: i.i.d. complex gaussian entries with variance 1/n, the spectrum fills the unit disk;
    # diagonalized once and loaded afterwards. Every realization is one block, the ratios are computed within it
    gin_path = os.path.join(ref_dir, f"GinUE_n{ref_size}_r{n_realizations}_seed{seed}.npz")

    if os.path.exists(gin_path):
        gin_evals, _ = load_block_evals(gin_path)
        print(f"GinUE eigenvalues loaded from {gin_path}")

    else:
        rng = np.random.default_rng(seed)
        blocks = {("GinUE", i): (rng.normal(size=(ref_size, ref_size)) + 1j * rng.normal(size=(ref_size, ref_size)))
                                / np.sqrt(2 * ref_size)
                  for i in range(n_realizations)}
        gin_evals = evals_from_blocks(blocks, save_path=gin_path)
        del blocks

    # Poisson: uncorrelated points uniform in the unit disk (own generator, independent of whether GinUE was loaded)
    rng = np.random.default_rng(seed + 1)
    poi_evals = {("Poisson", i): np.sqrt(rng.uniform(size=ref_size)) * np.exp(2j * np.pi * rng.uniform(size=ref_size))
                 for i in range(n_realizations)}

    spectra += [("GinUE", gin_evals, True), ("Poisson", poi_evals, True)]


###################################### STATISTICS ######################################

summary = [] # (label, # of ratios, <|r|>, its error, -<cos θ>, its error)
windows = {} # label -> window dict

for label, block_evals, is_ref in spectra:

    print(f"\n#### {label} ####")

    # classic statistic, ratios computed per block and pooled
    z = csr_from_evals(block_evals, complex_spacing_ratios, frac=frac)
    n = len(z)
    summary.append((label, n, np.abs(z).mean(), np.abs(z).std() / np.sqrt(n),
                    -np.cos(np.angle(z)).mean(), np.cos(np.angle(z)).std() / np.sqrt(n)))

    # sliding window; a width / step in units of |λ| of the data means nothing for the unit-disk references,
    # they use width="auto" with the same min_count, i.e. windows with a comparable number of ratios
    _, windows[label] = csr_from_evals(block_evals, complex_spacing_ratios,
                                       width="auto" if is_ref else width, step=None if is_ref else step,
                                       slice_by=slice_by, min_count=min_count, include_start=include_start)


# printed before the figures, plt.show() blocks until they are closed
w = max(len(row[0]) for row in summary) + 10
print(f"\n{'spectrum':<{w}} {'# ratios':>8}   {'<|r|>':<17} {'-<cos th>':<17}")
for label, n, r, dr, c, dc in summary:
    print(f"{label:<{w}} {n:>8}   {r:.4f} +- {dr:.4f}  {c:.4f} +- {dc:.4f}")
for name, (r, c) in CSR_REF.items():
    print(f"{name + ' (theory)':<{w}} {'':>8}   {r:.4f}            {c:.4f}")

ref_windows = [(windows["GinUE"], "GinUE (simulated)"), (windows["Poisson"], "Poisson (simulated)")] if compare_refs else None


###################################### COMPARISON ######################################

if compare_refs:

    # reference spectra side by side, one realization each (all realizations together would fill the disk evenly
    # and hide the difference: GinUE eigenvalues repel each other, Poisson points cluster and leave gaps)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for a, (name, ref_evals) in zip(ax, [("GinUE", gin_evals), ("Poisson", poi_evals)]):
        ev = next(iter(ref_evals.values()))
        a.scatter(ev.real, ev.imag, s=4, alpha=0.7)
        a.add_artist(plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--"))
        a.set_aspect("equal")
        a.grid(True, alpha=0.3)
        a.set_xlabel("Re(λ)")
        a.set_ylabel("Im(λ)")
        a.set_title(f"{name}, {len(ev)} eigenvalues (one realization)")

    plt.tight_layout()

    if show_figs:
        plt.show()
    else:
        fig.savefig(os.path.join("figs", fig_dir, "refs_spectrum.png"))
        plt.close("all")

    for label, _, is_ref in spectra:

        if is_ref:
            continue

        # spectra of different scales -> window center relative to the largest eigenvalue size (a rank already is)
        fig = plot_csr_window(windows[label], show=show_figs, normalize=slice_by != "rank", title=label,
                              others=ref_windows)

        if not show_figs:
            fig.savefig(os.path.join("figs", fig_dir, f"{label}_window_vs_refs.png"))
            plt.close("all")


###################################### INTERACTIVE ######################################

if show_slider and show_figs:

    for label, _, is_ref in spectra:

        if not is_ref:
            plot_csr_window_slider(windows[label], others=ref_windows, title=label)
