import os
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
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
"""
    "data/evals/circulation_L3_N21_ncut21_J1_U-1_f0.7_det1_gamma(0.0, 0.0, 0.0, 3.0, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-5_f0.7_det1_gamma(0.0, 0.0, 0.0, 3.5, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-10_f0.7_det1_gamma(0.0, 0.0, 0.0, 0.5, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-15_f0.7_det1_gamma(0.0, 0.0, 0.0, 2.0, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-20_f0.7_det1_gamma(0.0, 0.0, 0.0, 1.0, 0.0, 0.3).npz",
"""
evals_files = [
    "data/evals/circulation_L3_N21_ncut21_J1_U-1_f0.7_det1_gamma(0.0, 0.0, 0.0, 3.0, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-5_f0.7_det1_gamma(0.0, 0.0, 0.0, 3.5, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-10_f0.7_det1_gamma(0.0, 0.0, 0.0, 0.5, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-15_f0.7_det1_gamma(0.0, 0.0, 0.0, 1.0, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-15_f0.7_det1_gamma(0.0, 0.0, 0.0, 2.0, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-20_f0.7_det1_gamma(0.0, 0.0, 0.0, 1.0, 0.0, 0.3).npz",
    "data/evals/circulation_L3_N21_ncut21_J1_U-20_f0.7_det1_gamma(0.0, 0.0, 0.0, 2.0, 0.0, 0.3).npz",
]

frac = 1 # classic statistic: fraction of the eigenvalues with the smallest |λ| (1 = whole spectrum)

# sliding window
width = "auto" # window width, "auto" = smallest width with at least min_count ratios in every window of the bulk
step = None # step of the window center, None = width/4
slice_by = "real" # "abs" for |λ|, "real" for |Re λ|, "rank" for the rank of |λ| (same # of ratios in every window)
min_count = 1500 # windows with fewer ratios are np.nan
include_start = False # True = the first window starts at the smallest λ (the steady state) and "auto" makes it hold
                      # min_count ratios too -> much wider windows, the spectrum is sparse there

# pseudodegeneracies (the central peak of the ratios)
mark_z = 0.1 # ratios with |z| < mark_z are marked (Poisson: a fraction mark_z^2, GinUE ~0), None = no marking
mark_s = 0.3 # a marked eigenvalue is a "pair" if its NN is closer than mark_s x the local mean spacing, otherwise a "gap"
k_local = 10 # the local mean spacing is estimated from the distance to the k_local-th neighbour
map_z = 0.2 # looser threshold for the map of the excess over Poisson, so that its cells have enough counts
map_bins = (25, 25) # (Re, Im) cells of the map
show_pseudo = False # figure per data file with the marked pairs, the map and the histograms (the table is always printed
                    # and the slider always marks them, both with mark_z only)

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
                                       slice_by=slice_by, min_count=min_count, include_start=include_start,
                                       mark=mark_z)


# printed before the figures, plt.show() blocks until they are closed
w = max(len(row[0]) for row in summary) + 10
print(f"\n{'spectrum':<{w}} {'# ratios':>8}   {'<|r|>':<17} {'-<cos th>':<17}")
for label, n, r, dr, c, dc in summary:
    print(f"{label:<{w}} {n:>8}   {r:.4f} +- {dr:.4f}  {c:.4f} +- {dc:.4f}")
for name, (r, c) in CSR_REF.items():
    print(f"{name + ' (theory)':<{w}} {'':>8}   {r:.4f}            {c:.4f}")

ref_windows = [(windows["GinUE"], "GinUE (simulated)"), (windows["Poisson"], "Poisson (simulated)")] if compare_refs else None


###################################### PSEUDODEGENERACIES ######################################

# the eigenvalues behind the central peak of the ratios: |z| < mark_z either because the nearest neighbour is
# unusually close (a "pair", s < mark_s) or because the next-nearest one is unusually far (a "gap": sparse region,
# spectrum edge); s = NN distance / local mean spacing (see complex_spacing_ratios)
pseudo = {} # label -> block -> per ratio and per pair arrays

if mark_z is not None:

    for label, block_evals, is_ref in spectra:

        pseudo[label] = {}

        for block, ev in block_evals.items():

            z, lam, nn, s = complex_spacing_ratios(ev, return_nn=True, k_local=k_local)
            marked, close = np.abs(z) < mark_z, s < mark_s

            # groups of eigenvalues linked to their NN by s < mark_s (2 = doublet, 3 = triplet, ...)
            graph = coo_matrix((np.ones(close.sum()), (nn[close, 0], nn[close, 1])), shape=(len(ev), len(ev)))
            _, group = connected_components(graph, directed=False)

            # both eigenvalues of a doublet are usually marked, every pair is counted once
            pair = np.unique(np.sort(nn[marked & close], axis=1), axis=0)

            pseudo[label][block] = {"lam": lam, "z": z, "s": s, "marked": marked, "gap": marked & ~close,
                                    "pair": ev[pair], "pair_size": np.bincount(group)[group[pair[:, 0]]]}

    print(f"\n#### pseudodegeneracies: marked = |z| < {mark_z}, pair = NN closer than {mark_s} x local mean spacing ####")
    print("Poisson = expected # of marked for uncorrelated eigenvalues, gaps = marked but not a pair, "
          "2/3/4+ = pairs in groups of 2, 3, >= 4 close eigenvalues,\nhoriz / vert = fraction of pairs split "
          "within 15 deg along Re (decay rate) / Im (frequency), a random direction gives 0.17 each")

    for label, _, is_ref in spectra:

        res = pseudo[label]
        print(f"\n{label}")
        print(f"{'block':<14} {'n':>7} {'marked':>7} {'Poisson':>8} {'x':>6} {'gaps':>6} {'pairs':>6} {'2/3/4+':>12} "
              f"{'med|dlam|':>10} {'horiz':>6} {'vert':>6}")

        # every block of the data, only the total of the references (one block per realization)
        groups = ([] if is_ref else [(str(b), [b]) for b in res]) + [("all", list(res))]
        for name, blocks in groups:
            n = sum(len(res[b]["z"]) for b in blocks)
            n_marked = sum(res[b]["marked"].sum() for b in blocks)
            n_gap = sum(res[b]["gap"].sum() for b in blocks)
            pair = np.concatenate([res[b]["pair"] for b in blocks])
            size = np.concatenate([res[b]["pair_size"] for b in blocks])
            d = pair[:, 1] - pair[:, 0]
            angle = np.degrees(np.arctan2(np.abs(d.imag), np.abs(d.real))) # 0 = along Re λ, 90 = along Im λ
            sizes = f"{(size == 2).sum()}/{(size == 3).sum()}/{(size >= 4).sum()}"
            if len(d):
                split = f"{np.median(np.abs(d)):>10.3g} {(angle < 15).mean():>6.2f} {(angle > 75).mean():>6.2f}"
            else:
                split = f"{'-':>10} {'-':>6} {'-':>6}"
            print(f"{name:<14} {n:>7} {n_marked:>7} {n * mark_z**2:>8.1f} {n_marked / (n * mark_z**2):>6.2f} "
                  f"{n_gap:>6} {len(pair):>6} {sizes:>12} {split}")

    for label, block_evals, is_ref in spectra:

        if is_ref or not show_pseudo:
            continue

        res = pseudo[label]
        lam, z, s, marked, pair = [np.concatenate([r[key] for r in res.values()])
                                   for key in ("lam", "z", "s", "marked", "pair")]

        n_col = max(3, len(res))
        fig, ax = plt.subplots(2, n_col, figsize=(5 * n_col, 9))
        for a in list(ax[0, len(res):]) + list(ax[1, 3:]):
            a.axis("off")

        # spectrum of every block with the marked eigenvalues
        for a, (block, r) in zip(ax[0], res.items()):
            ev = block_evals[block]
            gap = r["lam"][r["gap"]]
            a.plot(ev.real, ev.imag, ",", color="lightgray")
            a.plot(gap.real, gap.imag, ".", ms=4, color="C2", label=f"gap (|z| < {mark_z:g}, s ≥ {mark_s:g}): {len(gap)}")
            a.plot(r["pair"].real.ravel(), r["pair"].imag.ravel(), ".", ms=3, color="C3",
                   label=f"pair (|z| < {mark_z:g}, s < {mark_s:g}): {len(r['pair'])}")
            a.set_xlabel("Re(λ)")
            a.set_ylabel("Im(λ)")
            a.set_title(f"block {block}", fontsize=9)
            a.legend(fontsize=7, loc="upper left", markerscale=2)

        # cells of the spectrum, # of ratios with |z| < map_z relative to Poisson (log2): blue = fewer (level repulsion),
        # white = as for uncorrelated eigenvalues, red = more (pseudodegeneracies)
        a = ax[1, 0]
        n_all, re_edges, im_edges = np.histogram2d(lam.real, lam.imag, bins=map_bins)
        small = np.abs(z) < map_z
        n_small = np.histogram2d(lam.real[small], lam.imag[small], bins=(re_edges, im_edges))[0]
        expected = n_all * map_z**2
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = np.log2(np.clip(n_small / expected, 2.0**-3, 2.0**3))
        excess[expected < 3] = np.nan # too few eigenvalues for a meaningful ratio
        a.set_facecolor("lightgray")
        mesh = a.pcolormesh(re_edges, im_edges, excess.T, cmap="coolwarm", vmin=-3, vmax=3)
        cbar = fig.colorbar(mesh, ax=a, ticks=range(-3, 4), extend="both")
        cbar.ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"], fontsize=7)
        cbar.set_label(f"# of |z| < {map_z:g} / Poisson", fontsize=8)
        a.set_xlabel("Re(λ)")
        a.set_ylabel("Im(λ)")
        a.set_title(f"ratios |z| < {map_z:g} relative to Poisson (all blocks)\n"
                    "blue = repulsion (chaos), red = clustering, gray = too few eigenvalues", fontsize=9)

        # normalized NN distances, log scales for the small-s tail (in log10 s: Poisson ~ s^2, GinUE ~ s^4),
        # the marked ones with the normalization of all, so that their area is their fraction
        a = ax[1, 1]
        s_edges = np.linspace(-4, 1, 51)
        s_norm = 1 / (len(s) * (s_edges[1] - s_edges[0]))
        x = np.log10(np.clip(s, 10**s_edges[0], 10**s_edges[-1])) # outside the range into the edge bins
        a.hist(x, bins=s_edges, weights=np.full(len(x), s_norm), color="C0", alpha=0.5, label="all eigenvalues")
        a.hist(x[marked], bins=s_edges, weights=np.full(marked.sum(), s_norm), color="C3", alpha=0.7,
               label=f"marked (|z| < {mark_z:g})")
        if compare_refs:
            for i, ref in enumerate(["GinUE", "Poisson"]):
                s_ref = np.concatenate([r["s"] for r in pseudo[ref].values()])
                a.hist(np.log10(np.clip(s_ref, 10**s_edges[0], 10**s_edges[-1])), bins=s_edges, density=True,
                       histtype="step", color=f"C{i + 1}", label=f"{ref} (simulated)")
        # 2D Poisson P(s) = π/2 s exp(-π s^2 / 4) (mean 1) as a density in log10 s
        s_th = 10**np.linspace(s_edges[0], s_edges[-1], 400)
        a.plot(np.log10(s_th), np.log(10) * np.pi / 2 * s_th**2 * np.exp(-np.pi * s_th**2 / 4), "k--", lw=1,
               label="Poisson (theory)")
        a.axvline(np.log10(mark_s), color="gray", ls=":", label=f"s = {mark_s:g}")
        a.set_yscale("log")
        a.set_ylim(1e-4, 5)
        a.set_xlabel("log10 s,  s = NN distance / local mean spacing")
        a.set_ylabel("density")
        a.set_title("how close the nearest neighbour is\n(pairs: s < 1, gaps: s ~ 1 or larger)", fontsize=9)
        a.legend(fontsize=7, loc="lower right")

        # direction of the splitting: along Re λ = the two decay rates differ, along Im λ = the two frequencies
        a = ax[1, 2]
        d = pair[:, 1] - pair[:, 0]
        if len(d):
            a.hist(np.degrees(np.arctan2(np.abs(d.imag), np.abs(d.real))), bins=np.linspace(0, 90, 19), density=True,
                   color="C3", alpha=0.7, label=f"pairs: {len(d)}")
        a.axhline(1 / 90, color="gray", ls="--", label="random direction")
        a.set_xlim(0, 90)
        a.set_xlabel("angle of λ_NN - λ to the real axis [°]")
        a.set_ylabel("density")
        a.set_title("splitting of the pairs\n0° = different decay rate, 90° = different frequency", fontsize=9)
        a.legend(fontsize=7)

        fig.suptitle(f"{label}\nred = the pairs of close eigenvalues behind the central peak of the ratios, "
                     f"green = |z| small only because the next-nearest neighbour is far", fontsize=9)
        plt.tight_layout()

        if show_figs:
            #plt.show()
            print("show degen plot")
        else:
            fig.savefig(os.path.join("figs", fig_dir, f"{label}_pseudodegeneracies.png"))
            plt.close("all")


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
