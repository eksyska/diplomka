import numpy as np
import matplotlib.pyplot as plt
from qm_statistics import *
from datetime import datetime
from scipy.stats import gaussian_kde
from scipy import ndimage
from matplotlib.widgets import Slider



def plot_level_density (spacings, funcs=[], bins=50, range=(0,3)):
    """Plots level density (spacings histogram)

    Args:
        spacings (float array): spectrum spacings
        funcs (list, optional): reference functions to plot. Defaults to [].
        bins (int, optional): # of histogram bins. Defaults to 50.
        range (tuple, optional): histogram range. Defaults to (0,3).
    """

    plt.hist(spacings, bins=bins, density=True, range=range)

    if len(funcs) > 0:
        x = np.linspace(range[0], range[1], 100)
        for f in funcs:
            plt.plot(x, f(x))

    plt.show()


def plot_complex_ratios(z, show=True, path=None, real=False, map="color", title=None):
    """Plots complex spacing ratio statistics in the complex plane and both the absolute value |z| and arg(z) dependent histograms

    Args:
        z (complex float array): complex spacing ratios
        title (str, optional): figure title. Defaults to None (no title)
    """

    if not real:

        fig, ax = plt.subplots(1, 3, figsize=(12, 3))

        if map == "scatter":
            ax[0].scatter(z.real, z.imag, s=1, alpha=0.5)

        elif map == "color":
            xi, yi, zi = kde_grid(z)

            im = ax[0].pcolormesh(xi, yi, zi, cmap='coolwarm', shading='gouraud')
            plt.colorbar(im, ax=ax[0], label='Density')

        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
        ax[0].add_artist(circle)
        ax[0].set_aspect('equal')
        ax[0].set_title("Complex spacing ratios")

        ax[1].hist(np.abs(z), bins=50, density=True, color='C1', alpha=0.7)
        ax[1].set_xlabel("|z|")
        ax[1].set_ylabel("P(|z|)")
        ax[1].set_title("Histogram of |z|")

        ax[2].hist(np.angle(z), bins=50, density=True, color='C2', alpha=0.7)
        ax[2].set_xlabel("arg(z)")
        ax[2].set_ylabel("P(arg z)")
        ax[2].set_title("Histogram of arg(z)")

    if real:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        ax.hist(z.real, bins=50, density=True, color='C2', alpha=0.7)
        ax.set_xlabel("r")
        ax.set_ylabel("P(r)")
        ax.set_title("Histogram of arg(z)")

    if title is not None:
        fig.suptitle(title, fontsize=9)

    plt.tight_layout()

    if show:
        plt.show()

    elif not show:

        filename=path[0]
        if len(path) > 0:
            subfolder = path[1] + "/"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename != None:
            
            plt.savefig(f"figs/{subfolder}{filename}_{timestamp}.png")
        else:
            plt.savefig(f"figs/{subfolder}{timestamp}.png")


def plot_complex_ratios_slices(z, slices, show=True, bins=40, map="scatter", show_total=True):
    """Plots complex spacing ratios of each eigenvalue slice in its own row.
    The complex plane plot and the |z| and arg(z) histograms. The histograms of the total are drawn in gray for comparison

    Args:
        z (complex float array): all complex spacing ratios
        slices (list): slice dicts
        bins (int, optional): # of histogram bins. Defaults to 40.
        map (str, optional): "scatter" or "color" (KDE density) for the complex plane plot. Defaults to "scatter".
        show_total (bool, optional): add a last row with all ratios together. Defaults to True.
    """

    rows = [(f"slice {i+1}: [{s['range'][0]:.2f}, {s['range'][1]:.2f}]", s["z"]) for i, s in enumerate(slices)]
    if show_total:
        rows.append(("total", z))

    fig, ax = plt.subplots(len(rows), 3, figsize=(12, 3*len(rows)), sharex="col", sharey="col", squeeze=False)

    grid_lim = 1.2

    for a, (label, z_i) in zip(ax, rows):

        if map == "scatter":
            a[0].scatter(z_i.real, z_i.imag, s=1, alpha=0.5)

        elif map == "color" and len(z_i) > 2:
            xi, yi, zi = kde_grid(z_i, grid_lim)
            a[0].pcolormesh(xi, yi, zi, cmap='coolwarm', shading='gouraud')

        a[0].add_artist(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--'))
        a[0].set_xlim(-grid_lim, grid_lim)
        a[0].set_ylim(-grid_lim, grid_lim)
        a[0].set_aspect('equal')
        a[0].set_ylabel(label)

        if len(z_i):
            a[0].set_title(f"n = {len(z_i)}, <|r|> = {np.abs(z_i).mean():.3f}, "
                           f"-<cos θ> = {-np.cos(np.angle(z_i)).mean():.3f}", fontsize=9)

        if show_total and z_i is not z:
            a[1].hist(np.abs(z), bins=bins, range=(0, 1), density=True, histtype="step", color="gray", label="total", zorder=3)
            a[2].hist(np.angle(z), bins=bins, range=(-np.pi, np.pi), density=True, histtype="step", color="gray", label="total", zorder=3)

        a[1].hist(np.abs(z_i), bins=bins, range=(0, 1), density=True, color='C1', alpha=0.7)
        a[2].hist(np.angle(z_i), bins=bins, range=(-np.pi, np.pi), density=True, color='C2', alpha=0.7)
        a[1].set_ylabel("P(|z|)")
        a[2].set_ylabel("P(arg z)")

    ax[0, 1].set_title("Histogram of |z|")
    ax[0, 2].set_title("Histogram of arg(z)")
    if show_total and len(rows) > 1:
        ax[0, 2].legend(fontsize=7)

    ax[-1, 0].set_xlabel("Re z")
    ax[-1, 1].set_xlabel("|z|")
    ax[-1, 2].set_xlabel("arg(z)")

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_csr_window(window, show=True, others=None, normalize=False, title=None):
    """Plots <|r|> and -<cos θ> of the sliding window statistics against the window center,
    with horizontal reference lines for GinUE (chaos) and Poisson (regular)

    Args:
        window (dict): window dict from window_csr / csr_from_evals(..., width=...)
        others (list, optional): (window dict, label) of further curves drawn into the same axes,
            the curve of window is then labelled "data". Defaults to None
        normalize (bool, optional): window center divided by the largest eigenvalue size of each spectrum,
            for spectra of different scales. Defaults to False
        title (str, optional): line above the default title. Defaults to None

    For slice_by="rank" a second x axis on top shows the |λ| of window (the data) belonging to each rank.
    """

    xlabel = {"abs": "|λ|", "real": "|Re λ|", "rank": "rank of |λ|"}[window["slice_by"]]
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    curves = [(window, "data" if others else None)] + list(others or [])

    for a, key, ylabel, ref_idx, c in [(ax[0], "mean_abs", "<|r|>", 0, "C0"), (ax[1], "mean_cos", "-<cos θ>", 1, "C1")]:

        for i, (w, label) in enumerate(curves):
            x = w["center"] / w["key_max"] if normalize else w["center"]
            # a single curve keeps its color per panel, several curves get one color each
            a.plot(x, w[key], ".-", color=c if len(curves) == 1 else f"C{i}", label=label)

        for (name, ref), ls in zip(CSR_REF.items(), ["--", ":"]):
            a.axhline(ref[ref_idx], color="gray", ls=ls, lw=1, label=name)

        a.set_ylabel(ylabel)
        a.legend(fontsize=8)

    ax[0].set_title(("" if title is None else title + "\n") +
                    f"Sliding window statistics (width {window['width']:g}, step {window['step']:g})",
                    fontsize=None if title is None else 9)
    ax[1].set_xlabel(f"window center {xlabel} / max {xlabel}" if normalize else f"window center {xlabel}")

    if window["slice_by"] == "rank":
        # rank -> |λ| of the data (the rank order is the |λ| order), the curves of the other spectra are read off at the same rank
        a_sorted = np.abs(window["evals"])
        q = np.linspace(0, 1, len(a_sorted))
        top = ax[0].secondary_xaxis("top", functions=(lambda r: np.interp(r, q, a_sorted),
                                                      lambda v: np.interp(v, a_sorted, q)))
        top.set_xlabel("|λ| of data" if others else "|λ|", fontsize=8)

        # ticks at evenly spaced ranks, ticks evenly spaced in |λ| would pile up in the sparse tail
        ticks = np.interp(np.linspace(0, 1, 6), q, a_sorted)
        ticks[ticks < 1e-9 * a_sorted[-1]] = 0 # steady state, |λ| ~ 1e-13 from rounding
        top.set_xticks(ticks)
        top.set_xticklabels([f"{t:.3g}" for t in ticks], fontsize=8)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_csr_window_slider(window, others=None, title=None, bins=40, map="color", show=True):
    """Interactive sliding window statistics: the window curves on top, below them for the window at the slider
    position the complex spacing ratios in the complex plane, the |z| and arg(z) histograms and the eigenvalues
    of the window highlighted in the spectrum

    The slider only works in an interactive backend (script run from a terminal), not in saved figures.

    Args:
        window (dict): window dict of the data from window_csr / csr_from_evals(..., width=...)
        others (list, optional): (window dict, label) of further spectra (e.g. GinUE, Poisson), their curves and
            histograms at the same position are drawn for comparison. The position is the same rank for
            slice_by="rank", otherwise it is scaled by the largest eigenvalue size of each spectrum. Defaults to None
        title (str, optional): figure title. Defaults to None
        bins (int, optional): # of histogram bins. Defaults to 40
        map (str, optional): "color" (KDE density) or "scatter" for the complex plane plot. Defaults to "color"
        show (bool, optional): Defaults to True

    Returns:
        matplotlib.figure.Figure: the slider is kept as fig.slider, it stops responding once it is garbage collected
    """

    curves = [(window, "data" if others else None)] + list(others or [])
    scale = [window["key_max"] / w["key_max"] for w, _ in curves] # other spectra on the x axis of the data
    xlabel = {"abs": "|λ|", "real": "|Re λ|", "rank": "rank of |λ|"}[window["slice_by"]]

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.3])
    ax_r, ax_c = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])
    ax_z, ax_abs, ax_arg, ax_sp = [fig.add_subplot(gs[1, i]) for i in range(4)]

    for a, key, ylabel, ref_idx in [(ax_r, "mean_abs", "<|r|>", 0), (ax_c, "mean_cos", "-<cos θ>", 1)]:

        for i, ((w, label), s) in enumerate(zip(curves, scale)):
            a.plot(w["center"] * s, w[key], ".-", ms=2, lw=0.8, color=f"C{i}", label=label)

        for (name, ref), ls in zip(CSR_REF.items(), ["--", ":"]):
            a.axhline(ref[ref_idx], color="gray", ls=ls, lw=1, label=name)

        a.set_xlabel(f"window center {xlabel}")
        a.set_ylabel(ylabel)
        a.legend(fontsize=7)

    # all artists below are created once, moving the slider only replaces their data (redrawing them is slow)

    # the whole spectrum once in gray, the eigenvalues of the window are drawn over it
    ev = window["evals"]
    ax_sp.plot(ev.real, ev.imag, ",", color="lightgray")
    highlight, = ax_sp.plot([], [], ".", ms=2, color="C0")
    ax_sp.set_xlabel("Re(λ)")
    ax_sp.set_ylabel("Im(λ)")
    ax_sp.set_title("eigenvalues of the window", fontsize=9)

    # complex plane of the data; imshow with bilinear interpolation looks like pcolormesh(shading="gouraud")
    # in plot_complex_ratios but draws several times faster
    lim, n_grid = 1.2, 300 # as in kde_grid
    half = lim / (n_grid - 1)
    if map == "color":
        density = ax_z.imshow(np.zeros((n_grid, n_grid)), origin="lower", cmap="coolwarm", interpolation="bilinear",
                              extent=(-lim - half, lim + half, -lim - half, lim + half))
    else:
        density, = ax_z.plot([], [], ".", ms=1, alpha=0.5)
    ax_z.add_artist(plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--"))
    ax_z.set_xlim(-lim, lim)
    ax_z.set_ylim(-lim, lim)
    ax_z.set_aspect("equal")
    ax_z.set_title("complex spacing ratios" + (" (data)" if others else ""), fontsize=9)

    # histograms, the data filled and the other spectra as outlines
    edges_abs, edges_arg = np.linspace(0, 1, bins + 1), np.linspace(-np.pi, np.pi, bins + 1)
    hists = []
    for i, (_, label) in enumerate(curves):
        style = dict(fill=True, alpha=0.6) if i == 0 else dict(lw=1.5)
        hists.append((ax_abs.stairs(np.zeros(bins), edges_abs, color=f"C{i}", label=label, **style),
                      ax_arg.stairs(np.zeros(bins), edges_arg, color=f"C{i}", label=label, **style)))
    ax_abs.set_xlabel("|z|")
    ax_abs.set_ylabel("P(|z|)")
    ax_arg.set_xlabel("arg(z)")
    ax_arg.set_ylabel("P(arg z)")
    if others:
        ax_abs.legend(fontsize=7)

    bands = []

    def update(pos):

        # ratios of every spectrum in the window around pos, keys are sorted -> one slice per spectrum
        sel = []
        for (w, _), s in zip(curves, scale):
            lo = np.searchsorted(w["key"] * s, pos - s * w["width"] / 2, side="left")
            hi = np.searchsorted(w["key"] * s, pos + s * w["width"] / 2, side="right")
            sel.append((w["z"][lo:hi], lo, hi))

        z, lo, hi = sel[0]

        if map != "color":
            density.set_data(z.real, z.imag)
        elif len(z) > 2:
            zi = kde_grid(z, lim, n_grid)[2]
            density.set_data(zi.T) # kde_grid is indexed (x, y), imshow (row = y, column = x)
            density.set_clim(0, zi.max())

        top_abs, top_arg = 0, 0
        for (z_i, _, _), (h_abs, h_arg) in zip(sel, hists):
            if len(z_i):
                p_abs = np.histogram(np.abs(z_i), bins=edges_abs, density=True)[0]
                p_arg = np.histogram(np.angle(z_i), bins=edges_arg, density=True)[0]
                h_abs.set_data(p_abs)
                h_arg.set_data(p_arg)
                top_abs, top_arg = max(top_abs, p_abs.max()), max(top_arg, p_arg.max())
        ax_abs.set_ylim(0, 1.1 * top_abs or 1)
        ax_arg.set_ylim(0, 1.1 * top_arg or 1)

        highlight.set_data(ev.real[lo:hi], ev.imag[lo:hi])

        for b in bands:
            b.remove()
        bands[:] = [a.axvspan(pos - window["width"] / 2, pos + window["width"] / 2, color="gray", alpha=0.3)
                    for a in (ax_r, ax_c)]

        info = f"window {xlabel} in [{pos - window['width'] / 2:.3g}, {pos + window['width'] / 2:.3g}]"
        if window["slice_by"] == "rank" and len(z):
            info += f", |λ| in [{abs(ev[lo]):.3g}, {abs(ev[hi - 1]):.3g}]"
        if len(z):
            info += f":  n = {len(z)},  <|r|> = {np.abs(z).mean():.3f},  -<cos θ> = {-np.cos(np.angle(z)).mean():.3f}"
        fig.suptitle(info if title is None else f"{title}\n{info}", fontsize=9)

        fig.canvas.draw_idle()

    # space for the title on top and the slider at the bottom
    plt.tight_layout(rect=(0, 0.05, 1, 0.94))

    slider = Slider(fig.add_axes([0.1, 0.015, 0.8, 0.025]), "window center", window["key"][0], window["key"][-1],
                    valinit=np.median(window["key"]), valstep=window["step"])
    slider.on_changed(update)
    update(slider.val)
    fig.slider = slider

    if show:
        plt.show()

    return fig


def plot_NN_spacings(s_norm, funcs=[], bins=50, range=(0,3)):
    """Plots normalized nearest neighbour spacings for complex eigenvalues

    Args:
        s_norm (complex float array): normalized spacings
        bins (int, optional): # of bins. Defaults to 50.
    """
    s = np.asarray(s_norm)

    plt.figure(figsize=(6,4))
    counts, edges, _ = plt.hist(s, bins=bins, density=True, alpha=0.6, label='data')
    xs = np.linspace(0, np.percentile(s, 99), 400)

    if len(funcs) > 0:
        x = np.linspace(range[0], range[1], 100)
        for f in funcs:
            plt.plot(x, f(x))

    plt.xlabel('normalized spacing s')
    plt.ylabel('pdf')
    plt.legend()
    plt.title("Normalized NN spacings")
    plt.tight_layout()
    plt.show()


def plot_spectrum(evals):
    """Plots eigenvalues in the complex plane

    Args:
        evals (complex float array): eigenvalues
    """

    evals = np.asarray(evals)

    plt.figure(figsize=(7,5))

    plt.scatter(evals.real, evals.imag, s=4, alpha=0.7)

    plt.axhline(0, color='black', alpha=0.5, linewidth=0.8)
    plt.axvline(0, color='black', alpha=0.5, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Spectrum in the complex plane")
    
    plt.tight_layout()
    plt.show()


def kde_grid(z, grid_lim=1.2, n_grid=300, bw=0.15):
    """Fast (binned) gaussian KDE of complex points on a square grid

    Equivalent to gaussian_kde(..., bw_method=bw) evaluated on the grid, but instead of summing the kernels of all points at every grid point, the points are binned into a 2D histogram on the grid which is then smoothed by a gaussian filter of the same width

    Args:
        z (complex np.ndarray): points in the complex plane
        grid_lim (float, optional): grid spans [-grid_lim, grid_lim] in both directions. Defaults to 1.2.
        n_grid (int, optional): # of grid points along each axis. Defaults to 300.
        bw (float, optional): bandwidth factor as in gaussian_kde bw_method. Defaults to 0.15.

    Returns:
        tuple: xi, yi grid coordinates and zi density (as from np.mgrid)
    """

    centers = np.linspace(-grid_lim, grid_lim, n_grid)
    dx = centers[1] - centers[0]
    edges = np.append(centers - dx/2, centers[-1] + dx/2)

    hist, _, _ = np.histogram2d(z.real, z.imag, bins=[edges, edges])
    sigma = bw * np.array([z.real.std(ddof=1), z.imag.std(ddof=1)]) / dx
    zi = ndimage.gaussian_filter(hist, sigma, mode="constant") / (len(z) * dx**2)

    xi, yi = np.meshgrid(centers, centers, indexing="ij")
    
    return xi, yi, zi