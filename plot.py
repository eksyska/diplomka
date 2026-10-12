import numpy as np
import matplotlib.pyplot as plt
from qm_statistics import *
from datetime import datetime
from scipy.stats import gaussian_kde



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

def plot_complex_ratios(z, show=True, path=None, real=False, map="color"):
    """Plots complex spacing ratio statistics in the complex plane and both the absolute value |z| and arg(z) dependent histograms

    Args:
        z (complex float array): complex spacing ratios
    """

    if not real:

        fig, ax = plt.subplots(1, 3, figsize=(12, 3))

        if map == "scatter":
            ax[0].scatter(z.real, z.imag, s=1, alpha=0.5)

        elif map == "color":
            points = np.vstack([z.real, z.imag])
            kde = gaussian_kde(points, bw_method=0.15)

            grid_lim = 1.2
            xi, yi = np.mgrid[-grid_lim:grid_lim:300j, -grid_lim:grid_lim:300j]
            zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

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
            kde = gaussian_kde(np.vstack([z_i.real, z_i.imag]), bw_method=0.15)
            xi, yi = np.mgrid[-grid_lim:grid_lim:300j, -grid_lim:grid_lim:300j]
            zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
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


