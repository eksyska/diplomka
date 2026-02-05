import numpy as np
import matplotlib.pyplot as plt
from qm_statistics import *



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

def plot_complex_ratios(z):
    """Plots complex spacing ratio statistics in the complex plane and both the absolute value |z| and arg(z) dependent histograms

    Args:
        z (complex float array): complex spacing ratios
    """

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    ax[0].scatter(z.real, z.imag, s=1, alpha=0.5)
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

    plt.tight_layout()
    plt.show()


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


def plot_eigenvalues(evals):
    """Plots eigenvalues in the complex plane

    Args:
        evals (complex float array): eigenvalues
    """

    evals = np.asarray(evals)

    plt.figure()

    plt.scatter(evals.real, evals.imag, s=4, alpha=0.7)

    plt.axhline(0, color='black', alpha=0.5, linewidth=0.8)
    plt.axvline(0, color='black', alpha=0.5, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Spectrum in the complex plane")
    
    plt.tight_layout()
    plt.show()
