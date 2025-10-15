import numpy as np
import matplotlib.pyplot as plt

def plot_level_density (spectrum, func=None, bins=50, range=(0,3), exclude_zeros=False):

    spacings = np.diff(spectrum)
    spacings = spacings[spacings != 0]

    plt.hist(spacings, bins=bins, density=True, range=range)

    if func != None:
        x = np.linspace(range[0], range[1], 100)
        plt.plot(x, func(x))

    plt.show()

def plot_complex_ratios(z):

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    ax[0].scatter(z.real, z.imag, s=5, alpha=0.5)
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


def plot_spacing_hist(s_norm, bins=50, title='Normalized NN spacings'):
    s = np.asarray(s_norm)
    plt.figure(figsize=(6,4))
    counts, edges, _ = plt.hist(s, bins=bins, density=True, alpha=0.6, label='data')
    xs = np.linspace(0, np.percentile(s, 99), 400)
    # 2D Poisson normalized pdf approx:
    pdf_poisson = (np.pi/2) * xs * np.exp(- (np.pi/4) * xs**2)
    plt.plot(xs, pdf_poisson, lw=2, label='2D Poisson (uncorrelated)')
    plt.xlabel('normalized spacing s')
    plt.ylabel('pdf')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()