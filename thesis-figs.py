import numpy as np
import matplotlib.pyplot as plt

def spacing_distributions_2D():

    s = np.linspace(0, 5, 500)

    P_orth = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

    P_unit = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

    P_symp = (2**18 / (3**6 * np.pi**3)) * s**4 * np.exp(-64 * s**2 / (9 * np.pi))

    P_pois = np.exp(-s)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(s, P_orth, label="GOE", lw=2)
    plt.plot(s, P_unit, label="GUE", lw=2)
    plt.plot(s, P_symp, label="GSE", lw=2)
    plt.plot(s, P_pois, '--', label="Poisson", lw=2, color="black")

    plt.xlabel(r"$s$")
    plt.ylabel(r"$P(s)$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
