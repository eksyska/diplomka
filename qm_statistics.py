import numpy as np
import matplotlib.pyplot as plt


def get_energies(H):

    if H.all() == H.conj().T.all(): #is hermitian
        eigvals = np.linalg.eigvalsh(H)
    else: #is not hermitian
        eigvals = np.linalg.eigvals(H)
    eigvals = np.sort(eigvals)
    return eigvals

def unfolding_polyfit (spectrum, deg=6):
    
    spectrum = np.sort(spectrum)
    smooth_fit = np.polynomial.Polynomial.fit(spectrum, range(len(spectrum)), deg)
    
    plt.plot(spectrum, range(len(spectrum)))
    plt.plot(*smooth_fit.linspace())
    plt.show()
    

    unfolded = smooth_fit(spectrum)
    return unfolded
