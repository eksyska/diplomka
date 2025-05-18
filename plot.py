import numpy as np
import matplotlib.pyplot as plt

def plot_level_density (spectrum, func=None, bins=100):

    spacings = np.diff(spectrum)
    print(spacings)

    plt.hist(spacings, bins=bins, density=True, range=(0,5))

    if func != None:
        x = np.linspace(spectrum[0], spectrum[-1], 100)
        plt.plot(x, func(x))

    plt.show()