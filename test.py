import numpy as np
import matplotlib.pyplot as plt
from models import *
from qm_statistics import *
from plot import *

N=2

"""
N=5 
u3 = u3(N)
u3.H_ij_fill()
print(u3.eigvals())

vec = np.zeros(encode(N, N))
vec[0] = 1
u3.state_vec = vec
print(u3.state_vec)
"""

u3 = u3(N)
u3.H_ij_fill()

energies = get_energies(u3.H_ij)

energies = unfolding_polyfit(energies, deg=8)

plot_level_density(energies)
"""

well = well_2d(10000, 1, np.sqrt(np.pi/3))
spectrum = well.compute_spectrum()
spectrum = unfolding_polyfit(spectrum, deg=8)
plot_level_density(spectrum)
"""
