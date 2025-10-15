import numpy as np
import matplotlib.pyplot as plt
from models import *
from qm_statistics import *
from plot import *
from models import *

L = 3
N = 3
J = 1.0
U = 1.0
gamma = 0.5

test_lindbladian = True
test_hamiltonian = False

if test_lindbladian:

    L_op = bose_hubbard_lindbladian(L, N, J, U, gamma)
    print(L_op.shape)

    evals, evecs = L_op.eigenstates()

    s_norm_g, rho_g, keep_g, s_raw_g = normalized_nn_spacings(evals, kde_bw=0.2, remove_edge_frac=0.12)
    plot_spacing_hist(s_norm_g, title='Ginibre sample normalized NN spacings (N=200)')

    #z = complex_spacing_ratios(evals)
    #plot_complex_ratios(z)

if test_hamiltonian:

    H_op = bose_hubbard_hamiltonian(L, N, J, U)
    print(H_op.shape)

    eigvals, evecs = H_op.eigenstates()

    spacings = np.diff(np.sort(eigvals))
    spacings = spacings[spacings > 0]
    spacings /= np.mean(spacings)  # makes ⟨s⟩ = 1

    plot_level_density(spacings, func=poisson, range=(0,5))

