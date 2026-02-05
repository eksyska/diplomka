import numpy as np
import matplotlib.pyplot as plt
import winsound
import time
from models import *
from qm_statistics import *
from plot import *
from models import *
from miscelaneous import *


L = 8
N = 10
J = 0.5
U = 1.0
gamma = 0.2
k = 1 #translation subspace - momentum sector
c_ops_template1 = [1,np.sqrt(3),np.sqrt(5)]
c_ops_template2 = [1,2,3,2,3]

# L=3, N=4, J=-0.5, gamma=0.5
# L=3, N=4, J=-0.5, gamma=0.2-0.3
# L=3, N=5, J=-0.5, gamma=0.1, c_ops_template2 = [1,1,3]
# L=3, N=4, J=-0.5, gamma=0.2-0.3, c_ops_template2 = [1,1,10] -> zajímavé pruhy eigenvalues
#outdated N=2, gamma = 0.5, J = 0.1 - 0.4 and -0.6 - -0.2

test_lindbladian = False
test_hamiltonian = True

time_start = time.time()

"""
evals = np.cumsum(np.random.exponential(1, 2000))
spacings = level_spacings(evals, unfolding=True)
plot_level_density(spacings, funcs=[poisson, wigner], range=(0,5))
"""

if test_lindbladian:

    L_op = bose_hubbard_lindbladian(L, N, J, U, gamma, c_ops_template1, restrict_symmetry=False, k=k)
    evals = L_op.eigenenergies()

    #winsound.Beep(1000, 500)

    plot_eigenvalues(evals)

    spacings = normalized_nn_spacings(evals, remove_edge_frac=0.1)
    #plot_NN_spacings(spacings, funcs=[poisson2d])

    z = complex_spacing_ratios(evals)
    plot_complex_ratios(z)

if test_hamiltonian:

    H_op = bose_hubbard_hamiltonian(L, N, J, U, restrict_symmetry=True, k=k)
    evals = H_op.eigenenergies()
    print(evals)
    evals = evals[evals<-4]
    print(evals)
    print(r_statistic(evals))

    spacings = level_spacings(evals, unfolding=True, degree=4)
    print(r_statistic(evals))

    plot_level_density(spacings, funcs=[poisson, wigner], range=(0,5))

time_end = time.time()

print("time: " + str(time_end - time_start))