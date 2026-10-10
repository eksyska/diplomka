import numpy as np
import matplotlib.pyplot as plt
import time
from qm_statistics import *
from plot import *
from basis_models import *
from models import *
from models_old import *
from outputs import *
from BoseHubbard import BoseHubbard

from config import *


#print full arrays
#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(linewidth=150)

"""
kappa = 0.2
eta = 0.3
gamma = kappa*eta
gamma2 = kappa*(eta+1)
c_ops_template1 = (1,0.7,1.3,1.5)
"""

################## THE IMPORTANT SETTINGS ##################

L = 3
N = 12

n_cut = N
n_local_max = n_cut

J = 1
U = -10

f = 0.4 #driving
det = 0.8 #detuning

g_loss = 0.5 # loss
g_pump = 0.2 # pumping  /  kappa = g_loss - g_pump
g_deph = 0.0 # dephasing
g_circ_p = 3.5 # directed circulation plus
g_circ_m = 0.5 # directed circulation minus
g_bpl = 0.3 # bond phase locking

driving = (f, det)
gamma = (g_loss, g_pump, g_deph, g_circ_p, g_circ_m, g_bpl)

config=CONFIGS["circulation"] #config encodes symmetries, driving and nullified gammas

############################################################

symmetric_dissipation = True #now always set True

time_start = time.time()

if symmetric_dissipation:

    bh = BoseHubbard(L, N, J, U, config, driving, gamma, n_local_max=n_local_max, n_cut=n_cut,
                    kappa_list=[0], pi_list=[], N_pairs=[(N,N)])

    # does the Fock cutoff actually hold the state these parameters ask for?
    cutoff_report(L, N, J, U, driving, gamma, n_cut=n_cut)

    fock_basis = bh.build_basis(fixed_N=False)
    L_basis = bh.build_sym_basis(fock_basis)

    blocks = bh.build_L_blocks(fock_basis, L_basis)

    block_evals = evals_from_blocks(blocks)
    pooled = pool_evals(block_evals)

    plot_spectrum(pooled)

    all_z = csr_from_evals(block_evals, complex_spacing_ratios)
    plot_complex_ratios(all_z, show=True, map="scatter") 
    
    
    # test code for comparing evals of full Lindbladian and evals by sectors
    """
    L_full = bose_hubbard_L_full(L, N, J, U, driving, gamma, config, n_cut, n_local_max=n_local_max)
    evals_full = L_full.L_op.eigenenergies()
    print("arrays equal:", compare_complex(clean_num_error(pooled), clean_num_error(evals_full)))
    """


time_end = time.time()

print(f"time = {(time_end-time_start):.3f} s")