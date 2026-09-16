import numpy as np
import matplotlib.pyplot as plt
import time
from qm_statistics import *
from plot import *
from basis_models import *
from models import *
from models_old import *
from outputs import *

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
N = 3
n_local_max = N
J = -0.2
U = 1.0
f = 1
det = 1
gamma1 = 0.1
gamma2 = 0.05

############################################################

symmetric_dissipation = True
dissipation = "PUMPLOSS" # DEPHASING / LOSS / PUMPLOSS

if dissipation=="PUMPLOSS":
    gamma = (gamma1, gamma2)
else:
    gamma = (gamma1,)

filename = f"L{L}_N{N}_J{J}_U{U}_{dissipation}"
subfolder = f"{dissipation}"

time_start = time.time()

if symmetric_dissipation:

    bh = BoseHubbard(L, N, J, U, f, det, dissipation, gamma, n_local_max=n_local_max, M_list=[], kappa_list=[], pi_list=[])

    fock_basis = build_bose_basis(bh.L, bh.N, fixed_N=False)
    L_basis = build_sym_L_basis(fock_basis)

    blocks = bh.build_L_blocks(fock_basis, L_basis)

    block_evals = evals_from_blocks(blocks)
    pooled = pool_evals(block_evals)

    #all_z = csr_from_evals(block_evals, complex_spacing_ratios)
    #plot_complex_ratios(all_z, show=True) 

    # test code for comparing evals of full Lindbladian and evals by sectors
    
    
    L_full = bose_hubbard_L_full(L, N, J, U, f, det, gamma, "PUMPLOSS", (1,1,1), is_symmetric=False)
    evals_full = L_full.L_op.eigenenergies()
    
    print("arrays equal:", compare_complex(clean_num_error(pooled), clean_num_error(evals_full)))
    


time_end = time.time()

print(f"time = {(time_end-time_start):.3f} s")