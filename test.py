import numpy as np
import matplotlib.pyplot as plt
import time
from qm_statistics import *
from plot import *
from basis_models import *
from models import *
from outputs import *
from math_funcs import _ALL

#print full arrays
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=150)

L = 3
N = 2
n_local_max = 3
J = -0.2
U = 1.0
gamma1 = 0.1
gamma2 = 0.05

"""
kappa = 0.2
eta = 0.3
gamma = kappa*eta
gamma2 = kappa*(eta+1)
"""

symmetric_dissipation = True
dissipation = "PUMPLOSS" # DEPHASING / LOSS / PUMPLOSS

if dissipation=="PUMPLOSS":
    gamma = (gamma1, gamma2)
else:
    gamma = (gamma1,)

c_ops_template1 = (1,0.7,1.3,1.5)
c_ops_template2 = (1,0,0,0,0)
c_ops_template_sym = (1,1,1,1,1,1)

M=0 # used for filename only now
filename = f"L{L}_N{N}_J{J}_U{U}_gamma{gamma}_{dissipation}_M={M}"
subfolder = "dephasing"

test_lindbladian = True
test_hamiltonian = False

time_start = time.time()


if test_lindbladian:

    if symmetric_dissipation:
        
        n_pairs = [(N,N)]
        
        blocks = bose_hubbard_L_blocks(L, N, J, U, gamma, dissipation, c_ops_template_sym, is_symmetric=True,
                                        kappa_list=[0], M_list=[1])
        """blocks = bose_hubbard_L_blocks(L, N, J, U, gamma, dissipation, c_ops_template_sym, is_symmetric=True,
                                        k_L_list=[0], k_R_list=[0], p_L_list=[1], p_R_list=[1], M_list=[0])"""

        for key, val in find_blocks(blocks).items():
            print(f"{key}, {val["n_blocks"]}")

        block_evals = evals_from_blocks(blocks)
        pooled = pool_evals(block_evals)
        #plot_spectrum(pooled)

        all_z = csr_from_evals(block_evals, complex_spacing_ratios)
        #plot_complex_ratios(all_z, show=True) 
       


        #all_z = csr_from_evals(block_evals, complex_spacing_ratios)
        
        #plot_complex_ratios(all_z, show=True) 

    if not symmetric_dissipation:

        blocks = bose_hubbard_L_blocks(L, N, J, U, gamma, dissipation, c_ops_template2, n_local_max=n_local_max, is_symmetric=False,
                                        M_list=[-1,1])
        
        block_evals = evals_from_blocks(blocks)

        all_z = csr_from_evals(block_evals, complex_spacing_ratios)
        plot_complex_ratios(all_z, show=True)
        

if test_hamiltonian:

    H_op = bose_hubbard_hamiltonian(L, N, J, U, restrict_symmetry=True)
    evals = H_op.eigenenergies()

    #test complex spacing ratios on closed system hamiltonian (works)
    z, r = complex_spacing_ratios(evals)
    r_mean = r.mean()
    print(f"⟨r⟩ = {r_mean:.4f}  (GOE: 0.536, Poisson: 0.386)")
    plot_complex_ratios(z, filename=filename, real=True)


#check of consistency between hamiltonian and lindbladian (for no dissipacy) code:
"""
im_evals = np.load("im_evals.npy")
E_diff = np.load("E_diffs.npy")

def contained(value, array):
    return np.any(np.abs(array - value) < 1e8)

missing = []
for val in E_diff:
    if not contained(val, im_evals):
        missing.append(val)

print("Number missing:", len(missing))

print(len(im_evals), len(E_diff))
"""

time_end = time.time()

print(f"time = {(time_end-time_start):.3f} s")