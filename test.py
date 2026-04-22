import numpy as np
import matplotlib.pyplot as plt
import time
from qm_statistics import *
from plot import *
from basis_models import *
from models import *
from outputs import *

#print full arrays
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=150)

L = 3
N = 10
n_local_max = N
J = -0.1
U = 1.0
gamma1 = 0.15
gamma2 = 0.1

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
c_ops_template_sym = (1,1,1,1,1,1)

M=0
filename = f"L{L}_N{N}_J{J}_U{U}_gamma{gamma}_{dissipation}_M={M}"
subfolder = "dephasing"

test_lindbladian = True
test_hamiltonian = False

time_start = time.time()


if test_lindbladian:

    if symmetric_dissipation:
        
        

        
        blocks = bose_hubbard_L_blocks(L, N, J, U, gamma, dissipation, c_ops_template_sym, is_symmetric=True,
                                        kappa_list=[0], M_list=[1,-1])
        
        block_evals = evals_from_blocks(blocks)

        all_z = csr_from_evals(block_evals, complex_spacing_ratios)
        plot_complex_ratios(all_z, show=True) 

        #some testing code
        """
        basis_fock= build_bose_basis(L ,N, fixed_N=False, n_local_max=n_local_max)
        basis_sym = build_sym_basis(basis_fock)
        all_n = [s.n for s in basis_sym]
        blocks = bose_hubbard_L_blocks(L, N, J, U, gamma, dissipation, c_ops_template_sym, is_symmetric=True)
        block_evals = evals_from_blocks(blocks)
        l1 = bose_hubbard_L_full(L, N, J, U, gamma, dissipation, c_ops_template_sym, is_symmetric=True).L_op
        l2 = bose_hubbard_L_full(L, N, J, U, gamma, dissipation, c_ops_template_sym, is_symmetric=False).L_op

        evals1 = np.sort_complex(clean_num_error(l1.eigenenergies()))
        evals2 = np.sort_complex(clean_num_error(l2.eigenenergies()))
        block_evals_pooled = np.sort_complex(clean_num_error(pool_evals(block_evals)))

        print(evals1)
        print(len(evals2))
        print(len(block_evals_pooled))
        for label, value in block_evals.items():
            print(f"{label}: {value}")

        print(compare_complex(evals1,evals2))
        print(compare_complex(evals1,block_evals_pooled))
        print(compare_complex(block_evals_pooled,evals2))
        """
        
        """
        lind = bose_hubbard_L_full(L, N, J, U, gamma, dissipation, c_ops_template_sym, is_symmetric=True)
        L_op = lind.L_op
        evals = clean_num_error(L_op.eigenenergies())
        evals = np.sort_complex(evals)

        print(evals)
        print(block_evals_pooled)
        print(len(evals), len(block_evals_pooled))

        print(f"allclose: {np.allclose(evals, block_evals_pooled)}")"""

        """
        idx = get_block_indices(basis_sym, all_n, k_L=0, k_R=0, p_L=1, p_R=1, M=0)
        idx = np.concatenate([idx, [10,15]])
        print("indices:", idx)
        for alpha in idx:
            i = alpha % len(basis_sym)
            j = alpha // len(basis_sym)
            print(f"  alpha={alpha}: |{basis_sym[i]}><{basis_sym[j]}|")

        print("block:\n", blocks[(0,0,1,1,0)])

        
        # find the 2x2 block label
        label = (0, 0, 1, 1, 0)

        # block from direct construction
        block_direct = blocks[label]
        print("direct:\n", block_direct)

        dim = len(basis_sym)
        a_list = [build_a_i_sym(i, basis_sym) for i in range(L)]
        H, c_ops = build_H_and_cops(a_list, L, N, J, U, gamma, dissipation, c_ops_template_sym, len(basis_sym))
        H_op = H.full()
        c_ops_np = [c.full() for c in c_ops]

        # use the QuTiP Liouvillian, not the directly built one
        L_qutip = qt.liouvillian(H, c_ops).full()
        idx = get_block_indices(basis_sym, all_n, k_L=0, k_R=0, p_L=1, p_R=1, M=0)

        block_qutip = L_qutip[np.ix_(idx, idx)]
        print("qutip block:\n", block_qutip)
        print("qutip block evals:", np.linalg.eigvals(block_qutip))"""
        """
        # block from slicing the full matrix
        all_indices = np.arange(len(basis_sym)**2)
        L_full = build_L_block_direct(H_op, c_ops_np, all_indices)
        idx = get_block_indices(basis_sym, all_n, k_L=0, k_R=0, p_L=1, p_R=1, M=0)
        block_sliced = L_full[np.ix_(idx, idx)]
        print("sliced:\n", block_sliced)

        print(qt.Qobj(block_direct).eigenenergies())
        print("match:", np.allclose(block_direct, block_sliced))

        print(len(pool_evals(block_evals)), len(evals))"""


        #all_z = csr_from_evals(block_evals, complex_spacing_ratios)
        
        #plot_complex_ratios(all_z, show=True) 

    if not symmetric_dissipation:

        print("test")
        

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