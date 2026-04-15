import numpy as np
import matplotlib.pyplot as plt
import time
from qm_statistics import *
from plot import *
from models import *
from outputs import *

#np.set_printoptions(threshold=np.inf)

L = 4
N = 2
n_local_max = 5
J = -0.2
U = 1.0
gamma1 = 0.2
gamma2 = 0.05

"""
kappa = 0.2
eta = 0.3
gamma = kappa*eta
gamma2 = kappa*(eta+1)
"""
M=None


symmetric_dissipation = True
dissipation = "PUMPLOSS" # DEPHASING / LOSS / PUMPLOSS

if dissipation=="PUMPLOSS":
    gamma = (gamma1, gamma2)
else:
    gamma = (gamma1,)

c_ops_template1 = (1,0.7,1.3,1.5)
c_ops_template_sym = (1,1,1,1)

filename = f"L{L}_N{N}_J{J}_U{U}_gamma{gamma}_{dissipation}_M={M}"
subfolder = "dephasing"

# L=3, N=4, J=-0.5, gamma=0.2-0.3, c_ops_template2 = [1,1,10] -> zajímavé pruhy eigenvalues

test_lindbladian = True
test_hamiltonian = False

time_start = time.time()

def symmetric_routine(L, N, J, U, gamma, dissipation, c_ops_template, n_local_max=n_local_max, restrict_symmetry=True):

    L_red = bose_hubbard_lindbladian(L, N, J, U, gamma, dissipation, c_ops_template, n_local_max=n_local_max, restrict_symmetry=restrict_symmetry)

    all_z = []
    for k_val, L_red_k in enumerate(L_red):
        evals = L_red_k.eigenenergies()
        #plot_spectrum(evals)
        z, r = complex_spacing_ratios(evals)
        all_z.append(z)
        #print(f"sector {k_val}: ⟨r⟩={r.mean():.4f}")

    z_pooled = np.concatenate(all_z)
    print(f"Pooled ⟨r⟩ = {np.abs(z_pooled).mean():.4f}")

    plot_complex_ratios(z_pooled, show=False, filename=filename, map="color")

def routine1(L, N, J, U, gamma, dissipation, c_ops_template, n_local_max=n_local_max):

    L_op = bose_hubbard_lindbladian(L, N, J, U, gamma, dissipation, c_ops_template, n_local_max=n_local_max)
    sector_evals = lindblad_eigenvalues_by_M(L_op, L, N, n_local_max=n_local_max, M_chosen=M, symmetric=False)
    #plot_spectrum(np.concatenate(list(sector_evals.values())))
    all_z = sector_stats(sector_evals)
    plot_complex_ratios(all_z, show=True, path=(filename, subfolder))
    #csv_results(csv_path, L, N, J, U, gamma, dissipation, M, np.concatenate(list(sector_evals.values())), all_z)

if test_lindbladian:

    if symmetric_dissipation:

        L=3
        N=5
        n_local_max = N
        
        """
        basis = build_bose_basis(L, N, fixed_N=False)
        sym_basis = build_sym_basis(basis)
        """
        lind = bose_hubbard_lindbladian(L, N, J, U, gamma, dissipation, c_ops_template_sym, n_local_max=n_local_max, is_symmetric=True)
        L_op = lind.L_op
        
        block_evals = compute_block_evals(lind)
        all_evals = pool_evals(block_evals)
        plot_spectrum(all_evals)

        all_z = compute_csr_from_evals(block_evals, complex_spacing_ratios)
        plot_complex_ratios(all_z, show=True) 
        """

        #print_lindbladian_orbits(2,1, n_local_max=1,k=None)

        #print_ket_orbits(L,N, n_local_max=n_local_max, k=0)
        #print_lindbladian_orbits(L,N,n_local_max=n_local_max, k=0)
        #symmetric_routine_sectors(L, N, J, U, gamma, dissipation, c_ops_template_sym, n_local_max=n_local_max)
        
        for J in ((-0.5,-0.2,0.2,0.5)):
            for gamma in ((0.1,0.2,0.4)):

                filename = f"L{L}_N{N}_J{J}_U{U}_gamma{gamma}_{dissipation}"
                symmetric_routine(L, N, J, U, gamma, dissipation, c_ops_template_sym, n_local_max=n_local_max)
        

        L_op = bose_hubbard_lindbladian(L, N, J, U, gamma, dissipation, c_ops_template_sym, n_local_max=n_local_max)
        sector_evals = lindblad_eigenvalues_by_M(L_op, L, N, n_local_max=n_local_max, M_chosen=1, symmetric=True)
        #plot_spectrum(np.concatenate(list(sector_evals.values())))
        all_z = sector_stats(sector_evals)
        plot_complex_ratios(all_z, show=True, filename=filename)"""

    if not symmetric_dissipation:

        """
        csv_path = "data/results.csv"
        for J in ((-0.5,-0.2,0.2,0.5)):
            for gamma1 in ((0.1,0.2,0.3,0.4)):

                gamma = (gamma1,)
                filename = f"L{L}_N{N}_J{J}_U{U}_gamma{gamma}_{dissipation}_M={M}"
                routine1(L, N, J, U, gamma, dissipation, c_ops_template1, n_local_max=n_local_max)
        """
        routine1(L, N, J, U, gamma, dissipation, c_ops_template1, n_local_max=n_local_max)     

        """
        L_op = bose_hubbard_lindbladian(L, N, J, U, gamma, dissipation, c_ops_template1, n_local_max=n_local_max, restrict_symmetry=False)
        sector_evals = lindblad_eigenvalues_by_M(L_op, L, N, n_local_max=n_local_max, M_chosen=1)
        plot_spectrum(np.concatenate(list(sector_evals.values())))
        all_z = sector_stats(sector_evals)
        plot_complex_ratios(all_z, show=False, filename=filename)
        """

        """
        evals = L_op.eigenenergies()
        evals = clean_num_error(evals)
        plot_spectrum(evals)
        z, r = complex_spacing_ratios(evals)
        r_mean = r.mean()
        #print(f"⟨r⟩ = {r_mean:.4f}  (GinUE: 0.739, Poisson: 0.500)")
        plot_complex_ratios(z, show=True, filename=filename, map="color")"""

    """
    for J in ((-0.5,-0.3,-0.1,0.1,0.3,0.5)):
        for gamma in ((0.1,0.2,0.3,0.5)):

            filename = f"L{L}_N{N}_nlmax{n_local_max}_J{J}_U{U}_gamma{gamma}"
            
    
            L_op = bose_hubbard_lindbladian(L, N, J, U, gamma, c_ops_template1, n_local_max=n_local_max, restrict_symmetry=False, k=k, gamma2=gamma2)
                    
            evals = L_op.eigenenergies()
            evals = clean_num_error(evals)
            #plot_spectrum(evals)
            z, r = complex_spacing_ratios(evals)
            r_mean = r.mean()
            print(f"⟨r⟩ = {r_mean:.4f}  (GinUE: 0.739, Poisson: 0.500)")
            plot_complex_ratios(z, filename=filename)
    """

    #evals = evals_by_sector(L_op, L, N, n_local_max=n_local_max)
    #plot_spectrum_by_NaNb(evals, N, is_single_color=True)
    #plot_complex_ratios_by_NaNb(evals, N, is_single_color=True)
    

    #im_evals = np.imag(evals)
    #im_evals = np.sort(im_evals[abs(im_evals) < 1e10])

    #spacings = normalized_nn_spacings(evals, remove_edge_frac=0.1)

    #plot_NN_spacings(spacings, funcs=[poisson2d])

if test_hamiltonian:

    H_op = bose_hubbard_hamiltonian(L, N, J, U, restrict_symmetry=True)
    evals = H_op.eigenenergies()


    #spacings = level_spacings(evals, unfolding=True, degree=4)

    #plot_level_density(spacings, funcs=[poisson, wigner], range=(0,5))

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