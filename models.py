import numpy as np
import qutip as qt

import qm_statistics as stat
import plot as plot

def bose_hubbard_lindbladian(L=3, N=2, J=1.0, U=1.0, gamma=0.1):
    """
    Construct Bose-Hubbard Lindbladian for L sites, N bosons max.

    Parameters:
    -----------
    L : int
        Number of lattice sites
    N : int
        Max total boson number (Fock cutoff per site = N)
    J : float
        Hopping amplitude
    U : float
        On-site interaction
    gamma : float
        Local loss rate
    """

    # local Hilbert space dimension (cutoff N)
    d = N + 1

    # bosonic operators for one site
    a  = qt.destroy(d)
    n  = a.dag() * a

    # Build tensor operators for each site
    a_list, n_list = [], []
    for j in range(L):
        op_list = [qt.qeye(d) for _ in range(L)]
        op_list[j] = a
        a_list.append(qt.tensor(op_list))

        op_list[j] = n
        n_list.append(qt.tensor(op_list))

    # Bose-Hubbard Hamiltonian
    H = 0
    # hopping (closed boundary)
    for i in range(L):
        j = (i + 1) % L
        H += -0.5 * J * (a_list[i].dag() * a_list[j] + a_list[j].dag() * a_list[i])
    # onsite interaction
    for j in range(L):
        H += 0.5 * U * n_list[j] * (n_list[j] - qt.qeye(H.dims[0]))

    # collapse operators (loss at each site)
    c_ops = [np.sqrt(gamma) * a_j for a_j in a_list]

    # Build Liouvillian
    L_op = qt.liouvillian(H, c_ops)

    return L_op

def bose_hubbard_hamiltonian(L=3, N=2, J=1.0, U=1.0):

    # local Hilbert space dimension (cutoff N)
    d = N + 1

    # bosonic operators for one site
    a  = qt.destroy(d)
    n  = a.dag() * a

    # Build tensor operators for each site
    a_list, n_list = [], []
    for j in range(L):
        op_list = [qt.qeye(d) for _ in range(L)]
        op_list[j] = a
        a_list.append(qt.tensor(op_list))

        op_list[j] = n
        n_list.append(qt.tensor(op_list))

    # Bose-Hubbard Hamiltonian
    H = 0
    # hopping (closed boundary)
    """
    for i in range(L):
        j = (i + 1) % L
        H += -0.5 * J * (a_list[i].dag() * a_list[j] + a_list[j].dag() * a_list[i])
    """
    for i in range(L-1):
        H += -0.5 * J * (a_list[i].dag() * a_list[i+1] + a_list[i+1].dag() * a_list[i])
    # onsite interaction
    for j in range(L):
        H += 0.5 * U * n_list[j] * (n_list[j] - qt.qeye(H.dims[0]))

    return H


