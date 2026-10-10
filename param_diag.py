import numpy as np

###################################### PARAMETER DIAGNOSTICS ######################################

def uniform_fixed_points(g, det_tilde, kappa, f):
    """Uniform classical fixed points n = |z|^2
        g^2 n^3 - 2 g det_tilde n^2 + (det_tilde^2 + kappa^2/4) n - f^2 = 0

    Returns:
        np.ndarray: real roots, sorted (1 root = monostable, 3 roots = bistable)
    """

    roots = np.roots([g**2, -2 * g * det_tilde, det_tilde**2 + kappa**2 / 4, -f**2])
    return np.sort(roots[np.abs(roots.imag) < 1e-9].real)


def cutoff_report(L, N, J, U, driving, gamma, n_cut=None):
    """Checks whether the Fock cutoff can hold the state the parameters ask for

    The drive is scaled as f*sqrt(N) and g = U*N is held fixed, so the classical density
    n = |z|^2 corresponds to a mean site occupation <n_i> = N * n. The basis, however, is
    cut at sum_i n_i <= n_cut. If L * N * n exceeds n_cut, the computed spectrum is not the faithful spectrum of the model.

    Args:
        L, N, J, U, driving: as in BoseHubbard
        gamma (tuple of floats): dissipation RATES
        n_cut (int, optional): total-boson cutoff of the basis. Defaults to N.
    """

    f = driving[0]
    det = driving[1]

    if n_cut is None:
        n_cut = N

    b = 2 if L > 2 else L - 1  # Number of bonds of a periodic chain

    g = 2.0 * U                     
    det_tilde = det + b * J + U / N
    kappa = gamma[0] - (gamma[1] if len(gamma) > 1 else 0.0)  # Warning! This kappa is not the same as the kappa in the symmetry sector labels, which is a quasimomentum index.

    n_cl = uniform_fixed_points(g, det_tilde, kappa, f)
    n_site = N * n_cl.max()
    n_total = L * n_site
    n_pump = (gamma[1] / kappa) if (len(gamma) > 1 and kappa > 0) else 0.0

    bistable = (g * det_tilde > 0) and (abs(det_tilde) > np.sqrt(3) / 2 * kappa)
    ok = (n_total < 0.5 * n_cut) and (kappa > 0)

    print(f"[params] g = {g:.4g}, Delta_tilde = {det_tilde:.4g}, kappa = {kappa:.4g}, f = {f:.4g}")
    branches = "3 branches, f is inside the hysteresis window" if len(n_cl) == 3 else "single branch"
    region = "the region admits bistability" if bistable else "monostable region (|Delta_tilde| <= sqrt(3)/2 kappa or g*Delta_tilde < 0)"
    print(f"[params] uniform classical fixed points n = {np.array2string(n_cl, precision=4)}"
            f"  ({branches}; {region})")
    print(f"[cutoff] <n_i> = N*n = {n_site:.3f}  ->  sum_i <n_i> = {n_total:.3f}   vs   cutoff sum_i n_i <= {n_cut}")
    print(f"[cutoff] incoherent pump background gamma_p/kappa = {n_pump:.3f} per site "
            f"({L * n_pump:.3f} in total)")
    if kappa <= 0:
        print("[cutoff] !! kappa <= 0: gain exceeds loss, there is no normalisable steady state")
    elif not ok:
        print(f"[cutoff] !! the state does not fit in the basis: raise n_cut well above "
                f"{n_total:.0f}, or lower f / raise kappa. The spectrum is a cutoff artefact.")
    else:
        print("[cutoff] ok: the classical state fits comfortably inside the basis")

    return