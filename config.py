from dataclasses import dataclass

@dataclass
class Config:
    N_none: bool                        # has no N symmetry
    N_weak: bool                        # has weak N symmetry
    N_strong: bool                      # has strong N symmetry
    has_parity: bool                    # has parity symmetry
    has_driving: bool                   # Hamiltonian includes driving
    zero_gamma_idx: tuple[int, ...]     # indices of gamma coefficients that are zero;
                                        # gamma = (g_loss, g_pump, g_deph, g_circ_p, g_circ_m, g_bpl)

CONFIGS = {
    "pumploss_driving": Config(
        N_none=True,
        N_weak=False,
        N_strong=False,
        has_parity=True,
        has_driving=True,
        zero_gamma_idx=(2,3,4)
    ),

    "circulation": Config(
        N_none=False,
        N_weak=False,
        N_strong=True,
        has_parity=False,
        has_driving=False,
        zero_gamma_idx=(0,1,2)
    ),

    "circulation_pumploss": Config(
        N_none=False,
        N_weak=True,
        N_strong=False,
        has_parity=False,
        has_driving=False,
        zero_gamma_idx=(2,)
    ),
}