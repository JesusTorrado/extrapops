import numpy as np

import extrapops.constants as const

_C = 256 / 5 * np.pi**(8 / 3) * (const.G_m3_invkg_invs2 * const.c_m_invs**-3)**(5 / 3)
_C_inv = 1 / _C


def frequency_at_t(chirp_mass, t_coal, t):
    """
    Leading-order (Newtonian) frequency drifting.

    All quantities in SI units.

    Initial and final observed frequencies are the result of evaluating for
    t=0 and t=min(t_coal, t_obs) (clipped to the experimental bandwidth).

    Eq. (11a) from https://arxiv.org/abs/1806.10734,
    or integrating df/dr from eq. (4.196) in Maggiore, for f=0 at t=t_coal.
    """
    return (_C * chirp_mass**(5 / 3) * (t_coal - t))**(-3 / 8)


def t_of_frequency(chirp_mass, t_coal, f):
    """
    Leading-order (Newtonian) inverse frequency drifting.

    All quantities in SI units.

    Initial and final observed frequencies are defined respectively at t=0 and
    t=min(t_coal, t_obs) (clipped to the experimental bandwidth).

    From Eq. (11a) from https://arxiv.org/abs/1806.10734,
    or integrating df/dr from eq. (4.196) in Maggiore, for f=0 at t=t_coal.
    """
    return t_coal - _C_inv * chirp_mass**(-5 / 3) * f**(-8 / 3)
