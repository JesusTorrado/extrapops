"""
.. module:: cosmology

:Synopsis: Basic cosmological computations used throughout the code.
:Author: Jesus Torrado (partially based on work with collaborators, see README)

"""

from copy import deepcopy
import numpy as np
from scipy import integrate, interpolate

import extrapops.constants as const


# Values used is LIGO/Virgo O3 Populations paper:
# https://arxiv.org/abs/2111.03634
# This is probably an attempt at Planck 2015 + external
# (page 66 of https://wiki.cosmos.esa.int/planckpla2015/images/f/f7/Baseline_params_table_2015_limit68.pdf)
# but with the wrong Planck data combination:
# plikM_TE instead of plikHM_TTTEEE + lowTEB
_default_cosmo = {
    "H_0": 67.90,  # Km / (s MPc)
    "Omega_m": 0.3065,
    "Omega_l": 0.6935,
    "Omega_r": 0,
    "Omega_k": 0.,
}

_default_z_range = (1e-5, 1)
_default_z_perdecade = 500

H_to_invs = 1e3 / const.Mpc_m  # Conversion factor for H from km/s/Mpc to 1/s


def H(z, H_0=_default_cosmo["H_0"], Omega_r=_default_cosmo["Omega_r"],
      Omega_m=_default_cosmo["Omega_m"], Omega_k=_default_cosmo["Omega_k"],
      Omega_l=_default_cosmo["Omega_l"]):
    """Friedman eq. for evolution of H. Returns units of given `H_0`."""
    return H_0 * np.sqrt(
        Omega_r * (1 + z)**4 + Omega_m * (1. + z)**3. +
        Omega_k * (1. + z)**2. + Omega_l)


_distance_cache_params = None
_distance_interpolator = None


def _check_cache_distance(z_range, z_perdecade, **cosmo_params):
    if not _distance_cache_params or not _distance_interpolator:
        return False
    if not np.all(np.close(cosmo_params[param], cached_value)
                  for param, cached_value in _distance_cache_params["cosmo"].items()):
        return False
    if z_range[0] < _distance_cache_params["z_range"][0] or \
       z_range[1] > _distance_cache_params["z_range"][1] or \
       z_perdecade > _distance_cache_params["z_perdecade"]:
        return False
    return True


def _cache_distance(z_range, z_perdecade, **cosmo_params):
    global _distance_cache_params, _distance_interpolator
    if _check_cache_distance(z_range, z_perdecade, **cosmo_params):
        return
    _distance_cache_params = deepcopy(
        {"z_range": z_range, "z_perdecade": z_perdecade, "cosmo": cosmo_params})
    z_log10range = np.log10([z_range[0] / 2, z_range[1] * 2])
    n_z_decades = z_log10range[1] - z_log10range[0]
    zs = np.logspace(
        z_log10range[0], z_log10range[1], int(np.ceil(z_perdecade * n_z_decades)))
    distances = const.c_m_invs * 1e-3 * \
        np.array([integrate.quad(lambda zzz: 1 / H(zzz, **cosmo_params), 0, zz)[0]
                  for zz in zs])
    _distance_interpolator = interpolate.splrep(zs, distances)


def distance_to_redshift(
        z, H_0=_default_cosmo["H_0"], Omega_r=_default_cosmo["Omega_r"],
        Omega_m=_default_cosmo["Omega_m"], Omega_k=_default_cosmo["Omega_k"],
        Omega_l=_default_cosmo["Omega_l"], z_range=_default_z_range,
        z_perdecade=_default_z_perdecade):
    """
    Physical distance travelled by photon received with z:

    ds^2 = 0 --> dr = c dt / a -->
    --> r = c int_{a_emmited}^{a_received=a_0=1} H^-1 da / a^2 = c int_0^z dz / H

    Returns "units of 1/H", i.e. Mpc if H given in units of Km/(s*MPc) (default).
    """
    _cache_distance(z_range, z_perdecade, H_0=H_0, Omega_r=Omega_r, Omega_m=Omega_m,
                    Omega_k=Omega_k, Omega_l=Omega_l)
    return interpolate.splev(z, _distance_interpolator)


def luminosity_distance_to_redshift(
        z, H_0=_default_cosmo["H_0"], Omega_r=_default_cosmo["Omega_r"],
        Omega_m=_default_cosmo["Omega_m"], Omega_k=_default_cosmo["Omega_k"],
        Omega_l=_default_cosmo["Omega_l"], z_range=_default_z_range,
        z_perdecade=_default_z_perdecade):
    """
    Physical luminosity distance travelled by photon received with z, equals
    (1 + z) times the physical comoving distance.

    Returns units of Mpc, assuming H in units of Km/(s*MPc)
    """
    _cache_distance(z_range, z_perdecade, H_0=H_0, Omega_r=Omega_r, Omega_m=Omega_m,
                    Omega_k=Omega_k, Omega_l=Omega_l)
    return (1 + z) * interpolate.splev(z, _distance_interpolator)


def comoving_volume(
        z, H_0=_default_cosmo["H_0"], Omega_r=_default_cosmo["Omega_r"],
        Omega_m=_default_cosmo["Omega_m"], Omega_k=_default_cosmo["Omega_k"],
        Omega_l=_default_cosmo["Omega_l"], z_range=_default_z_range,
        z_perdecade=_default_z_perdecade):
    """
    Physical volume of comoving volume element within distance to redshift:

    4 / 3 pi r^3

    Returns units of Mpc^3
    """
    return 4 / 3 * np.pi * distance_to_redshift(
        z, z_range=z_range, z_perdecade=z_perdecade, H_0=H_0, Omega_r=Omega_r,
        Omega_m=Omega_m, Omega_k=Omega_k, Omega_l=Omega_l)**3


def diff_comoving_volume_re_z(
        z, H_0=_default_cosmo["H_0"], Omega_r=_default_cosmo["Omega_r"],
        Omega_m=_default_cosmo["Omega_m"], Omega_k=_default_cosmo["Omega_k"],
        Omega_l=_default_cosmo["Omega_l"], z_range=_default_z_range,
        z_perdecade=_default_z_perdecade, trust_cache=False):
    """
    (d/dz) of physical volume of comoving volume element within distance to redshift:

    (d/dz) 4 / 3 pi r^3

    Returns units of Mpc^3
    """
    # Since we use the interpolator twice, we don't call `distance_to_redshift` to save
    # one cache check
    if not trust_cache:
        _cache_distance(z_range, z_perdecade, H_0=H_0, Omega_r=Omega_r, Omega_m=Omega_m,
                        Omega_k=Omega_k, Omega_l=Omega_l)
    return 4 * np.pi * interpolate.splev(z, _distance_interpolator, der=1) * \
        interpolate.splev(z, _distance_interpolator)**2
