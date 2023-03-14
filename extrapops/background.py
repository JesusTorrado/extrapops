"""
Background computation.

As of commit 9b97699, the background h_c^2(.01Hz), with default parameters is 1.10423e-44.
(checked stable up to 1/1e5 for 10x z_perdecade and m_perdecade).
"""

from copy import deepcopy
import numpy as np
from scipy import integrate
from warnings import warn

import extrapops.constants as const
from extrapops.cosmology import _default_cosmo, distance_to_redshift, H_to_invs, H
from extrapops.redshifts import _default_redshift_params, event_rate_persec, R
from extrapops.mass import pdf_m1_m2, chirp_mass, _default_mass_params


# min z_max for an error <1% error
_min_z_max_background = 5


def _warn_background(z_max):
    """Produces a warning if z_max used for background computation is too small."""
    if z_max < _min_z_max_background:
        warn("Background would be smaller than actual one, unless "
             f"z_max>={_min_z_max_background}")


def char_strain_sq_single_at_f1Hz(
        z, M, comov_distance, T_gen=_default_redshift_params["T_yr"]):
    """
    Single-event averaged characteristic strain squared at f=1Hz.

    Returns strain in units of s^(-4/3). To be multiplied by f^(-4/3)
    (freq in detector reference frame) to get unitless characteristic strain.

    Masses (in kg) and distances (in m) are expected in source reference frame.
    The generation time ``T_gen`` should be in detector frame and in sec.

    The sum of the values returned by this function for a full population is the Monte
    Carlo version of `char_strain_sq_numerical`.
    """
    constant_prefactor = 1 / 3 * np.pi**(-4 / 3) * const.G_m3_invkg_invs2**(5 / 3) * \
        const.c_m_invs**-3 * 1 / T_gen
    return constant_prefactor * M**(5 / 3) * comov_distance**-2 * (1 + z)**(-1 / 3)


def ndens_pm1m2(cosmo_params=_default_cosmo, redshift_params=_default_redshift_params,
                mass_params=_default_mass_params):
    """
    Product of number density times mass probability density function.
    """
    redshift_params = deepcopy(redshift_params)
    T_yr = redshift_params.pop("T_yr")
    return lambda z, m1, m2, trust_cache: (
        T_yr * const.yr_s * event_rate_persec(
            z, trust_cache=trust_cache, cosmo_params=cosmo_params, **redshift_params) *
        pdf_m1_m2(m1, m2, trust_cache=trust_cache, **mass_params))


_prefactor_background = (
    4 / 3 / np.pi**(1 / 3) * const.G_m3_invkg_invs2**(5 / 3) / const.c_m_invs**2 *
    # Units of R and 1/H
    const.Gpc_m**-3 * const.yr_s**-1 * H_to_invs**-1)


def char_strain_sq_numerical(
        cosmo_params=_default_cosmo, redshift_params=_default_redshift_params,
        mass_params=_default_mass_params, epsabs=1e-4, epsrel=1e-4, return_terms=False):
    """
    Characteristic strain squared a at f=1Hz obtained by numerical integration.

    Masses are expected in source reference frame.

    Returns strain in units of s^(-4/3). To be multiplied by f^(-4/3)
    (freq in detector reference frame) to get unitless characteristic strain.

    ``epsabs`` and ``epsrel`` are precision arguments of ``scipy.integrate.[X]quad``.
    A value of ``1e-4`` (default) for both is enough to make the numerical error larger
    than the population realisation error (~0.5%), though larger values may be enough. For
    smaller values, ``char_strain_sq_numerical_triple`` is faster, though less accurate
    (~0.2% undervalued).

    If ``return_terms=True`` (default: ``False``), returns a tuple containing the redshift
    and mass terms separately.
    """
    z_term = _char_strain_sq_numerical_redshift(
        cosmo_params=cosmo_params, redshift_params=redshift_params,
        epsabs=epsabs, epsrel=epsrel)
    m_term = _char_strain_sq_numerical_mass(
        mass_params=mass_params, epsabs=epsabs, epsrel=epsrel)
    if return_terms:
        return (z_term, m_term)
    else:
        return z_term * m_term


def _char_strain_sq_numerical_redshift(
        cosmo_params=_default_cosmo, redshift_params=_default_redshift_params,
        epsabs=1e-4, epsrel=1e-4):
    """
    Computes the redshift part of the background integral, in SI units:

        C * int R(z) H(z)**-1 * (1 + z)**(-4/3)

    where C is the prefactor

        4/3 / np.pi**(1 / 3) * G**(5/3) / c**2

    See ``char_strain_sq_numerical`` for full computation and more documentation.
    """
    z_min, z_max = redshift_params["z_range"]
    z_integrand = lambda z: (
        R(z, model=redshift_params.get("merger_rate_model",
                                       _default_redshift_params["merger_rate_model"]),
          params=redshift_params["merger_rate_params"]) *
        H(z, **cosmo_params)**-1 * (1 + z)**(-4 / 3))
    return integrate.quad(z_integrand, z_min, z_max,
                          epsabs=epsabs, epsrel=epsrel)[0] * _prefactor_background


# NB: Main way to make integral faster would be by writing *scalar* versions of the mass
#     pdf functions (w/o np.where, np.atleast_1d...), since scipy.integrate.Xquad only
#     does scalar calls
#     (alternative, change to an integrator that does vectorised calls)
def _char_strain_sq_numerical_mass(
        mass_params=_default_mass_params, epsabs=1e-4, epsrel=1e-4):
    """
    Computes the mass part of the background integral, in SI units:

        int dm1 dm2 p(m1, m2) * chirp_mass(m1, m2)**(5/3)

    See ``char_strain_sq_numerical`` for full computation and more documentation.
    """
    m_min, m_max = mass_params["m_range"]
    # Force cache recomputation, to avoid checking cache during integral
    m_avg = (m_max + m_min) / 2
    m_eps = (m_max - m_min) * 1e-3
    pdf_m1_m2(m_avg, m_avg - m_eps, trust_cache=False, **mass_params)
    # Define the integral without cache checks, to be faster
    m_integrand = lambda m2, m1: (
        pdf_m1_m2(m1, m2, trust_cache=True, **mass_params) * chirp_mass(m1, m2)**(5 / 3))
    return integrate.dblquad(
        m_integrand, m_min, m_max, lambda m1: m_min, lambda m1: m1,
        epsabs=epsabs, epsrel=epsrel)[0] * const.Msun_kg**(5 / 3)


def avg_strain_squared(z, M, comov_distance, f):
    """
    Returns unitless individual strain squared at frequency f. Expects source reference
    frame and SI units for all quantities.
    """
    return (64 / 10 * np.pi**(4 / 3) * comov_distance**-2 *
            (const.G_m3_invkg_invs2 * M / const.c_m_invs**2)**(10 / 3) *
            (f / const.c_m_invs)**(4 / 3))


def char_strain_squared_to_Omegah2(f, h2):
    """
    Converts unitless characteristic strain squared to Omega h-squared units:

    :math:`Omega h^2 = 2 pi^2 / 3 / (H0/h)^2 f^2 * h_char^2(f)
    """
    return 2 * np.pi**2 / 3 / (100 * H_to_invs)**2 * np.array(f)**2 * np.array(h2)


def Omegah2_to_char_strain_squared(f, omh2):
    """
    Converts Omega h-squared to unitless characteristic strain squared units:

    :math:`Omega h^2 = 2 pi^2 / 3 / (H0/h)^2 f^2 * h_char^2(f)
    """
    return np.array(omh2) / (2 * np.pi**2 / 3 / (100 * H_to_invs)**2 * np.array(f)**2)
