"""
Approximate simplified SNR calculation, assuming monochromatic sources

Calculation by S. Babak.
"""

import numpy as np
from scipy import integrate, interpolate

import extrapops.constants as const


# Eqs (10), (13), (19) from https://arxiv.org/pdf/2108.01167.pdf
# using dimensionless strain (dnu/nu)
two_pi_cinv = 2 * np.pi / const.c_m_invs
omega_L = lambda f: two_pi_cinv * f * const.LISA_arm_length
Sacc = lambda f: (3e-15**2 / (2 * np.pi * f * const.c_m_invs)**2 *
                  (1 + (.4e-3 / f)**2) * (1 + (f / 8e-3)**4))
Soms = lambda f: (15e-12**2 * (two_pi_cinv * f)**2 *
                  (1 + (2e-3 / f)**4))
S_X = lambda f: (16 * np.sin(omega_L(f))**2 *
                 (Soms(f) + (3 + np.cos(2 * omega_L(f))) * Sacc(f)))
# TDI units (eq 56 same source):
#     S_{h,X} = 20 / 3 * (1 + 0.6 * (omega L)**2) * S_{X,1.5}
#                      / ((4 omega L)**2 * sin**2(omega L))
Sn = lambda f: (20 / 3 * (1 + 0.6 * omega_L(f)**2) /
                ((4 * omega_L(f))**2 * np.sin(omega_L(f))**2) *
                S_X(f))
# NB: that is for 1 channel. One needs to divide by 2 for the full TDI 1.5 with 6 links

# Common integrand for the precomputed integral
f_integrand = lambda f: f**(-7 / 3) / Sn(f)

# Since the integrand varies more smoothly in log-scale, let's log-sample
_f_integral_n = 5000
_f_integral_acc_loglog_interpolator = None


def _f_integral_interpolated(f_min, f_max, acc_loglog_interpolator):
    return (
        np.exp(acc_loglog_interpolator(np.log(np.clip(f_max, *const.LISA_bandwidth)))) -
        np.exp(acc_loglog_interpolator(np.log(np.clip(f_min, *const.LISA_bandwidth)))))


def _cache_f_integral():
    global _f_integral_acc_loglog_interpolator
    if _f_integral_acc_loglog_interpolator is None:
        fs_integral = np.logspace(np.log10(const.LISA_bandwidth[0]),
                                  np.log10(const.LISA_bandwidth[1]), _f_integral_n + 1)
        segment_integrals = np.empty(_f_integral_n)
        for i in range(_f_integral_n):
            segment_integrals[i] = integrate.quad(
                f_integrand, fs_integral[i], fs_integral[i + 1])[0]
        accumulated_integrals = [np.sum(segment_integrals[:i])
                                 for i in range(1, 1 + _f_integral_n)]
        _f_integral_acc_loglog_interpolator = interpolate.interp1d(
            np.log(fs_integral[1:]), np.log(accumulated_integrals))


def _snr_approx_factorless(chirp_mass, luminosity_distance, fmin, fmax, use_cache=True,
                           numerical=None):
    """
    Value of the event-dependent terms in the SNR calculation.
    Needs additional factors dependent on integration/maximisation over inclination.

    `chirp_mass` expected in kg, and `luminosity_distance` in m.

    All quantities must be in Solar System Barycentre reference frame.

    If `use_cache` is True, integrals over frequencies are interpolated from a
    precomputed table. Otherwise, they are computed on the fly (slower!).

    If `numerical` is set to a function of frequency containing the numerical interpolated
    "geometrical" response, this one is used and cache is ignored (slower, could be
    implemented with cache).
    """
    no_freqs_factor = (2 / np.pi**(2 / 3) * np.sqrt(5 / 96) *
                       (chirp_mass * const.G_m3_invkg_invs2)**(5 / 6) /
                       luminosity_distance * const.c_m_invs**(-3 / 2))
    if use_cache and not numerical:
        _cache_f_integral()
        integrator = lambda f_min, f_max: (
            _f_integral_interpolated(f_min, f_max, _f_integral_acc_loglog_interpolator))
        return no_freqs_factor * np.sqrt(integrator(fmin, fmax))
    else:
        if numerical:
            f_integrand_final = lambda f: (
                f_integrand(f) / (3 / 10 / (1 + 0.6 *omega_L(f)**2 )) * numerical(f))
        else:
            f_integrand_final = f_integrand
        integrator = lambda f_min, f_max: integrate.quad(
            f_integrand_final, np.clip(f_min, *const.LISA_bandwidth),
            np.clip(f_max, *const.LISA_bandwidth))[0]
        return no_freqs_factor * np.sqrt([
            integrator(this_fmin, this_fmax) for this_fmin, this_fmax in zip(
                np.atleast_1d(fmin), np.atleast_1d(fmax))])


def snr_avg_inclination(chirp_mass, luminosity_distance, fmin, fmax, use_cache=True,
                        numerical=None):
    """
    Approximate SNR, averaged over inclination.

    `chirp_mass` expected in kg, and `luminosity_distance` in m.

    All quantities must be in Solar System Barycentre reference frame.

    If `use_cache` is True, integrals over frequencies are interpolated from a
    precomputed table. Otherwise, they are computed on the fly (slower, and not
    vectorised!).

    If `numerical` is set to a function of frequency containing the numerical interpolated
    "geometrical" response, this one is used and cache is ignored (slower, could be
    implemented with cache).
    """
    return np.sqrt(32 / 5) * _snr_approx_factorless(
        chirp_mass, luminosity_distance, fmin, fmax, use_cache=use_cache,
        numerical=numerical)


def snr_max_inclination(chirp_mass, luminosity_distance, fmin, fmax, use_cache=True):
    """
    Approximate SNR, maximised over inclination, i.e. usually overestimated.

    Useful for conservative catalogues, in which events with actual SNR higher than a
    given one are less likely to be missed if estimated this way (but events with actually
    lower SNR will be included too).

    `chirp_mass` expected in kg, and `luminosity_distance` in m.

    All quantities must be in Solar System Barycentre reference frame.

    If `use_cache` is True, integrals over frequencies are interpolated from a
    precomputed table. Otherwise, they are computed on the fly (slower, and not
    vectorised!).
    """
    return np.sqrt(16) * _snr_approx_factorless(
        chirp_mass, luminosity_distance, fmin, fmax, use_cache=use_cache)
