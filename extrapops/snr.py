"""
Approximate simplified SNR calculation, assuming monochromatic sources

Calculation by S. Babak.
"""

import numpy as np

import extrapops.constants as const
from extrapops.experiments import LISA


def _snr_approx_factorless(chirp_mass, luminosity_distance, fmin, fmax, exp):
    """
    Value of the event-dependent terms in the SNR calculation.
    Needs additional factors dependent on integration/maximisation over inclination.

    `chirp_mass` expected in kg, and `luminosity_distance` in m.

    Initial and final frequency will be clipped by the experiment's bandwidth.

    All quantities must be in Solar System Barycentre reference frame.

    The first call of this function for a newly instantiated ``Experiment`` will involve
    some short caching.
    """
    no_freqs_factor = (2 / np.pi**(2 / 3) * np.sqrt(5 / 96) *
                       (chirp_mass * const.G_m3_invkg_invs2)**(5 / 6) /
                       luminosity_distance * const.c_m_invs**(-3 / 2))
    # multiply SNR**2 by #channels (effective)
    no_freqs_factor *= np.sqrt(exp.n_channels_effective)
    # Clip frequencies to bandwidth
    f_integral_interpolated = lambda f_min, f_max: (
        np.exp(exp.f_integral_acc_loglog_interpolator(
            np.log(np.clip(f_max, *exp.bandwidth_Hz)))) -
        np.exp(exp.f_integral_acc_loglog_interpolator(
            np.log(np.clip(f_min, *exp.bandwidth_Hz)))))
    return no_freqs_factor * np.sqrt(f_integral_interpolated(fmin, fmax))


def snr_avg_inclination(chirp_mass, luminosity_distance, fmin, fmax, exp=LISA):
    """
    Approximate SNR, averaged over inclination.

    `chirp_mass` expected in kg, and `luminosity_distance` in m.

    Initial and final frequency will be clipped by the experiment's bandwidth.

    All quantities must be in Solar System Barycentre reference frame.

    The first call of this function for a newly instantiated ``Experiment`` will involve
    some short caching.
    """
    return np.sqrt(32 / 5) * _snr_approx_factorless(
        chirp_mass, luminosity_distance, fmin, fmax, exp=exp)


def snr_max_inclination(chirp_mass, luminosity_distance, fmin, fmax, exp=LISA):
    """
    Approximate SNR, maximised over inclination, i.e. usually overestimated.

    Useful for conservative catalogues, in which events with actual SNR higher than a
    given one are less likely to be missed if estimated this way (but events with actually
    lower SNR will be included too).

    `chirp_mass` expected in kg, and `luminosity_distance` in m.

    Initial and final frequency will be clipped by the experiment's bandwidth.

    All quantities must be in Solar System Barycentre reference frame.

    The first call of this function for a newly instantiated ``Experiment`` will involve
    some short caching.
    """
    return np.sqrt(16) * _snr_approx_factorless(
        chirp_mass, luminosity_distance, fmin, fmax, exp=exp)
