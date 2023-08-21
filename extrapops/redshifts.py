"""
.. module:: redshifts

:Synopsis: Redshift-dependent part of the population
:Author: Jesus Torrado (partially based on work with collaborators, see README)


With respect to z, the # events is a non-homogeneous Poisson point process with
lambda=p(z). This means that we can sample it as such, following 2 steps:

1. get the number of events as a draw from the Poisson pdf, for the full interval using
   Lambda=int(lambda=p(z) over z)

2. sample the events uniformly on the interval, in this case uniform-transformed using
   inverse CDF sampling (https://en.wikipedia.org/wiki/Inverse_transform_sampling), since
   the process is non-homogeneous.

"""

from copy import deepcopy
from functools import partial
import numpy as np
from scipy import stats, interpolate, integrate

import extrapops.constants as const
from extrapops.cosmology import diff_comoving_volume_re_z, _default_cosmo, \
    _default_z_perdecade
from extrapops.tools import invCDFinterp


# Fid values from Madau & Fragos '16 (https://arxiv.org/abs/1606.07887)
# with d adjusted to 2.7 (median of GWTC-3 power law)
_madau_fiducial_params = {"d": 2.7, "r": -3.6, "z_peak": 2.04, "c": None}

# Fid values (median) from LIGO/Virgo O3 Populations paper
# (https://arxiv.org/abs/2111.03634)
_default_redshift_params = {
    # Redshift range and precision:
    # min set to avoid SNR divergence due to extremely close events
    "z_range": [1e-5, 1],  # (1e-5 -> ~45 kpc)
    "z_perdecade": _default_z_perdecade,
    # Max years of coalescence time for a BBH mergine event (def 10000)
    # Also used as generation time. In detector reference frame.
    "T_yr": 10000,
    # Merger rate model: "const"|"power_law"|"madau"...
    "merger_rate_model": "madau",
    "merger_rate_params": {
        "z_0": 0,
        "R_0": 17.3,  # +10.3 -6.7 Gpc^⁻3 yr^⁻1
        "d": 2.7,  # +1.8 -1.9
    },
}

_default_redshift_params["merger_rate_params"].update(_madau_fiducial_params)


def R(z, model=_default_redshift_params["merger_rate_model"],
      params=_default_redshift_params["merger_rate_params"]):
    """Merger rate in GPc^-3 yr^-1 units."""
    if callable(model):
        try:
            return model(z, **(params or {}))
        except TypeError as excpt:
            raise TypeError(f"Error with custom callable merger rate: {excpt}") from excpt
    elif model.lower() == "madau":
        return R_madau(z, **params)
    elif model.lower() == "power_law":
        return R_power_law(z, **params)
    elif model.lower() == "const":
        return R_const(z, **params)
    else:
        raise ValueError(f"Merger rate model {model} nor found. "
                         "Use one of 'madau', 'power_law', 'const'.")


def _madau_z_dependence(
        z, d=_madau_fiducial_params["d"], r=_madau_fiducial_params["r"],
        c=_madau_fiducial_params["c"], z_peak=_madau_fiducial_params["z_peak"]):
    """
    Redshift-dependent term for an evolving merger rate following Madau & Dickinson
    1403.0007.

    Rising power ``r`` is negative, and declining power ``d`` is positive, since they are
    understood towards smaller ``z``.
    """
    if d < 0 or r > 0:
        raise ValueError("'d' must be positive, and 'r' must be negative.")
    if (z_peak is None and c is None) or (z_peak is not None and c is not None):
        raise ValueError("Please specify either `c` or `z_peak`.")
    elif c is not None:
        bottom_term = ((1. + z) / c)**(d - r)
    else:  # z_peak is not None
        bottom_term = (d / -r) * ((1. + z) / (1 + z_peak))**(d - r)
    return (1. + z)**d / (1. + bottom_term)


def _madau_safe_z_dependence(
        z, d=_madau_fiducial_params["d"], r=_madau_fiducial_params["r"],
        c=_madau_fiducial_params["c"], z_peak=_madau_fiducial_params["z_peak"]):
    """
    Redshift-dependent term for an evolving merger rate following Madau & Dickinson
    1403.0007.

    For negative ``d``, turns into a simple power law.
    """
    if c is not None:
        raise NotImplementedError("The 'safe' version of Madau cannot take 'c'.")
    if r > 0:
        raise ValueError("'r' must be negative.")
    if d <= 0:
        return _power_law_z_dependence(z, p=d)
    return _madau_z_dependence(z, d=d, r=r, z_peak=z_peak)


def R_madau(z, R_0, z_0=0, d=_madau_fiducial_params["d"],
            r=_madau_fiducial_params["r"], c=_madau_fiducial_params["c"],
            z_peak=_madau_fiducial_params["z_peak"], safe=True):
    """
    Redshift-dependent merger rate following the SFR of Madau & Dickinson 1403.0007.

    Rising power ``r`` is negative, and declining power ``d`` is positive, since they are
    understood towards smaller ``z``.

    If ``safe=True`` (default), negative ``d`` values are allowed, and it turns into a
    simple power law.

    ``z_0`` is the redshift at which the pivot rate ``R_0`` is defined.

    For the position of the peak, one can specify either the actual position ``z_peak``,
    or the divisor ``c`` of the ``(1+z)`` term in the denominator (written sometimes as
    ``(1 + z_peak)``, though is not the actual peak position).
    """
    z_dependence = _madau_safe_z_dependence if safe else _madau_z_dependence
    result = R_0 * z_dependence(z, d=d, r=r, c=c, z_peak=z_peak)
    result /= z_dependence(z_0 or 0, d=d, r=r, c=c, z_peak=z_peak)
    return result


def _power_law_z_dependence(z, p=_madau_fiducial_params["d"]):
    """Redshift-dependent term for an evolving merger rate."""
    return (1. + z)**p


def R_power_law(z, R_0, p=_madau_fiducial_params["d"], z_0=None):
    """
    Redshift-dependent power-law merger rate with power ``p``.

    ``z_0`` is the redshift at which the pivot rate ``R_0`` is defined.
    """
    result = R_0 * _power_law_z_dependence(z, p=p)
    if z_0 is not None:
        result /= _power_law_z_dependence(z_0, p=p)
    return result


def R_const(z, R_0):
    """Constant merger-rate."""
    return R_0


def event_rate_persec(z, merger_rate_model=_default_redshift_params["merger_rate_model"],
                      merger_rate_params=_default_redshift_params["merger_rate_params"],
                      cosmo_params=_default_cosmo,
                      z_range=_default_redshift_params["z_range"],
                      z_perdecade=_default_redshift_params["z_perdecade"],
                      trust_cache=False):
    """Number density of events per second (in detector frame) with respect to z."""
    # The 1e-9 turns the Mpc**3 of diff_comoving_volume_re_z(z) into Gpc**3 to cancel the
    #  Gpc**-3 of R(z)
    return R(z, model=merger_rate_model, params=merger_rate_params) / const.yr_s * \
        diff_comoving_volume_re_z(
            z, z_range=z_range, z_perdecade=z_perdecade,
            trust_cache=trust_cache, **cosmo_params) * \
        (1e-3)**3 * (1. + z)**-1


_invCDF_cache_params = None
_invCDF_interpolator = None
_z_total_events_avg = None


def _check_cache_invCDF(T_yr, merger_rate_model, merger_rate_params, cosmo_params,
                        z_range, z_perdecade):
    if not _invCDF_cache_params or not _invCDF_interpolator or not _z_total_events_avg:
        return False
    if not np.isclose(T_yr, _invCDF_cache_params["T_yr"]):
        return False
    check_maybe_none = lambda a, b: (
        a == b if (a is None or b is None) else np.isclose(a, b))
    if callable(merger_rate_model):
        if not callable(_invCDF_cache_params["merger_rate_model"]):
            return False
        # Comparison of function with copy fails e.g. for partials -- Workaround:
        if isinstance(merger_rate_model, partial):
            if not isinstance(_invCDF_cache_params["merger_rate_model"], partial):
                return False
            if any(
                    (getattr(merger_rate_model, attr) !=
                     getattr(_invCDF_cache_params["merger_rate_model"], attr))
                    for attr in ["func", "args", "keywords"]):
                return False
        # All other cases
        elif merger_rate_model != _invCDF_cache_params["merger_rate_model"]:
            return False
    else:  # not callable
        if merger_rate_model != _invCDF_cache_params["merger_rate_model"]:
            return False
    if not np.all([
            check_maybe_none(merger_rate_params.get(p, None), cached_value)
            for p, cached_value
            in _invCDF_cache_params["merger_rate_params"].items()]):
        return False
    if not np.all(
            np.isclose(cosmo_params[param], cached_value)
            for param, cached_value in _invCDF_cache_params["cosmo_params"].items()):
        return False
    if not np.allclose(z_range, _invCDF_cache_params["z_range"]) or \
       z_perdecade > _invCDF_cache_params["z_perdecade"]:
        return False
    return True


def _cache_invCDF(T_yr, merger_rate_model, merger_rate_params, cosmo_params, z_range,
                  z_perdecade, plot=False):
    global _invCDF_cache_params, _invCDF_interpolator, _z_total_events_avg
    if _check_cache_invCDF(
            T_yr=T_yr, merger_rate_model=merger_rate_model,
            merger_rate_params=merger_rate_params, cosmo_params=cosmo_params,
            z_range=z_range, z_perdecade=z_perdecade):
        return
    _invCDF_cache_params = deepcopy(
        {"T_yr": T_yr, "z_range": z_range, "z_perdecade": z_perdecade,
         "merger_rate_model": merger_rate_model, "merger_rate_params": merger_rate_params,
         "cosmo_params": cosmo_params})
    # recaching takes ~1s (dep on precision)
    # Prepare distribution
    assert z_range[0] > 0, "z_min must be >0 (e.g. 1e-5)"
    z_log10range = np.log10(z_range)
    n_z_decades = z_log10range[1] - z_log10range[0]
    zs = np.logspace(
        z_log10range[0], z_log10range[1], int(np.ceil(z_perdecade * n_z_decades)))
    func = lambda z, trust_cache=False: T_yr * const.yr_s * event_rate_persec(
        z, merger_rate_model=merger_rate_model,
        merger_rate_params=merger_rate_params, cosmo_params=cosmo_params,
        z_range=z_range, z_perdecade=z_perdecade, trust_cache=trust_cache)
    # Ensure cache is generated by calling the function once
    func((z_range[1] - z_range[0]) / 2)
    CDF_samples, _invCDF_interpolator = invCDFinterp(zs, func, pdf_args=(True,))
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(zs, CDF_samples, "-o")
        plt.loglog()
        plt.show()
    # Average number of samples in the whole z range
    z_Nsamples_avg_integral = integrate.quad(func, *z_range)
    if z_Nsamples_avg_integral[1] > 0.5:
        raise ValueError("Not enough integral precision for avg #draws "
                         "(must be precise up to integers).")
    _z_total_events_avg = int(z_Nsamples_avg_integral[0])


def avg_n_events(T_yr=_default_redshift_params["T_yr"],
                 merger_rate_model=_default_redshift_params["merger_rate_model"],
                 merger_rate_params=_default_redshift_params["merger_rate_params"],
                 cosmo_params=_default_cosmo, z_range=_default_redshift_params["z_range"],
                 z_perdecade=_default_redshift_params["z_perdecade"]):
    """
    Average total number of events. Simulate realisation sizes with

        import scipy.stats as stats
        n_avg = avg_n_events(...)
        n_realisation = stats.poisson(n_avg).rvs()

    The standard deviation is `sqrt(avg_n_events)`.
    """
    _cache_invCDF(T_yr=T_yr, merger_rate_model=merger_rate_model,
                  merger_rate_params=merger_rate_params, cosmo_params=cosmo_params,
                  z_range=z_range, z_perdecade=z_perdecade)
    return deepcopy(_z_total_events_avg)


def draw_Nsamples(T_yr=_default_redshift_params["T_yr"],
                  merger_rate_model=_default_redshift_params["merger_rate_model"],
                  merger_rate_params=_default_redshift_params["merger_rate_params"],
                  cosmo_params=_default_cosmo,
                  z_range=_default_redshift_params["z_range"],
                  z_perdecade=_default_redshift_params["z_perdecade"], trust_cache=False):
    """
    Draws a random number of events for a population,  generated during time `T_yr` (in
    years, in detector frame) in the redshift range `z_range`.
    """
    if not trust_cache:
        _cache_invCDF(T_yr=T_yr, merger_rate_model=merger_rate_model,
                      merger_rate_params=merger_rate_params, cosmo_params=cosmo_params,
                      z_range=z_range, z_perdecade=z_perdecade)
    return stats.poisson(_z_total_events_avg).rvs()


def sample_z(Nsamples=None, T_yr=_default_redshift_params["T_yr"],
             merger_rate_model=_default_redshift_params["merger_rate_model"],
             merger_rate_params=_default_redshift_params["merger_rate_params"],
             cosmo_params=_default_cosmo, z_range=_default_redshift_params["z_range"],
             z_perdecade=_default_redshift_params["z_perdecade"]):
    """
    Generate redshift for events during time `T_yr` (in years, in detector frame) in the
    redshift range `z_range`.

    Use `Nsamples` to override the number of samples generated (for testing only!
    it will not be consistent in general with the rest of parameters).
    """
    _cache_invCDF(T_yr=T_yr, merger_rate_model=merger_rate_model,
                  merger_rate_params=merger_rate_params, cosmo_params=cosmo_params,
                  z_range=z_range, z_perdecade=z_perdecade)
    if Nsamples is None:
        Nsamples = draw_Nsamples(T_yr=T_yr, merger_rate_model=merger_rate_model,
                                 merger_rate_params=merger_rate_params,
                                 cosmo_params=cosmo_params, z_range=z_range,
                                 z_perdecade=z_perdecade, trust_cache=True)
    # Check that it fits in memory, since generated by multiple processes at the same time
    # NB: not needed if not using parallelisation
    try:
        # numpy.empty does not actually take up mem, but fails if not available
        np.empty(Nsamples, dtype=float)
    except MemoryError as excpt:
        raise MemoryError("Not enough memory for z sample. Reduce the merger rate, the "
                          "redshift range or the total generation time.") from excpt
    # Parallel implementation (disabled for now, since threads were not being closed
    # at KeyboardInterrupt
    # Nsamples_chunk = NUM_THREADS * [int(Nsamples / NUM_THREADS)]
    # for i in range(Nsamples % NUM_THREADS):
    #     Nsamples_chunk[i] += 1
    # with Pool(NUM_THREADS) as pool:
    #     try:
    #         res = pool.map(
    #             lambda n: interpolate.splev(np.random.random(n), _invCDF_interpolator), Nsamples_chunk, chunksize=1)
    #     except:
    #         pool._clear()
    #         raise
    #     pool._clear()
    # return np.concatenate(res)
    return interpolate.splev(np.random.random(Nsamples), _invCDF_interpolator)
