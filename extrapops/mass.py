"""
.. module:: mass

:Synopsis: Mass-dependent part of the population
:Author: Jesus Torrado


For now, it implements the "Power Law + Peak" mass model described in appendix B.2 of
arxiv:2010.14533

In it, m1 and m2 **need** to be generated jointly, i.e. we cannot generate m2
independently, either when generating m2's or q's (where q = m2/m1)

First, notice that we can generate m1 by first sampling from the mixture, then
reject-sample from smoothing (removes ~1/2 of the draws -- the slowest part of the code!).
A possible faster implementation, which is the one finally used, we use instead inv CDF
transform sampling.

For m2, we generate q for each m1 from the power law in `[m_min/m_1, 1]`, and do rejection
sampling with the smoothing on m2 in `[m_min, m1]`. Due to the reduced domain for
rejection sampling when compared to m1, rejection sampling here is much faster than for
m1. After one step of the generation, for some m1 we will have rejected the m2 generated
from the power law in q, so we repeat the algorithm for the rejected ones until we have
an m2 for each m1. This converges in ~1s for ~1e6 events.

(Notice that converting the q prior to pi(m2|m1) makes it even more complicated to
generate m1 and m2 independtly, since now m1 enters as a boundary. But may be more
practical when doing inv CDF transf sampling.)

"""

import numpy as np
from copy import deepcopy
from scipy import stats, integrate, interpolate
from extrapops.tools import power_law_generate, invCDFinterp

# Fid values (median) from LIGO/Virgo O3 Populations paper
# (https://arxiv.org/abs/2111.03634)
_default_mass_params = {
    "m_range": [
        5.1,  # +0.9 -1.5 Msun
        86.9],  # +11.4 -9.4 Msun
    "delta_m": 4.8,  # +3.3 -3.2 Msun
    "alpha": 3.40,  # +0.58 -0.49
    "lambda_peak": 0.039,  # +0.058 -0.026
    "mu_m": 33.7,  # +2.3 -3.8 Msun
    "sigma_m": 3.6,  # +4.6 -2.1 Msun
    "beta_q": 1.1,  # +1.8 -1.3
    "m_perdecade": 400,  # precision parameter for invCDF sampling
    # TODO: can be smaller with cubic splines, but unstable. Maybe interp in log-space?
}


def chirp_mass(m1, m2):
    """Chirp mass (non-frame-dependent)."""
    return (m1 * m2)**(3 / 5) * (m1 + m2)**(-1 / 5)


def power_law_m1(m, m_range=_default_mass_params["m_range"],
                 alpha=_default_mass_params["alpha"]):
    """Power-law component of mass population."""
    return (1 - alpha) * \
        (m_range[1]**(1 - alpha) - m_range[0]**(1 - alpha))**-1 * m**(-alpha)


def gaussian_m1(m, mu_m=_default_mass_params["mu_m"],
                sigma_m=_default_mass_params["sigma_m"]):
    """Gaussian componet of the mass population."""
    return stats.norm.pdf(m, loc=mu_m, scale=sigma_m)


def smoothing(mprime, delta_m=_default_mass_params["delta_m"]):
    """Assumes won't get any mprime<=0 value!"""
    with np.errstate(divide="ignore", over="ignore"):
        # Select non-problematic values and assign 1 after delta_m
        mprime_array = np.atleast_1d(mprime)
        result = np.heaviside(mprime_array - delta_m, 1)
        i_smooth = np.logical_and(np.asarray(result == 0), np.asarray(mprime != 0))
        good_mprime = mprime_array[i_smooth]
        result[i_smooth] = (
            np.exp(delta_m / good_mprime + delta_m / (good_mprime - delta_m)) + 1)**-1
        return result


# TODO: caching could be split into the different components
_m1_norm_cache_params = None
_m1_norm_factor = None
_power_law_m1_norm_inv = None
_gaussian_m1_norm_inv = None


def _pdf_m1_unnorm(m, m_range, lambda_peak, alpha, mu_m, sigma_m, delta_m,
                   power_law_m1_norm_inv, gaussian_m1_norm_inv):
    """Non-normalised m1 pdf."""
    return np.where(np.logical_and(m_range[0] < m, m < m_range[1]),
                    ((1 - lambda_peak) * power_law_m1(m, m_range, alpha) /
                     power_law_m1_norm_inv +
                     lambda_peak * gaussian_m1(m, mu_m, sigma_m) / gaussian_m1_norm_inv) *
                    smoothing(m - m_range[0], delta_m),
                    0)


def _check_cache_m1(m_range, lambda_peak, alpha, mu_m, sigma_m, delta_m, cached_params):
    if not np.allclose(m_range, cached_params["m_range"]):
        return False
    if not np.isclose(lambda_peak, cached_params["lambda_peak"]):
        return False
    if not np.isclose(alpha, cached_params["alpha"]):
        return False
    if not np.isclose(mu_m, cached_params["mu_m"]) or \
       not np.isclose(sigma_m, cached_params["sigma_m"]):
        return False
    if not np.isclose(delta_m, cached_params["delta_m"]):
        return False
    return True


def _check_cache_m1_norm(m_range, lambda_peak, alpha, mu_m, sigma_m, delta_m):
    if not _m1_norm_cache_params or not _m1_norm_factor:
        return False
    if not _check_cache_m1(m_range, lambda_peak, alpha, mu_m, sigma_m, delta_m,
                           _m1_norm_cache_params):
        return False
    return True


def _cache_m1_norm(m_range, lambda_peak, alpha, mu_m, sigma_m, delta_m):
    global _m1_norm_cache_params, _m1_norm_factor, \
        _power_law_m1_norm_inv, _gaussian_m1_norm_inv
    if _check_cache_m1_norm(
            m_range=m_range, lambda_peak=lambda_peak,
            alpha=alpha, mu_m=mu_m, sigma_m=sigma_m, delta_m=delta_m):
        return
    _m1_norm_cache_params = deepcopy(
        {"m_range": m_range, "lambda_peak": lambda_peak, "alpha": alpha, "mu_m": mu_m,
         "sigma_m": sigma_m, "delta_m": delta_m})
    # Compute normalisation factor
    _power_law_m1_norm_inv = integrate.quad(
        lambda m1: power_law_m1(m1, m_range=m_range, alpha=alpha), *m_range)[0]
    _gaussian_m1_norm_inv = integrate.quad(
        lambda m1: gaussian_m1(m1, mu_m=mu_m, sigma_m=sigma_m), *m_range)[0]
    _m1_norm_factor = 1 / integrate.quad(
        lambda m: _pdf_m1_unnorm(
            m, m_range=m_range, lambda_peak=lambda_peak, alpha=alpha, mu_m=mu_m,
            sigma_m=sigma_m, delta_m=delta_m,
            power_law_m1_norm_inv=_power_law_m1_norm_inv,
            gaussian_m1_norm_inv=_gaussian_m1_norm_inv),
        m_range[0], m_range[1])[0]


def pdf_m1(m, m_range=_default_mass_params["m_range"],
           lambda_peak=_default_mass_params["lambda_peak"],
           alpha=_default_mass_params["alpha"], mu_m=_default_mass_params["mu_m"],
           sigma_m=_default_mass_params["sigma_m"],
           delta_m=_default_mass_params["delta_m"], trust_cache=False):
    """Normalised m1 pdf."""
    if not trust_cache:
        _cache_m1_norm(m_range=m_range, lambda_peak=lambda_peak,
                       alpha=alpha, mu_m=mu_m, sigma_m=sigma_m, delta_m=delta_m)
    return np.where(np.logical_and(m_range[0] < m, m < m_range[1]),
                    _pdf_m1_unnorm(
                        m, m_range=m_range, lambda_peak=lambda_peak,
                        alpha=alpha, mu_m=mu_m, sigma_m=sigma_m, delta_m=delta_m,
                        power_law_m1_norm_inv=_power_law_m1_norm_inv,
                        gaussian_m1_norm_inv=_gaussian_m1_norm_inv),
                    0) * _m1_norm_factor


# This one is ACTUALLY FASTER than the invCDF one if a lot of caching is needed (e.g.
# when simulating while sampling on some pdf of hyperparameters).
# It can be made MUCH faster if assuming the smoothing only affects the power law
# (reasonable: the smoothing is =1 for the whole gaussian range, for sensible gaussian
#  parameters). In that case, one can sample from the mixture of gaussian and
# [power_law*smoothing], without the costlier rejection step.
def sample_m1_mix_reject(N, use_smoothing=True, delta_m=_default_mass_params["delta_m"],
                         lambda_peak=_default_mass_params["lambda_peak"],
                         alpha=_default_mass_params["alpha"],
                         m_range=_default_mass_params["m_range"],
                         sigma_m=_default_mass_params["sigma_m"],
                         mu_m=_default_mass_params["mu_m"],):
    """
    Generate N draws from the m1 mass function, sampling from the population mixture and
    applying smoothing via rejection sampling.

    Slow but more precise method.
    """
    # COULD EVEN BE IMPROVED!!!!
    # - preselect the number of samples to be generated from each component
    # - also find alternative to rejection (what makes the code slowest)
    # - **parallelise!!!**
    # But at least this is fully vectorised!
    m_min, m_max = m_range
    chunk_size = int(N / 100)
    # different chunk_size fractions will be more efficient for different N
    # (for m1, more or less 1/2 are rejected, due to "log-scale" of neg powlaw generation)
    m1_sample = np.empty((N))
    m1_sample[:] = np.nan
    n_done = 0
    while n_done < N:
        # Sample from the mixture
        component = np.random.random(chunk_size) > lambda_peak
        sample_pl = power_law_generate(chunk_size, -alpha, m_min, m_max)
        sample_gauss = stats.norm(loc=mu_m, scale=sigma_m).rvs(chunk_size)
        m1_sample_chunk = np.where(component, sample_pl, sample_gauss)
        # Rejection sampling from smoothing function
        if use_smoothing:
            rejected = np.random.random(chunk_size) > \
                smoothing(m1_sample_chunk - m_min, delta_m)
            if sum(rejected) == chunk_size:
                continue
            m1_sample_chunk = m1_sample_chunk[np.where(np.logical_not(rejected))]
        # Store
        last_index = min(n_done + len(m1_sample_chunk), N)
        m1_sample[n_done:last_index] = m1_sample_chunk[:last_index - n_done]
        n_done += len(m1_sample_chunk)
    return m1_sample


# Faster alternative: inv CDF transform sampling

_m1_invCDF_cache_params = None
_m1_invCDF_interpolator = None


def _check_cache_m1_invCDF(m_range, lambda_peak, alpha,
                           mu_m, sigma_m, delta_m, m_perdecade):
    if not _m1_invCDF_cache_params or not _m1_invCDF_interpolator:
        return False
    if not _check_cache_m1(m_range, lambda_peak, alpha, mu_m, sigma_m, delta_m,
                           _m1_invCDF_cache_params):
        return False
    if m_perdecade > _m1_invCDF_cache_params["m_perdecade"]:
        return False
    return True


# FIXME: Sometimes, for extremely small values or m_min (w.r.t. delta_m, I think),
#        creates an interpolator that only returns NAN
def _cache_m1_invCDF(m_range, lambda_peak, alpha, mu_m, sigma_m, delta_m, m_perdecade,
                     plot=False):
    global _m1_invCDF_cache_params, _m1_invCDF_interpolator
    # Ensure recomputation also if *population parameters* change
    if _check_cache_m1_invCDF(m_range=m_range, lambda_peak=lambda_peak, alpha=alpha,
                              mu_m=mu_m, sigma_m=sigma_m, delta_m=delta_m,
                              m_perdecade=m_perdecade):
        return
    # NB: in case m1_norm cache test fails, make sure m1_norm recached.
    #     (We need to do test above for quick skip anyway.)
    _cache_m1_norm(m_range=m_range, lambda_peak=lambda_peak, alpha=alpha,
                   mu_m=mu_m, sigma_m=sigma_m, delta_m=delta_m)
    _m1_invCDF_cache_params = deepcopy(
        {"m_range": m_range, "lambda_peak": lambda_peak, "alpha": alpha, "mu_m": mu_m,
         "sigma_m": sigma_m, "delta_m": delta_m, "m_perdecade": m_perdecade})
    # Prepare distribution
    m_log10range = np.log10(m_range)
    n_m_decades = m_log10range[1] - m_log10range[0]
    m1s = np.logspace(
        m_log10range[0], m_log10range[1], int(np.ceil(m_perdecade * n_m_decades)))
    # We can trust_cache, because we have just tested it above!
    func = lambda m: pdf_m1(
        m, m_range=m_range, lambda_peak=lambda_peak, alpha=alpha, mu_m=mu_m,
        sigma_m=sigma_m, delta_m=delta_m, trust_cache=True)
    # NB: linear interpolation is enough (cubic fails!)
    CDF_samples, _m1_invCDF_interpolator = invCDFinterp(m1s, func, splrep_kwargs={"k": 1})
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(m1s, CDF_samples, "-o")
        plt.loglog()
        plt.show()


def sample_m1_invcdf(N, m_range=_default_mass_params["m_range"],
                     m_perdecade=_default_mass_params["m_perdecade"],
                     lambda_peak=_default_mass_params["lambda_peak"],
                     alpha=_default_mass_params["alpha"],
                     mu_m=_default_mass_params["mu_m"],
                     sigma_m=_default_mass_params["sigma_m"],
                     delta_m=_default_mass_params["delta_m"], trust_cache=False):
    """Generate N draws from the m1 mass function using inverse CDF transform sampling."""
    if not trust_cache:
        _cache_m1_invCDF(m_range=m_range, m_perdecade=m_perdecade,
                         lambda_peak=lambda_peak, alpha=alpha, mu_m=mu_m, sigma_m=sigma_m,
                         delta_m=delta_m)
    # TODO: see redshift sampling implementation in sample_z for parallelisation
    return interpolate.splev(np.random.random(N), _m1_invCDF_interpolator)


# m2 ###########################################################

# problem: though norm "does not matter", its dependency on the "given" parameters
# does matter in particular, pi_q|m1 norm depends on m1, and this should be taken into
# account!

def _pdf_q_cond_m1_unnorm(q, m1, m_range, beta_q, delta_m):
    """Non-normalised q|m1 pdf, where q=m2/m1."""
    return np.where(np.logical_and(m_range[0] / m1 <= q, q <= 1),
                    q**beta_q * smoothing(q * m1 - m_range[0], delta_m), 0)


_q_cond_m1_norm_cache_params = None
_q_cond_m1_norm_interpolator = None


def _check_cache_q_cond_m1_norm(beta_q, m_perdecade):
    # m_range and delta_m are checked elsewhere
    if not _q_cond_m1_norm_cache_params or not _q_cond_m1_norm_interpolator:
        return False
    if not np.isclose(beta_q, _q_cond_m1_norm_cache_params["beta_q"]):
        return False
    if m_perdecade > _q_cond_m1_norm_cache_params["m_perdecade"]:
        return False
    return True


def _cache_q_cond_m1_norm(m_range, m_perdecade, lambda_peak, alpha, mu_m, sigma_m,
                          delta_m, beta_q):
    global _q_cond_m1_norm_cache_params, _q_cond_m1_norm_interpolator
    # Ensure recomputation also if *m1 population parameters* change
    if _check_cache_m1_norm(m_range=m_range, lambda_peak=lambda_peak, alpha=alpha,
                            mu_m=mu_m, sigma_m=sigma_m, delta_m=delta_m) and \
       _check_cache_q_cond_m1_norm(beta_q=beta_q, m_perdecade=m_perdecade):
        return
    # NB: in case m1_norm cache test fails, make sure m1_norm recached.
    #     (We need to do test above for quick skip anyway.)
    _cache_m1_norm(m_range=m_range, lambda_peak=lambda_peak, alpha=alpha,
                   mu_m=mu_m, sigma_m=sigma_m, delta_m=delta_m)
    _q_cond_m1_norm_cache_params = deepcopy(
        {"beta_q": beta_q, "m_perdecade": m_perdecade})
    # Prepare distribution
    m_log10range = np.log10(m_range)
    n_m_decades = m_log10range[1] - m_log10range[0]
    m1s = np.logspace(
        m_log10range[0], m_log10range[1], int(np.ceil(m_perdecade * n_m_decades)))
    func = lambda q, m1: _pdf_q_cond_m1_unnorm(
        q, m1, m_range=m_range, beta_q=beta_q, delta_m=delta_m)
    _q_cond_m1_norm_interpolator = interpolate.interp1d(
        m1s, np.array([integrate.quad(
            lambda q: func(q, m1), m_range[0] / m_range[1], 1)[0] for m1 in m1s]),
        bounds_error=False, fill_value="extrapolate")


def pdf_q_cond_m1(
        q, m1, m_range=_default_mass_params["m_range"],
        lambda_peak=_default_mass_params["lambda_peak"],
        alpha=_default_mass_params["alpha"], mu_m=_default_mass_params["mu_m"],
        sigma_m=_default_mass_params["sigma_m"], delta_m=_default_mass_params["delta_m"],
        beta_q=_default_mass_params["beta_q"],
        m_perdecade=_default_mass_params["m_perdecade"], trust_cache=False):
    """Normalised q|m1 pdf, where q=m2/m1."""
    if not trust_cache:
        _cache_q_cond_m1_norm(
            m_range=m_range, lambda_peak=lambda_peak, alpha=alpha, mu_m=mu_m,
            sigma_m=sigma_m, delta_m=delta_m, beta_q=beta_q, m_perdecade=m_perdecade)
    unnorm_values = np.atleast_1d(_pdf_q_cond_m1_unnorm(
        q, m1, m_range=m_range, beta_q=beta_q, delta_m=delta_m))
    inv_norm_factors = _q_cond_m1_norm_interpolator(np.atleast_1d(m1))
    i_nonzero = np.logical_and(unnorm_values != 0, inv_norm_factors != 0)
    if not np.any(i_nonzero):
        return np.zeros(shape=unnorm_values.shape)
    unnorm_values[i_nonzero] /= inv_norm_factors[i_nonzero]
    return unnorm_values  # which are now normalised!


def pdf_m1_q(
        m1, q, m_range=_default_mass_params["m_range"],
        lambda_peak=_default_mass_params["lambda_peak"],
        alpha=_default_mass_params["alpha"], mu_m=_default_mass_params["mu_m"],
        sigma_m=_default_mass_params["sigma_m"], delta_m=_default_mass_params["delta_m"],
        beta_q=_default_mass_params["beta_q"],
        m_perdecade=_default_mass_params["m_perdecade"], trust_cache=False):
    """Normalised joint prior for m1, q."""
    cond_term = pdf_q_cond_m1(
        q, m1, m_range=m_range, lambda_peak=lambda_peak, alpha=alpha, mu_m=mu_m,
        sigma_m=sigma_m, delta_m=delta_m, beta_q=beta_q, m_perdecade=m_perdecade,
        trust_cache=trust_cache)
    # We can trust the cache for m1, since the previous call has checked it!
    return pdf_m1(m1, m_range=m_range, lambda_peak=lambda_peak, alpha=alpha, mu_m=mu_m,
                  sigma_m=sigma_m, delta_m=delta_m, trust_cache=trust_cache) * cond_term


def pdf_m1_m2(
        m1, m2, m_range=_default_mass_params["m_range"],
        lambda_peak=_default_mass_params["lambda_peak"],
        alpha=_default_mass_params["alpha"], mu_m=_default_mass_params["mu_m"],
        sigma_m=_default_mass_params["sigma_m"], delta_m=_default_mass_params["delta_m"],
        beta_q=_default_mass_params["beta_q"],
        m_perdecade=_default_mass_params["m_perdecade"], trust_cache=False):
    """Normalised joint prior for m1, m2."""
    return pdf_m1_q(m1, m2 / m1, m_range=m_range, lambda_peak=lambda_peak, alpha=alpha,
                    mu_m=mu_m, sigma_m=sigma_m, delta_m=delta_m, beta_q=beta_q,
                    m_perdecade=m_perdecade, trust_cache=trust_cache) / m1


# Using rejection sampling for now.
# We could also precompute some interpolation of the inverse CDF,
# but it would have to be 2D
# and we would need to be more careful with the error. Rejection sampling is safer.

def sample_m2_cond_m1_rejection(m1_sample, m_range=_default_mass_params["m_range"],
                                beta_q=_default_mass_params["beta_q"],
                                delta_m=_default_mass_params["delta_m"], verbose=False):
    """
    Draws m2 conditioned to a given m1 sample, drawing from the power law and rejecting
    with the smoothing function.
    """
    Nsamples = len(m1_sample)
    m_min = m_range[0]
    m2_sample = np.full(shape=Nsamples, fill_value=np.nan)
    m2_empty = np.full(shape=Nsamples, fill_value=True, dtype=bool)
    while True:
        i_this_m2_empty = np.where(m2_empty)
        this_m1 = m1_sample[i_this_m2_empty]
        N_m2 = len(i_this_m2_empty[0])
        if verbose:
            print("%d left" % N_m2)
        if not N_m2:
            break
        # Sample m2 from power law in q, and transform
        q_sample = power_law_generate(N_m2, beta_q, m_min / this_m1, 1)
        this_m2 = q_sample * this_m1
        # Rejection sampling from m1 smoothing function
        # In the next line, the division by smoothing(m1) does not change the relative
        # acceptance rate between two m2 given some m1, but does change the total
        # acceptance rate, making the algorithm faster.
        # Analogy: acceptance-rejection on a circle: we get the shape of the circle
        # regardless of the size of the prior (as long as it's bigger than the circle).
        m2_rejected = np.random.random(N_m2) > \
            (smoothing(this_m2 - m_min, delta_m) / smoothing(this_m1 - m_min, delta_m))
        i_accepted = np.where(np.logical_not(m2_rejected))
        if not len(i_accepted[0]):
            continue
        i_accepted_fullarray = i_this_m2_empty[0][i_accepted]
        m2_sample[i_accepted_fullarray] = this_m2[i_accepted]
        m2_empty[i_accepted_fullarray] = False
        # TODO: we could reuse this in order not to have to recompute i_this_m2_empty at
        # the beginning of the next iteration!!!
        # --> checked would improve like 10% only :(
    return m2_sample


# Parallelised implementation
# def sample_m2_cond_m1(m1_sample, verbose=False):
#     chunks = np.array_split(m1_sample, NUM_THREADS)
#     with Pool(NUM_THREADS) as pool:
#         fun = lambda i_and_chunk: \
#             (i_and_chunk[0], sample_m2_cond_m1_rejection(
#                 i_and_chunk[1], verbose=verbose))
#         res = pool.map(fun, list(enumerate(chunks)), chunksize=1)
#         pool._clear()
#     # ensure initial order
#     order, samples = list(zip(*res))
#     return np.concatenate([samples[i] for i in np.argsort(order)])


# Final choice of algorithms
def sample_m1_m2(N, m_range=_default_mass_params["m_range"],
                 lambda_peak=_default_mass_params["lambda_peak"],
                 alpha=_default_mass_params["alpha"],
                 sigma_m=_default_mass_params["sigma_m"],
                 mu_m=_default_mass_params["mu_m"],
                 m_perdecade=_default_mass_params["m_perdecade"],
                 beta_q=_default_mass_params["beta_q"],
                 delta_m=_default_mass_params["delta_m"]):
    m1_sample = sample_m1_invcdf(N, m_range=m_range, lambda_peak=lambda_peak, alpha=alpha,
                                 sigma_m=sigma_m, mu_m=mu_m, m_perdecade=m_perdecade,
                                 delta_m=delta_m)
    m2_sample = sample_m2_cond_m1_rejection(m1_sample, m_range=m_range, beta_q=beta_q,
                                            delta_m=delta_m)
    return m1_sample, m2_sample
