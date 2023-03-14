"""
Generators for the spin-related quantities.
"""

import numpy as np
from scipy import stats


# Fid values (median) from LIGO/Virgo O3 Populations paper
# (https://arxiv.org/abs/2111.03634)
_default_spin_params = {
    "expected_a": 0.25,  # +0.09 - 0.07
    "var_a": 0.03,  # +0.02 - 0.01
    "zeta_spin": 0.66,  # + 0.31 - 0.52
    "sigma_t": 1.5,  # +2.0 -0.8
}


# Function from Paolo
def beta_params_from_spin_amplitude_params(Expected_a, Var_a):
    expec_rel = (Expected_a / (1 - Expected_a))
    beta_a = ((expec_rel - Var_a * np.power(1. + expec_rel, 2.)) /
              (Var_a * np.power(1. + expec_rel, 3.)))
    alpha_a = expec_rel * beta_a
    return alpha_a, beta_a


def _frozen_pdf_spin_amplitude(expected_a=_default_spin_params["expected_a"],
                               var_a=_default_spin_params["var_a"]):
    alpha, beta = beta_params_from_spin_amplitude_params(expected_a, var_a)
    return stats.beta(alpha, beta)


def pdf_spin_amplitude(a, expected_a=_default_spin_params["expected_a"],
                       var_a=_default_spin_params["var_a"]):
    return _frozen_pdf_spin_amplitude(
        expected_a=expected_a, var_a=var_a).pdf(a)


def sample_spin_amplitude(N, expected_a=_default_spin_params["expected_a"],
                          var_a=_default_spin_params["var_a"]):
    return _frozen_pdf_spin_amplitude(
        expected_a=expected_a, var_a=var_a).rvs(N)


# Alignment #######

# NB: p(cos(theta)) = const is isotropic bc it takes into acount the
#    "amount of directions" per phi in the sphere, propto radius(theta) = cos(theta)
# The two spins tilts are correlated via the mixture parameter: both need to belong to
# the same population. Thus, we cannot simulate them both independently using the
# 1d-marginalised distributions.


def _frozen_pdf_spin_cos_truncnorm(sigma):
    # NB: a,b are defined in the loc-scale scale (see truncnorm docs)
    mean, std = 1, sigma
    clip_a, clip_b = -1, 1
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return stats.truncnorm(loc=mean, scale=std, a=a, b=b)


def pdf_2spins_cos(cosine1, cosine2, zeta_spin=_default_spin_params["zeta_spin"],
                   sigma_t=_default_spin_params["sigma_t"]):
    density_isotropic = 0.25  # = 1 / (1 - (-1))**2
    return ((1 - zeta_spin) * density_isotropic +
            zeta_spin * (_frozen_pdf_spin_cos_truncnorm(sigma=sigma_t).pdf(cosine1) *
                         _frozen_pdf_spin_cos_truncnorm(sigma=sigma_t).pdf(cosine2)))


def pdf_spin_cos_marg(cosine, zeta_spin=_default_spin_params["zeta_spin"],
                      sigma_t=_default_spin_params["sigma_t"]):
    density_isotropic = 0.5  # = 1 / (1 - (-1))
    return ((1 - zeta_spin) * density_isotropic +
            zeta_spin * _frozen_pdf_spin_cos_truncnorm(sigma=sigma_t).pdf(cosine))


# TODO: it may be faster to get np.argwhere and generate only the exact number of events
#       from the uniform and the truncnorm.
def sample_2spins_cos(N, zeta_spin=_default_spin_params["zeta_spin"],
                      sigma_t=_default_spin_params["sigma_t"]):
    component = np.random.random(N) > zeta_spin
    sample_isotropic_1 = 2 * np.random.random(N) - 1
    sample_isotropic_2 = 2 * np.random.random(N) - 1
    sample_truncgauss_1 = _frozen_pdf_spin_cos_truncnorm(sigma=sigma_t).rvs(N)
    sample_truncgauss_2 = _frozen_pdf_spin_cos_truncnorm(sigma=sigma_t).rvs(N)
    return (np.where(component, sample_isotropic_1, sample_truncgauss_1),
            np.where(component, sample_isotropic_2, sample_truncgauss_2))


# All together

def sample_spin(N, expected_a=_default_spin_params["expected_a"],
                var_a=_default_spin_params["var_a"],
                zeta_spin=_default_spin_params["zeta_spin"],
                sigma_t=_default_spin_params["sigma_t"]):
    """
    Sample N events. Returns amplitude1, amplitude2, cos1, cos2.
    """
    return (sample_spin_amplitude(N, expected_a=expected_a, var_a=var_a),
            sample_spin_amplitude(N, expected_a=expected_a, var_a=var_a),
            *sample_2spins_cos(N, zeta_spin=zeta_spin, sigma_t=sigma_t))
