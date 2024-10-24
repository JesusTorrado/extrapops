"""
Data and formulas about experiments.
"""

from warnings import warn
import dataclasses
import numpy as np
from scipy import integrate
from scipy.interpolate import UnivariateSpline

import extrapops.constants as const


@dataclasses.dataclass
class Experiment:
    """
    Relevant quantities and functions of particular probes.

    Sensitivities are given as sqrt of strain noise (1/sqrt(Hz) units). If given per
    channel, indicate the number of effective channels with ``n_channels_effective``
    (otherwise leave it as 1)
    """

    desc: str
    arm_length_m: float
    t_obs_yr_effective: float
    n_channels_effective: int
    bandwidth_Hz: [float, float]
    # Sensitivity single channel
    sensitivity_func: callable = dataclasses.field(default=None)
    sensitivity_table: [float, float] = dataclasses.field(default=None)
    sensitivity_interp_kwargs: dict = dataclasses.field(default=None)
    sensitivity_interp_f_scale: str = dataclasses.field(default="lin")
    _sensitivity_interpolator: callable = dataclasses.field(default=None)
    # Precision for the f-integral of the SNR, and placeholder for its interpolator
    f_integral_n: int = 5000
    _f_integral_acc_loglog_interpolator: callable = dataclasses.field(default=None)

    def __post_init__(self):
        """
        Checked that sensitivity has been given, and creates an interpolator if table
        passed.
        """
        if (
            self.sensitivity_func is None and
            self.sensitivity_table is None
        ):
            raise ValueError(
                "Please pass the sensitivity either as a function or a table.")
        if self.sensitivity_table is not None:
            table = np.atleast_2d(self.sensitivity_table)
            if len(table.shape) != 2 or table.shape[1] != 2:
                raise ValueError(
                    "If passing a table, it should have two columns: "
                    "frequencies and sensitivities (as sqrt of noise strain)"
                )
            if table[0][0] > self.bandwidth_Hz[0] or table[-1][0] < self.bandwidth_Hz[1]:
                warn(
                    "Sensitivity table does not expand to bandwidth limits. "
                    "Will extrapolate."
                )
            if self.sensitivity_interp_f_scale == "lin":
                self._sensitivity_interpolator = UnivariateSpline(
                    table[:, 0], table[:, 1], bbox=self.bandwidth_Hz,
                    **(self.sensitivity_interp_kwargs or {}))
                self.sensitivity_func = self._sensitivity_interpolator
            elif self.sensitivity_interp_f_scale == "log":
                self._sensitivity_interpolator = UnivariateSpline(
                    np.log(table[:, 0]), table[:, 1], bbox=np.log(self.bandwidth_Hz),
                    **(self.sensitivity_interp_kwargs or {}))
                self.sensitivity_func = \
                    lambda f: self._sensitivity_interpolator(np.log(f))
            else:
                raise ValueError("'sensitivity_table_f_scale' must be one of [lin|log].")
                    

    def _cache_f_integral(self):
        """
        Caches the f-integral of the SNR calculation.

        Since the integrand varies more smoothly in log-scale, it's log-sampled.
        """
        fs_integral = np.logspace(np.log10(self.bandwidth_Hz[0]),
                                  np.log10(self.bandwidth_Hz[1]), self.f_integral_n + 1)
        segment_integrals = np.empty(self.f_integral_n)
        f_integrand = lambda f: f**(-7 / 3) / self.sensitivity_func(f)**2
        for i in range(self.f_integral_n):
            segment_integrals[i] = integrate.quad(
                f_integrand, fs_integral[i], fs_integral[i + 1])[0]
        accumulated_integrals = [np.sum(segment_integrals[:i])
                                 for i in range(1, 1 + self.f_integral_n)]
        return UnivariateSpline(np.log(fs_integral[1:]), np.log(accumulated_integrals))

    @property
    def f_integral_acc_loglog_interpolator(self):
        """
        Interpolator for the f-integral of the SNR.

        In the first call after class initialisation, it will compute the interpolator
        and cache it.
        """
        if self._f_integral_acc_loglog_interpolator is None:
            self._f_integral_acc_loglog_interpolator = self._cache_f_integral()
        return self._f_integral_acc_loglog_interpolator



            
two_pi_cinv = 2 * np.pi / const.c_m_invs
LISA_arm_length = 2.5e9  # m
# Eqs (10), (13), (19) from https://arxiv.org/pdf/2108.01167.pdf
# using dimensionless strain (dnu/nu)

# LISA
LISA_omega_L = lambda f: two_pi_cinv * f * LISA_arm_length
LISA_Sacc = lambda f: (3e-15**2 / (2 * np.pi * f * const.c_m_invs)**2 *
                  (1 + (.4e-3 / f)**2) * (1 + (f / 8e-3)**4))
LISA_Soms = lambda f: (15e-12**2 * (two_pi_cinv * f)**2 *
                  (1 + (2e-3 / f)**4))
S_X = lambda f: (16 * np.sin(LISA_omega_L(f))**2 *
                 (LISA_Soms(f) + (3 + np.cos(2 * LISA_omega_L(f))) * LISA_Sacc(f)))
# TDI units (eq 56 same source):
#     S_{h,X} = 20 / 3 * (1 + 0.6 * (omega L)**2) * S_{X,1.5}
#                      / ((4 omega L)**2 * sin**2(omega L))
LISA_sens = lambda f: np.sqrt((20 / 3 * (1 + 0.6 * LISA_omega_L(f)**2) /
                               ((4 * LISA_omega_L(f))**2 * np.sin(LISA_omega_L(f))**2) *
                               S_X(f)))
# NB: that is for 1 channel. One needs to divide by 2 for the full TDI 1.5 with 6 links

            
LISA = Experiment(
    desc="LISA TDI 1.5 with effective t_obs = 4yr (analytical approx responses).",
    arm_length_m=LISA_arm_length,
    t_obs_yr_effective=4,
    n_channels_effective=2, # TDI 1.5
    bandwidth_Hz=[3e-5, 0.5],  # max is f_Nyquist = 1/2 * sampling_rate
    sensitivity_func=LISA_sens,
)
