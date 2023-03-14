"""
Generates and manages populations.
"""

import os
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from warnings import warn

import extrapops.constants as const
from extrapops.cosmology import _default_cosmo, luminosity_distance_to_redshift
from extrapops.redshifts import sample_z, _default_redshift_params
from extrapops.mass import chirp_mass, sample_m1_m2, _default_mass_params
from extrapops.spin import sample_spin, _default_spin_params
from extrapops.background import char_strain_sq_single_at_f1Hz, avg_strain_squared, \
    char_strain_squared_to_Omegah2, char_strain_sq_numerical, _warn_background
from extrapops.sobbh_waveform import frequency_at_t, t_of_frequency
from extrapops.snr import snr_avg_inclination, snr_max_inclination
from extrapops.yaml import yaml_dump

_yaml_ext = ".yaml"
_hdf_ext = ".h5"
_LISA_ext = ".LISA.h5"
_hdf_key = "SOBBH"
# Columns that will not be saved: used just for caching derived values
_excl_cols = []

# Units and some comments on LISA-format tables
LISA_units = {"Redshift": "Unit",
              "Mass1": "SolarMass",  # "redshifted" mass: M * (1+z)
              "Mass2": "SolarMass",  # "redshifted" mass: M * (1+z)
              "InitialFrequency": "Hertz",  # detector frame
              "InBandTime": "Years",  # detector frame
              "EclipticLongitude": "Radian",
              "EclipticLatitude": "Radian",
              "Inclination": "Radian",
              "Polarization": "Radian",
              "InitialPhase": "Radian",
              "CoalTime": "Years",  # detector frame
              "Distance": "GigaParsec",  # Luminosity distance
              "Spin1": "Unit",
              "Spin2": "Unit",
              "AzimuthalAngleOfSpin1": "Radian",
              "AzimuthalAngleOfSpin2": "Radian"}

labels =  {"Redshift": r"Redshift ($z$)",
           "Mass1": r"$M_1\;[M_\odot]$",
           "Mass2": r"$M_2\;[M_\odot]$",
           "ChirpMass": r"$\mathcal{M}\;[M_\odot]$",
           "InitialFrequency": r"$f_\mathrm{in}\;[\mathrm{Hz}]$",
           "InBandTime": r"t_\mathrm{in}\;[\mathrm{yr}]$",
           "EclipticLongitude": r"Ecliptic Lon $[\mathrm{rad}]$",
           "EclipticLatitude": r"Ecliptic Lat $[\mathrm{rad}]$",
           "Inclination": r"Inclination $[\mathrm{rad}]$",
           "Polarization": r"Polarization $[\mathrm{rad}]$",
           "InitialPhase": r"Initial Phase $[\mathrm{rad}]$",
           "CoalTime": r"$\tau_c\;[\mathrm{yr}]$",
           "Distance": r"$d_\mathrm{L}\; [\mathrm{Gpc}]$",  # Luminosity distance
           "Spin1": r"a_1",
           "Spin2": r"a_2",
           "AzimuthalAngleOfSpin1": r"$\phi_1\;[\mathrm{rad}]$",
           "AzimuthalAngleOfSpin2": r"$\phi_2\;[\mathrm{rad}]$"}


def splitext_LISA(file_name):
    """Equivalent to os.path.splitext, but takes into account %r extension.""" % _LISA_ext
    if file_name.endswith(_LISA_ext):
        return file_name[:-len(_LISA_ext)], _LISA_ext
    else:
        return os.path.splitext(file_name)


class Population():
    """
    Class generating/loading and holding a SOBHB population.

    All quantities are stored in the *source* reference frame, as oppossed to the detector
    reference frame used by LISA (but taken into account when loading/saving in LISACode
    format).
    """

    def __init__(self, load=None, lazy=True,
                 cosmo=_default_cosmo, redshift=_default_redshift_params,
                 mass=_default_mass_params, spin=_default_spin_params,
                 # Old argument names (raises warning)
                 cosmo_params=None, redshift_params=None,
                 mass_params=None, spin_params=None):
        """
        Creates a realisation of a SOBHB population.
        """
        if load is not None:
            self._load(load)
        else:
            # The next block should eventually be deprecated
            if cosmo_params is not None:
                cosmo = cosmo_params
                warn("'cosmo_params' argument will be deprecated. Use simply 'cosmo'.")
            if redshift_params is not None:
                redshift = redshift_params
                warn("'redshift_params' argument will be deprecated. Use simply 'redshift'.")
            if mass_params is not None:
                mass = mass_params
                warn("'mass_params' argument will be deprecated. Use simply 'mass'.")
            if spin_params is not None:
                spin = spin_params
                warn("'spin_params' argument will be deprecated. Use simply 'spin'.")
            # End of deprecation block
            self._params = {"cosmo": deepcopy(_default_cosmo),
                            "redshift": deepcopy(_default_redshift_params),
                            "mass": deepcopy(_default_mass_params),
                            "spin": deepcopy(_default_spin_params)}
            self._params["cosmo"].update(deepcopy(cosmo))
            self._params["redshift"].update(deepcopy(redshift))
            self._params["mass"].update(deepcopy(mass))
            self._params["spin"].update(deepcopy(spin))
            self._generate(lazy=lazy)

    def _load(self, file_name):
        name, ext = splitext_LISA(file_name)
        err_msg = "Could not find a population saved as %r" % file_name
        try:
            with open(name + _yaml_ext, "r") as f:
                self._params = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(err_msg + "(.yaml not present)")
        is_native = os.path.isfile(name + _hdf_ext)
        is_LISA = os.path.isfile(name + _LISA_ext)
        if is_LISA and is_native:
            print("Both LISA and native populations exist. Prefering native.")
            is_LISA = False
        if is_native:
            name_hdf = name + _hdf_ext
            self._data = pd.read_hdf(name_hdf)
        elif is_LISA:
            name_hdf = name + _LISA_ext
            self._load_LISA(name_hdf)
        else:
            raise FileNotFoundError(err_msg + "(.h5 not present)")

    def _load_LISA(self, file_name_with_ext):
        """
        Load catalogue in LISA format, and converts to source reference frame.
        """
        try:
            import h5py as h5
        except ImportError:
            raise ImportError("Loading in LISA format needs h5py!")
        f = h5.File(file_name_with_ext)
        fields = f["H5LISA"]["GWSources"]["SOBBH"]
        self._data = pd.DataFrame({"Redshift": np.array(fields["Redshift"])})
        for key, values in fields.items():
            if key in ["Redshift", "SourceType"]:
                continue
            self._data[key] = self._to_source_refframe(key, values=np.array(values))
        f.close()

    def _generate(self, lazy=True):
        """
        Generates redshifts and masses. And the rest of the parameters if `lazy=True`.

        All quantities generated in source reference frame.
        """
        redshift_params = deepcopy(self._params["redshift"])
        z_sample = sample_z(cosmo_params=self._params["cosmo"], **redshift_params)
        Ndraws = len(z_sample)
        m1_sample, m2_sample = sample_m1_m2(Ndraws, **self._params["mass"])
        self._data = pd.DataFrame({
            "Redshift": z_sample, "Mass1": m1_sample, "Mass2": m2_sample})
        # Not lazy: used for background computation
        # Notice that Distance is LuminosityDistance
        self._data["Distance"] = luminosity_distance_to_redshift(
            self.z, z_range=self._params["redshift"]["z_range"],
            z_perdecade=self._params["redshift"]["z_perdecade"],
            **self._params["cosmo"]) * 1e-3
        # NB: time-to-coalescence in the input parameters is given in detector frame,
        #     so it needs to be redshifted to source frame,
        #     where it is actually uniformly drawn from.
        self._data["CoalTime"] = (np.random.random(len(self)) *
                                  self._params["redshift"]["T_yr"] / (1 + self.z))
        # Frequencies and in-band time -- deterministic!
        # NB: there is some redundancy, specially when clipping, in the computation of the
        #     triad of (init f, final f, in-band t). But clearer and more robust this way.
        # TODO: these could be even more lazily computed, just on-demand, to save memory
        # TODO: these quantities involve experiment properties
        #       --> make these properties params of the population instance!
        self._data["InitialFrequency"] = \
            frequency_at_t(self.M * const.Msun_kg, self._data["CoalTime"] * const.yr_s, 0)
        self.add_final_freq_and_in_band_time()
        if not lazy:
            self._lazy_generate()

    def add_final_freq_and_in_band_time(self, force=False):
        """
        Adds the derived quantities `FinalFrequency` and `InBandTime`.

        If present, they are overwritten when `force=True` (default `False`).
        """
        # TODO: Kick out irrelevant events either
        #       - before computing InBandTime, using final-initial freq ~ 0 (saves memory)
        #       - using InBandTime < sampling delta time of experiment
        #       In any case, really few events are discarded.
        is_computed = ("FinalFrequency" in self._data.columns and
                       "InBandTime" in self._data.columns)
        if is_computed and not force:
            return
        # Final frequency can be achieved during observation time if Tobs > Tcoal!
        T_evol = np.clip(const.LISA_T_obs_yr / (1 + self.z), None, self._data["CoalTime"])
        self._data["FinalFrequency"] = \
            frequency_at_t(self.M * const.Msun_kg, self._data["CoalTime"] * const.yr_s,
                           T_evol * const.yr_s)
        # Initial and Final frequencies to account for experiment's bandwidth
        # (redshifted to each individual event)
        # In practice, for LISA and the populations usually generated, the clipping to the
        # left of the bandwidth will never apply (but will on the right)
        LISA_bandwith_source_frame = [const.LISA_bandwidth[0] * (1 + self.z),
                                      const.LISA_bandwidth[1] * (1 + self.z)]
        self._data["InitialFrequency"] = np.clip(
            self._data["InitialFrequency"], *LISA_bandwith_source_frame)
        self._data["FinalFrequency"] = np.clip(
            self._data["FinalFrequency"], *LISA_bandwith_source_frame)
        self._data["InBandTime"] = (
            t_of_frequency(self.M * const.Msun_kg, self._data["CoalTime"] * const.yr_s,
                           self._data["FinalFrequency"]) -
            t_of_frequency(self.M * const.Msun_kg, self._data["CoalTime"] * const.yr_s,
                           self._data["InitialFrequency"])) / const.yr_s

    def _lazy_generate(self):
        """
        Generates the rest of the parameters: spins, initial freq, etc.
        """
        # Make sure that they are not requested again.
        self._lazy_generated = True
        # Spins
        self._data["Spin1"], self._data["Spin2"], cos1, cos2 = \
            sample_spin(len(self), **self._params["spin"])
        self._data["AzimuthalAngleOfSpin1"] = np.arccos(cos1)
        self._data["AzimuthalAngleOfSpin2"] = np.arccos(cos2)
        # History
        self._data["InitialPhase"] = np.random.random(len(self)) * 2 * np.pi
        # Position
        # From the MLDC manual:
        # SSB Latitude beta = pi/2 - theta; Isotropic: Uniform in cosine!
        # SSB Longitude lambda = phi; Isotropic: Uniform
        self._data["EclipticLatitude"] = \
            np.arccos(2 * np.random.random(len(self)) - 1) - np.pi / 2
        # equiv to np.arcsin(2 * np.random.random(len(self)) - 1)
        self._data["EclipticLongitude"] = np.random.random(len(self)) * 2 * np.pi
        self._data["Inclination"] = \
            np.arccos(2 * np.random.random(len(self)) - 1)
        # Physical quantities, which depend on h^2, polarisation angle in [0, pi)
        # But strictly, pol. angle in [0, 2pi] (see  https://arxiv.org/pdf/2108.01167.pdf)
        self._data["Polarization"] = np.random.random(len(self)) * 2 * np.pi

    def __len__(self):
        """Number of events in the population."""
        return len(self._data)

    @property
    def params(self):
        """Parameters with which the population was generated. Returns a copy."""
        return deepcopy(self._params)

    @property
    def source_params(self):
        """Names of source parameters generated (so far, if `lazy=True`)."""
        return list(self._data.columns)

    def __getitem__(self, *args):
        """
        Direct access to the internal DataFrame.

        Returns views or copies as Pandas would do.
        """
        return self._data.__getitem__(*args)

    def _z(self, loc_arg=None):
        """Redshifts of individual events."""
        thisdata = self._data if loc_arg is None else self._data.loc[loc_arg]
        return thisdata["Redshift"].values

    @property
    def z(self):
        """Redshifts of individual events."""
        return self._z()

    def _m1(self, loc_arg=None):
        """Largest mass of individual events (source reference frame) in Msun units."""
        thisdata = self._data if loc_arg is None else self._data.loc[loc_arg]
        return thisdata["Mass1"].values

    @property
    def m1(self):
        """Largest mass of individual events (source reference frame) in Msun units."""
        return self._m1()

    def _m2(self, loc_arg=None):
        """Largest mass of individual events (source reference frame) in Msun units."""
        thisdata = self._data if loc_arg is None else self._data.loc[loc_arg]
        return thisdata["Mass2"].values

    @property
    def m2(self):
        """Largest mass of individual events (source reference frame) in Msun units."""
        return self._m2()

    def _M(self, loc_arg=None):
        """Chirp mass of individual events (source reference frame) in Msun units."""
        self._cache_chirp_mass()
        thisdata = self._data if loc_arg is None else self._data.loc[loc_arg]
        return thisdata["ChirpMass"].values

    @property
    def M(self):
        """Chirp mass of individual events (source reference frame) in Msun units."""
        return self._M()

    def _cache_chirp_mass(self):
        """Creates chirp mass column in the DataFrame."""
        if "ChirpMass" not in self._data:
            self._data["ChirpMass"] = chirp_mass(self.m1, self.m2)

    def _to_detector_refframe(self, k, values=None):
        if values is None:
            if k == "ChirpMass":
                self._cache_chirp_mass()
            values = self._data[k].to_numpy()
        if k in ["InitialFrequency", "FinalFrequency"]:
            return values / (1 + self.z)
        elif k in ["CoalTime", "InBandTime"]:
            return values * (1 + self.z)
        # Obviously masses don't redshift, but since what's observed it "redshifted"
        # chirp mass, we redshift them
        elif k in ["Mass1", "Mass2", "ChirpMass"]:
            return values * (1 + self.z)
        elif k in LISA_units:  # non-redshifting quantities
            return values
        elif k.lower().startswith("snr"):
            return values
        else:
            raise NotImplementedError(
                "Detector reference frame not implemented for %r" % k)

    def _to_source_refframe(self, k, values=None):
        if values is None:
            if k == "ChirpMass":
                self._cache_chirp_mass()
            values = self._data[k].to_numpy()
        if k in ["InitialFrequency", "FinalFrequency"]:
            return values * (1 + self.z)
        elif k in ["CoalTime", "InBandTime"]:
            return values / (1 + self.z)
        # Obviously masses don't redshift, but since what's observed it "redshifted"
        # chirp mass, we redshift them
        elif k in ["Mass1", "Mass2", "ChirpMass"]:
            return values / (1 + self.z)
        elif k in LISA_units:  # non-redshifting quantities
            return values
        elif k.lower().startswith("snr"):
            return values
        else:
            raise NotImplementedError(
                "Source reference frame not implemented for %r" % k)

    def save(self, file_name, force=False, LISA=False):
        """
        Save the population to a pair of files .yaml and .h5, containing respectively
        the parameters of the population and the population events.

        If `LISA=True`, it is saved in LISACode's format (LISACode needed).
        """
        if not getattr(self, "_lazy_generated", False):
            print("Some parameters have not yet been generated. Generating them now...")
            self._lazy_generate()
        name, ext = splitext_LISA(file_name)
        is_LISA = LISA or ext == _LISA_ext
        name_hdf = name + _LISA_ext if is_LISA else name + _hdf_ext
        if os.path.isfile(name_hdf):
            if force:
                os.remove(name_hdf)
            else:
                raise ValueError(
                    "File exists and will not be overwritten unless `force=True` passed.")
        path = os.path.dirname(os.path.realpath(name))
        if os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        with open(name + _yaml_ext, "w") as f:
            yaml_dump(self._params, f)
        if is_LISA:
            self._save_LISA(name_hdf)
        else:
            data_columns = [col for col in self._data.columns if col not in _excl_cols]
            self._data.to_hdf(name_hdf, key=_hdf_key, mode="w", data_columns=data_columns)

    def _save_LISA(self, name):
        """Save into LISACode format."""
        try:
            from LISAhdf5 import LISAhdf5, ParsUnits
        except ImportError:
            raise ImportError("Load/save in LISA format needs LISACode!")
        LH = LISAhdf5(name)
        pars_units = ParsUnits()
        for k, unit in LISA_units.items():
            pars_units.addPar(k, self._to_detector_refframe(k), unit)
        pars_units.addPar("SourceType", _hdf_key, "name")
        LH.addSource(_hdf_key, pars_units, overwrite=True)

    def data_in_detector_refframe(self):
        """Returns a copy of the data in the detector reference frame."""
        new_data = self._data.copy(deep=True)
        for col in new_data.columns:
            new_data[col] = self._to_detector_refframe(col)
        return new_data

    def add_snr_avg_inclination(self, T_obs_yr=const.LISA_T_obs_yr, use_cache=True):
        """
        Adds a column `snr_avg_inclination` with an approximate calculation of the SNR,
        averaging over possible inclinations of the given event.
        """
        self.add_final_freq_and_in_band_time()
        self._data["snr_avg_inclination"] = snr_avg_inclination(
            self._to_detector_refframe("ChirpMass") * const.Msun_kg,
            self._data["Distance"] * const.Gpc_m,  # lum distance
            self._to_detector_refframe("InitialFrequency"),
            self._to_detector_refframe("FinalFrequency"))

    def add_snr_max_inclination(self, T_obs_yr=const.LISA_T_obs_yr, use_cache=True):
        """
        Adds a column `snr_max_inclination` with an approximate calculation of the SNR,
        maximising over possible inclinations of the given event.
        """
        self.add_final_freq_and_in_band_time()
        self._data["snr_max_inclination"] = snr_max_inclination(
            self._to_detector_refframe("ChirpMass") * const.Msun_kg,
            self._data["Distance"] * const.Gpc_m,  # lum distance
            self._to_detector_refframe("InitialFrequency"),
            self._to_detector_refframe("FinalFrequency"))

    def total_strain_squared_avg(self, f=None, z_max=None, use_cached=True,
                                 omegah2_units=False, epsabs=1e-4, epsrel=1e-4):
        """
        Computes the average (realisation-independent) background as a numerical
        integration of the number density with the current population paramters.

        Returns unitless characteristic strain as a function of frequency (in Hz)
        that goes like f**(-4/3)

        Make `omega2_units=True` (default: False) to return the background in Omega h^2
        units, and in this case the frequency dependency is f**2/3.

        Use ``z_max`` to override the maximum redshift used for the integral, and thus
        get the actual value. The recommended value is ``z_max=5`` for a ``<1%``
        error with respect to ``z_max=inf``.

        ``epsabs`` and ``epsrel`` are precision arguments of ``scipy.integrate.[X]quad``.
        A value of ``1e-4`` (default) for both is enough to make the numerical error
        larger than the population realisation error (~0.5%), though larger values may be
        enough.

        Result is cached for subsequent calls.
        Pass ``use_cached=False`` (default: ``True``) to force recomputation.
        """
        _warn_background(max(self._params["redshift"]["z_range"][1], z_max or 0))
        if getattr(self, "hc2_1Hz_avg", False) and \
           z_max == getattr(self, "hc2_1Hz_avg_z_max", None) and use_cached:
            result = deepcopy(self.hc2_1Hz_avg)
        else:
            redshift_params = deepcopy(self._params["redshift"])
            if z_max is not None:
                redshift_params["z_range"][1] = z_max
            result = char_strain_sq_numerical(
                cosmo_params=self._params["cosmo"], redshift_params=redshift_params,
                mass_params=self._params["mass"], epsabs=epsabs, epsrel=epsrel)
            # Cache!
            self.hc2_1Hz_avg = deepcopy(result)
            self.hc2_1Hz_avg_z_max = deepcopy(z_max)
        conv = char_strain_squared_to_Omegah2 if omegah2_units else lambda f, h2: h2
        func = lambda f: conv(f, result * f**(-4 / 3))
        if f is None:
            return func
        else:
            return func(f)

    def total_strain_squared(self, f=None, use_cached=True, omegah2_units=False,
                             extra_factor=1):
        """
        Computes the expected background from the sum of individual strains.

        Returns unitless characteristic strain as a function of frequency (in Hz)
        that goes like f**(-4/3)

        Make `omega2_units=True` (default: False) to return the background in Omega h^2
        units, and in this case the frequency dependency is f**2/3.

        Result is cached for subsequent calls.
        Pass ``use_cached=False`` (default: ``True``) to force recomputation.

        ``extra_factor`` multiplies the integrand, for tests. It can be an array
        with one element per member of the population. Combine with ``use_cached=False``!
        """
        _warn_background(self._params["redshift"]["z_range"][1])
        if getattr(self, "hc2_1Hz", False) and use_cached and extra_factor != 1:
            result = deepcopy(self.hc2_1Hz)
        else:
            result = np.sum(char_strain_sq_single_at_f1Hz(
                # Remember that Distance is luminosity distance
                self.z, self.M * const.Msun_kg,
                self._data["Distance"] * const.Gpc_m / (1 + self.z),
                T_gen=self._params["redshift"]["T_yr"] * const.yr_s) * extra_factor)
            # Cache only if no extra factor given
            if isinstance(extra_factor, int) and extra_factor == 1:
                self.hc2_1Hz = deepcopy(result)
        conv = char_strain_squared_to_Omegah2 if omegah2_units else lambda f, h2: h2
        func = lambda f: conv(f, result * f**(-4 / 3))
        if f is None:
            return func
        else:
            return func(f)

    def total_strain_squared_fbin(self, delta_f, f_min=None, f_max=None, subsample=1,
                                  use="in", T_obs_yr=const.LISA_T_obs_yr,
                                  progress_bar=False, omegah2_units=False):
        """
        Computes the expected background from the sum of individual strains, binned over
        frequencies with bin width `delta_f` (e.g. the inverse
        of the chunk length, ~11 days for LISA).

        Assigns to each event its initial frequency if ``use="in"`` (default), or its
        final frequency if ``use="out"``, computed for an observation time in years of
        `T_obs_yr` (default: 4 years).

        Ignoring drifting is a good approximation at low frequencies only, for the
        purpose of computing the gaussian f^(-4/3) part of the background.

        Returns a tuple of (frequency binning, unitless strain).

        Make `omega2_units=True` (default: False) to return the background in Omega h^2
        units.

        The frequencies are the left-most ones per bin, and the left and rightmost bins
        correspond to the min/max frequency of the events in the population.

        To sum a sub-sample of the population, give subsample > 1 (int, default 1).
        """
        _warn_background(self._params["redshift"]["z_range"][1])
        # Proposed by Alberto Sesana
        # - Create freq grid with delta_f = 1/T_chunk
        # - Per source, add to the bin of its f_in the amount h^2 * f_bin * T_chunk
        #   where h^2 is computed using eq. (26) of Angelo's notes
        # Let us work in detector reference frame for the frequency,
        if use.lower() == "in":
            f_subsample = \
                self._data["InitialFrequency"][::subsample].to_numpy(np.float64).copy()
        elif use.lower() == "out":
            self.add_final_freq_and_in_band_time()
            f_subsample = \
                self._data["FinalFrequency"][::subsample].to_numpy(np.float64).copy()
        else:
            raise ValueError("`use` must be 'in'|'out' for initial|final frequency.")
        # We need to REDSHIFT to detector reference frame BEFORE sorting,
        # since the sorting we see is that of the observed frequencies
        z = self.z[::subsample].copy()
        f_subsample /= (1 + z)  # --> detector ref. frame
        i_sorted = np.argsort(f_subsample)
        f_in_sorted = f_subsample[i_sorted]
        del f_subsample
        z_sorted = z[i_sorted]
        del z
        # chirp mass and comov distance in source ref frame
        chirp_mass_sorted = self.M[::subsample].copy()[i_sorted] * const.Msun_kg
        # Remember that Distance is luminosity distance
        comov_distance_sorted = \
            self._data["Distance"][::subsample].to_numpy(np.float64).copy()[i_sorted] * \
            const.Gpc_m / (1 + z_sorted)
        del i_sorted
        individual_h2s_sorted = avg_strain_squared(
            z_sorted, chirp_mass_sorted, comov_distance_sorted,
            f_in_sorted * (1 + z_sorted))
        del z_sorted, chirp_mass_sorted, comov_distance_sorted
        # Loop over evens, and per events over bins
        # Looping over bins remember last position, to make it faster.
        # TODO: the progress bar is not very accurate, bc the bins contain very different
        #       amounts of events
        if f_min is None:
            f_min = 0
        f_min = max(f_in_sorted[0], f_min) - delta_f / 2
        if f_max is None:
            f_max = np.inf
        f_max = min(f_in_sorted[-2], f_max) + delta_f / 2
        freqs = np.arange(f_min, f_max, delta_f)
        strains_sq_unitless = np.zeros(len(freqs))
        i_source = 0
        enum_iter_wrap = \
            lambda x: (tqdm(enumerate(x), total=len(x)) if progress_bar else enumerate(x))
        for i_bin, f_bin_left in enum_iter_wrap(freqs):
            try:
                i_source_first_next_bin = next(
                    j for j, f_in_source in enumerate(f_in_sorted[i_source:])
                    if f_in_source > f_bin_left + delta_f) + i_source
                strains_sq_unitless[i_bin] = \
                    sum(individual_h2s_sorted[i_source:i_source_first_next_bin])
                i_source = i_source_first_next_bin
            except StopIteration:
                strains_sq_unitless[i_bin] = sum(individual_h2s_sorted[i_source:])
        strains_sq_unitless *= (freqs + 0.5 * delta_f) / delta_f
        if omegah2_units:
            return (freqs, char_strain_squared_to_Omegah2(freqs, strains_sq_unitless))
        else:
            return (freqs, strains_sq_unitless)

    def plot(self, param, param2=None, color_by=None, ax=None, bins=40, cmap="viridis",
             frame="source"):
        """
        Plot distribution for (pairs of) source parameters.

        Use ``frame="source"`` (default) or ``"detector"`` to choose the reference frame
        of the quantities plotted (where applicable).

        If two parameters are specified, a 2d histogram will be produced, unless a
        ``color_by`` parameter is specified, in which case it will produce a scatter plot.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
        if not isinstance(param, str):
            raise TypeError("The first argument must be a parameter name as a string.")
        if param2 is not None and not isinstance(param2, str):
            raise TypeError(
                "The second argument needs must be a parameter name as a string."
            )
        if color_by is not None and not isinstance(color_by, str):
            raise TypeError(
                "'color_by' must be a parameter name as a string."
            )
        if not isinstance(frame, str) or frame.lower() not in ["source", "detector"]:
            print("'frame' must be 'source' (default) or 'detector'.")
        to_plot = {}
        for k, p in zip(["x", "y", "c"], [param, param2, color_by]):
            if not p:
                continue
            try:
                to_plot[k] = self._data[
                    next(c for c in self.source_params if p.lower() == c.lower())
                ]
            except StopIteration:
                raise ValueError(
                    f"Argument {p} is not a valid source parameter name. "
                    f"Try one of (case insensitive) {self.source_params}."
                )
            if frame.lower() == "detector":
                to_plot[k] = self._to_detector_refframe(p, to_plot[k])
        if param and param2:
            if color_by:
                plot = ax.scatter(
                    to_plot["x"],
                    to_plot["y"],
                    c=to_plot["c"],
                    marker=".",
                    alpha=0.5,
                    cmap=cmap,
                    rasterized=True
                )
                cbar = plt.colorbar(plot)
                cbar.ax.set_ylabel(labels.get(color_by, color_by))
            else:
                ax.hist2d(to_plot["x"], to_plot["y"], bins=bins, cmap=cmap)
            ax.set_xlabel(labels.get(param, param))
            ax.set_ylabel(labels.get(param2, param2))
        else:
            ax.hist(to_plot["x"], bins=bins)
            ax.set_xlabel(labels.get(param, param))
        return ax
