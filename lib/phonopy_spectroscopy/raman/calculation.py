# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""This model provides a simplified API for combining Gamma-point phonon
and Raman calculations to generate simulated spectra through a
high-level `RamanCalculation` object."""


# -------
# Imports
# -------


import warnings

import numpy as np

from ..constants import ZERO_TOLERANCE
from ..phonon import GammaPhonons
from ..units import nm_to_ev

from .instrument import Polarisation

from .intensity import (
    calculate_single_crystal_raman_intensities,
    calculate_powder_raman_intensities,
)

from .spectrum import RamanSpectrum
from .tensors import RamanTensors

from ..utility.geometry import (
    rotation_matrix_from_vectors,
    rotation_matrix_from_axis_angle,
)

from ..utility.numpy_helper import (
    np_readonly_view,
    np_asarray_copy,
    np_check_shape,
    np_expand_dims,
)


# ----------------------
# RamanCalculation class
# ----------------------


class RamanCalculation:
    """Combine `GammaPhonons` and `RamanTensors` objects for a complete
    Raman calculation to generate simulated spectra."""

    def __init__(self, gamma_ph, r_t, band_inds=None):
        """Create a new instance of the `RamanCalculation` class.

        Parameters
        ----------
        gamma_ph : GammaPhonons
            Gamma-point phonon calculation.
        r_t : RamanTensors
            Raman tensors.
        band_inds : array_like or None, optional
            Band indices for which Raman tensors were calculated
            (default: all bands).
        """

        # Check/set band indices used in calculations and update irreps
        # for included indices if required.

        irreps = None

        if band_inds is not None:
            band_inds = np_asarray_copy(band_inds, dtype=int)

            if not np_check_shape(band_inds, (None,)):
                raise ValueError(
                    "If supplied, band_inds must be an array_like with "
                    "shape (N,)."
                )

            if len(band_inds) != len(set(band_inds)):
                raise ValueError(
                    "One or more indices in band_inds are duplicates."
                )

            for idx in band_inds:
                if idx < 0 or idx >= gamma_ph.num_modes:
                    raise ValueError(
                        "One or more indices in band_indices are "
                        "incompatible with the number of modes in the "
                        "phonon calculation."
                    )

            if gamma_ph.has_irreps:
                irreps = gamma_ph.irreps.get_subset(
                    band_inds=band_inds, reset_inds=True
                )

        else:
            band_inds = np.array(list(range(gamma_ph.num_modes)), dtype=int)

        self._gamma_ph = gamma_ph
        self._r_t = r_t

        self._band_inds = band_inds
        self._irreps = irreps

    def _get_calc_params(self, geom, i_pols, s_pols, lw, w):
        """Determine parameters for Raman calculations.

        Parameters
        ----------
        geom : Geometry
            Measurement geometry.
        i_pols, s_pols : str, Polarisation or array_like
            Incident and scattered light polarisation(s). `s_pol` may be
            specified by one of {'parallel', 'cross', 'sum'}.
        lw : float or None
            Uniform linewidth or scale factor for calculated linewidths,
            depending on whether calculation has linewidths.
        w : float or None
            Measurement wavelength (default: None).

        Returns
        -------
        params : dict
            Dictionary of calculation parameters with the following: {
                'geometry': `Geometry`,
                'incident_polarisations': numpy.ndarray,
                'scattered_polarisations': numpy.ndarray,
                'single_polarisation': bool,
                'linewidths': numpy.ndarray,
                'raman_tensors': numpy.ndarray
                }
        """

        params = {}

        # Determine polarisations.

        i_pols, i_pols_n_dim_add = np_expand_dims(
            np.asarray(i_pols, dtype=object), (None,)
        )

        s_pols_str = str(s_pols).lower()

        if s_pols_str == "parallel":
            s_pols = i_pols
        elif s_pols_str == "cross":
            s_pols = Polarisation.cross_to(i_pols, geom.incident_direction)
        elif s_pols_str == "sum":
            s_pols = Polarisation.sum_parallel_cross_to(
                i_pols, geom.incident_direction
            )
        else:
            s_pols, s_pols_n_dim_add = np_expand_dims(
                np.asarray(s_pols, dtype=object), (None,)
            )

            i_pols, s_pols = np.broadcast_arrays(i_pols, s_pols)

        params["geometry"] = geom
        params["incident_polarisations"] = i_pols
        params["scattered_polarisations"] = s_pols

        # Keep track of whether the calculation is working with a single
        # or multiple polarisations.

        params["single_polarisation"] = (
            i_pols_n_dim_add != 0 and s_pols_n_dim_add != 0
        )

        # Determine linewidths.

        lws = None

        if lw is None:
            # Default value.

            lw = 1.0 if self._gamma_ph.linewidths is not None else 0.5
        else:
            if lw < ZERO_TOLERANCE:
                raise ValueError("lw cannot be zero or negative.")

        if self._gamma_ph.linewidths is not None:
            lws = lw * self._gamma_ph.linewidths[self._band_inds]
        else:
            lws = [lw] * len(self._band_inds)

        params["linewidths"] = lws

        # Determine Raman tensors.

        rt_e = 0.0

        if w is not None:
            if w < 0.0:
                raise ValueError("Measurment wavelengths cannot be negative.")

            if self._r_t.is_energy_dependent:
                rt_e = nm_to_ev(w)
            else:
                warnings.warn(
                    "A measurement wavelength was specified but the "
                    "calculation is not using energy-dependent Raman "
                    "tensors. Raman intensities will be calculated "
                    "using the far from resonance approximation and "
                    "the wavelength will only be used to apply "
                    "intensity modulation.",
                    RuntimeWarning,
                )

        params["raman_tensors"] = self._r_t.get_tensors_at_energy(rt_e)

        return params

    @property
    def frequencies(self):
        """numpy.ndarray : Band frequencies for the subset of bands in
        the calculation."""
        return self._gamma_ph.frequencies[self._band_inds]

    @property
    def linewidths(self):
        """numpy.ndarrray : Linewidths for the subset of bands in the
        calculation."""
        lws = self._gamma_ph.linewidths
        return lws[self._band_inds] if lws is not None else None

    @property
    def irreps(self):
        """Irreps : Irreps for the subset of bands in the calculation."""

        # If the calculation is for a subset of the bands, and the
        # underlying phonon calculation has irreps, the internal
        # _irreps field will be set. If _irreps is None, then return the
        # irreps from the phonon calculation.

        return (
            self._irreps if self._irreps is not None else self._gamma_ph.irreps
        )

    @property
    def band_indices(self):
        """numpy.ndarray : Indices of the bands in the calculation."""
        return np_readonly_view(self._band_inds.view())

    @property
    def structure(self):
        """Structure : Underlying `Structure` object."""
        return self._gamma_ph.structure

    @property
    def gamma_phonons(self):
        """GammaPhonons : Underlying `GammaPhonons` object."""
        return self._gamma_ph

    @property
    def raman_tensors(self):
        """RamanTensors : Underlying `RamanTensors` object."""
        return self._r_t

    def single_crystal(
        self,
        hkl,
        geom,
        i_pol,
        s_pol,
        rot=None,
        w=None,
        t=None,
        lw=None,
        spectrum_type="stokes",
        x_range=None,
        x_res=None,
        x_units="thz",
    ):
        """Simulate a single-crystal Raman measurement.

        Parameters
        ----------
        hkl : tuple of int
            Miller index of the crystal surface to orient along the
            incident direction.
        geom : Geometry
            Measurement geometry.
        i_pol, s_pol : str, Polarisation or list of Polarisation
            Polarisation(s) of incident and scattered light. The
            scattered polarisation may also be specified by one of
            {'parallel', 'cross', 'sum'}.
        rot : array_like or None, optional
            Rotation matrix or set of matrices to realign crystal after
            the `hkl` rotation.
        w, t : float or None, optional
            Measurement wavelength and temperature.
        lw : float or None, optional
            Uniform linewidth or scale factor for calculated linewidths
            (defaults: 0.5 THz uniform linewidth or scale factor of 1.0),
            depending on whether calculation has linewidths.
        spectrum_type : {'stokes', 'anti-stokes', 'both'}, optional
            Type of spectrum (default: `'stokes'`).
        x_range : tuple of float or None, optional
            Range of spectrum as a `(min, max)` tuple, or `None` to
            automatically determine a suitable range (default: `None`).
        x_res : float or None, optional
            Resolution of spectrum or `None` to automatically determine
            a suitable resolution (default: `None`).
        x_units : str, optional
            Frequency units of spectrum (default: `'thz'`).

        Returns
        -------
        spectrum : RamanSpectrum
            Simulated Raman spectrum.

        Notes
        -----
        * If both `i_pols` and `s_pols` are single polarisations, and
            `rots` is either None or a single rotation, the function
            returns a 1D spectrum.
        * If `i_pols` and/or `s_pols`, or `rots`, specify multiple
            values, the function returns an 2D spectrum.
        * A single function call cannot combine multiple polarisations
            and multiple rotations.
        """

        params = self._get_calc_params(geom, i_pol, s_pol, lw, w)

        single_int = params["single_polarisation"]

        if rot is not None:
            rot, n_dim_add = np_expand_dims(np.asarray(rot), (None, 3, 3))

            if not rot.shape[0] > 1 and not params["single_polarisation"]:
                raise Exception(
                    "Multiple incident/scattered polarisations cannot be "
                    "combined with multiple crystal rotations."
                )

            single_int = single_int and n_dim_add > 0

        r = rotation_matrix_from_vectors(
            self._gamma_ph.structure.real_space_normal(hkl),
            -1.0 * geom.incident_direction,
        )

        if rot is not None:
            temp = np.zeros_like(rot)

            for i, r_2 in enumerate(rot):
                temp[i] = np.dot(r, r_2)
        else:
            rot = np.reshape(r, (1, 3, 3))

        ints = None

        if len(rot) == 1:
            # Single rotation, potentially with multiple polarisations.

            ints = np.zeros(
                (
                    len(params["incident_polarisations"]),
                    params["raman_tensors"].shape[0],
                ),
                dtype=np.float64,
            )

            for i, (i_pol, s_pol) in enumerate(
                zip(
                    params["incident_polarisations"],
                    params["scattered_polarisations"],
                )
            ):
                ints[i] = calculate_single_crystal_raman_intensities(
                    params["raman_tensors"],
                    params["geometry"],
                    i_pol,
                    s_pol,
                    rot[0],
                )
        else:
            # Single polarisation, potentially with multiple rotations.

            ints = np.zeros(
                (rot.shape[0], params["raman_tensors"].shape[0]),
                dtype=np.float64,
            )

            for i, r in enumerate(rot):
                ints[i] = calculate_single_crystal_raman_intensities(
                    params["raman_tensors"],
                    params["geometry"],
                    i_pol,
                    s_pol,
                    r,
                )

        ints = np.array(ints, dtype=np.float64).T

        if single_int:
            ints = ints[:, 0].reshape((-1,))

        return RamanSpectrum(
            self.frequencies,
            ints,
            params["linewidths"],
            self.irreps,
            w=w,
            t=t,
            spectrum_type=spectrum_type,
            x_range=x_range,
            x_res=x_res,
            x_units=x_units,
        )

    def single_crystal_polarisation_rotation(
        self,
        hkl,
        geom,
        i_pol,
        s_pol,
        chi_start=0.0,
        chi_end=360.0,
        chi_step=2.5,
        **kwargs
    ):
        """Simulate a single-crystal polarisation (chi) rotation
        measurement where one of the incident or scattered polarisations
        are rotated about the incident or collection direction
        respectively.

        Parameters
        ----------
        hkl : tuple of int
            Miller index of the crystal surface to orient along the
            incident direction.
        geom : Geometry
            Measurement geometry.
        i_pol, s_pol : str or Polarisation
            Polarisation of incident and scattered light. The
            polarisation to be rotated can be specified by 'rot'. The
            scattered polarisation may also be specified by one of
            {'parallel', 'cross', 'sum'}.
        chi_start, chi_end, chi_step : float, optional
            Start/end angle and angle step for polarisation rotation in
            degrees (defaults: 0 -> 360 deg in 2.5 deg steps).
        **kwargs : any
            Optional arguments to `single_crystal`.

        Returns
        -------
        spectrum : RamanSpectrum
            Simulated (2D) Raman spectrum.

        See Also
        --------
        single_crystal : Simulate a single-crystal measurement.

        Notes
        -----
        See `single_crystal` for optional keyword arguments.
        """

        i_pol_str = str(i_pol).lower()
        s_pol_str = str(s_pol).lower()

        if i_pol_str == "rot":
            if s_pol_str == "rot":
                raise Exception("i_pol and s_pol cannot both be set to 'rot'.")

            i_pol = Polarisation.from_rotation(
                geom.incident_direction, chi_start, chi_end, chi_step
            )

        if s_pol_str == "rot":
            s_pol = Polarisation.from_rotation(
                geom.collection_direction, chi_start, chi_end, chi_step
            )

        return self.single_crystal(hkl, geom, i_pol, s_pol, **kwargs)

    def single_crystal_crystal_rotation(
        self,
        hkl,
        geom,
        i_pol,
        s_pol,
        phi_start=0.0,
        phi_end=360.0,
        phi_step=2.5,
        rot_axis="incident",
        **kwargs
    ):
        """Simulate a single-crystal crystal (phi) rotation measurement
        where the crystal is rotated about the incident direction.

        Parameters
        ----------
        hkl : tuple of int
            Miller index of the crystal surface to orient along the
            incident direction.
        geom : Geometry
            Measurement geometry.
        i_pol, s_pol : str or Polarisation
            Polarisation(s) of incident and scattered light. The
            scattered polarisation may also be specified by one of
            {'parallel', 'cross', 'sum'}.
        phi_start, phi_end, phi_step : float, optional
            Start/end angle and angle step for crystal rotation in
            degrees (defaults: 0 -> 360 deg in 2.5 deg steps).
        rot_axis : {'incident', 'collection'}, optional
            Axis to rotate around (default: 'incident')
        **kwargs : any
            Optional arguments to `single_crystal`.

        Returns
        -------
        spectrum : RamanSpectrum
            Simulated (2D) Raman spectrum.

        See Also
        --------
        single_crystal : Simulate a single-crystal measurement.

        Notes
        -----
        See `single_crystal` for optional keyword arguments. Note that
        specifying the `rot` keyword is invalid and will raise an error.
        """

        if "rot" in kwargs and kwargs["rot"] is not None:
            raise ValueError(
                "The rot keyword to single_crystal is not valid for a "
                "crystal-rotation experiment - specify the required "
                "rotation(s) with phi_start, phi_end and phi_step "
                "instead."
            )

        rot_axis = str(rot_axis).lower()

        if rot_axis == "incident":
            rot_axis = geom.incident_direction
        elif rot_axis == "collection":
            rot_axis = geom.collection_direction
        else:
            raise ValueError("Unknown rot_axis = '{0}'.".format(rot_axis))

        angles = np.arange(phi_start, phi_end + phi_step / 10.0, phi_step)

        if len(angles) == 0:
            raise ValueError(
                "No angles between phi_start = {0:.2f} -> phi_end = "
                "{1:.2f} with phi_step = {2:.2f}."
                "".format(phi_start, phi_end, phi_step)
            )

        rots = np.array(
            [rotation_matrix_from_axis_angle(rot_axis, a) for a in angles],
            dtype=np.float64,
        )

        return self.single_crystal(hkl, geom, i_pol, s_pol, rot=rots, **kwargs)

    def powder(
        self,
        geom,
        i_pol,
        s_pol,
        pref_orient_hkl=None,
        pref_orient_eta=0.0,
        w=None,
        t=None,
        lw=None,
        spectrum_type="stokes",
        x_range=None,
        x_res=None,
        x_units="thz",
        method="best",
        lebedev_prec=5,
    ):
        """Simulate a powder Raman spectrum with optional preferred
        orientation.

        Parameters
        ----------
        geom : Geometry
            Measurement geometry.
        i_pol, s_pol : str, Polarisation or list of Polarisation
            Polarisation(s) of incident and scattered light. The
            scattered polarisation may also be specified by one of
            {'parallel', 'cross', 'sum'}.
        pref_orient_hkl : tuple of int or None, optional
            Miller index of the preferred orientation (default: None).
        pref_orient_eta : float, optional
            Fraction of crystallites with the preferred orientation
            (default: 0.0)
        w, t : float or None, optional
            Measurement wavelength and temperature.
        lw : float, optional
            Uniform linewidth or scale factor for calculated linewidths
            (defaults: 0.5 THz uniform linewidth or scale factor of 1.0),
            depending on whether calculation has linewidths.
        spectrum_type : {'stokes', 'anti-stokes', 'both'}, optional
            Type of spectrum (default: `'stokes'`).
        x_range : tuple of float or None, optional
            Range of spectrum as a `(min, max)` tuple, or `None` to
            automatically determine a suitable range (default: `None`).
        x_res : float or None, optional
            Resolution of spectrum or `None` to automatically determine
            a suitable resolution (default: `None`).
        x_units : str, optional
            Frequency units of spectrum (default: `'thz'`).
        method : {'nquad', 'lebedev+circle', 'best'}, optional
            Method for powder averaging (default: `'best'`).
        lebedev_prec : int, optional
            Precision of the Lebedev + circle numerical quadrature
            scheme (default: 5).
        """

        params = self._get_calc_params(geom, i_pol, s_pol, lw, w)

        if method.lower() == "best":
            # method = 'best' will use an analytical formula if
            # possible. If multiple incident polarisations are
            # supplied and could result in a mix of analytical and
            # numerical methods being used, raise a warning.

            may_use_analytical = (
                np.abs(pref_orient_eta) < ZERO_TOLERANCE
                and not np.iscomplex(params["raman_tensors"]).any()
            )

            if may_use_analytical:
                count = 0

                c_dir = params["geometry"].collection_direction

                for p in params["incident_polarisations"]:
                    if p.check_perpendicular(c_dir):
                        count += 1

                if count < len(params["incident_polarisations"]):
                    warnings.warn(
                        "The given set of incident polarisatios may "
                        "result in a mix of calculations with "
                        "analytical and numerical methods with method "
                        "= 'best'. To avoid this warning, set method = "
                        "'lebedev+circle' instead.",
                        RuntimeWarning,
                    )

        pref_orient_surf_norm = None

        if pref_orient_hkl is not None:
            pref_orient_surf_norm = self._gamma_ph.structure.real_space_normal(
                pref_orient_hkl
            )

        ints = np.zeros(
            (
                len(params["incident_polarisations"]),
                params["raman_tensors"].shape[0],
            ),
            dtype=np.float64,
        )

        for i, (i_pol, s_pol) in enumerate(
            zip(
                params["incident_polarisations"],
                params["scattered_polarisations"],
            )
        ):
            ints[i] = calculate_powder_raman_intensities(
                params["raman_tensors"],
                params["geometry"],
                i_pol,
                s_pol,
                pref_orient_eta=pref_orient_eta,
                pref_orient_surf_norm=pref_orient_surf_norm,
                method=method,
                lebedev_prec=lebedev_prec,
            )

        ints = np.array(ints, dtype=np.float64).T

        if params["single_polarisation"]:
            ints = ints[:, 0].reshape((-1,))

        return RamanSpectrum(
            self.frequencies,
            ints,
            params["linewidths"],
            self.irreps,
            w=w,
            t=t,
            spectrum_type=spectrum_type,
            x_range=x_range,
            x_res=x_res,
            x_units=x_units,
        )

    def powder_polariation_rotation(
        self,
        geom,
        i_pol,
        s_pol,
        chi_start=0.0,
        chi_end=360.0,
        chi_step=None,
        **kwargs
    ):
        """Simulate a powder polarisation (chi) rotation measurement
        where one of the incident or scattered polarisations are rotated
        about the incident or collection direction respectively.

        Parameters
        ----------
        geom : Geometry
            Measurement geometry.
        i_pol, s_pol : str or Polarisation
            Polarisation of incident and scattered light. The
            polarisation to be rotated can be specified by 'rot'. The
            scattered polarisation may also be specified by one of
            {'parallel', 'cross', 'sum'}.
        chi_start, chi_end : float, optional
            Start/end angle for polarisation rotation in degrees
            (defaults: 0 -> 360 deg).
        chi_step : float or None, optional
            Angle step for polarisation rotation in degrees (defaults:
            2.5 deg, or 22.5 deg with preferred orientation).
        **kwargs : any
            Optional arguments to `powder`.

        Returns
        -------
        spectrum : RamanSpectrum
            Simulated (2D) Raman spectrum.

        See Also
        --------
        powder : Simulate a powder measurement.

        Notes
        -----
        See `powder` for optional keyword arguments.
        """

        if chi_step is None:
            chi_step = 2.5

            if "pref_orient_eta" in kwargs and kwargs["pref_orient_eta"] > 0.0:
                chi_step = 22.5

        i_pol_str = str(i_pol).lower()
        s_pol_str = str(s_pol).lower()

        if i_pol_str == "rot":
            if s_pol_str == "rot":
                raise Exception("i_pol and s_pol cannot both be set to 'rot'.")

            i_pol = Polarisation.from_rotation(
                geom.incident_direction, chi_start, chi_end, chi_step
            )

        if s_pol_str == "rot":
            s_pol = Polarisation.from_rotation(
                geom.collection_direction, chi_start, chi_end, chi_step
            )

        return self.powder(geom, i_pol, s_pol, **kwargs)

    def to_dict(self):
        """Return the internal data as a dictionary of native Python
        types for serialisation.

        Returns
        -------
        d : dict
            Dictionary structure containing internal data as native
            Python types.
        """

        return {
            "gamma_phonons": self._gamma_ph.to_dict(),
            "raman_tensors": self._r_t.to_dict(),
            "band_indices": self._band_inds.tolist(),
        }

    @staticmethod
    def from_dict(d):
        """Create a new `RamanCalculation` instance from a dictionary
        generated by `RamanCalculation.to_dict()`.

        Parameters
        ----------
        d : dict
            Dictionary generated by `to_dict()`.

        Returns
        -------
        calc : RamanCalculator
            `RamanCalculator` object constructed from the data in `d`.
        """

        return RamanCalculation(
            GammaPhonons.from_dict(d["gamma_phonons"]),
            RamanTensors.from_dict(d["raman_tensors"]),
            d["band_indices"],
        )
