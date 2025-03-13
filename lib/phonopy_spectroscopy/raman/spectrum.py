# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Class and associated routines for simulating Raman spectra."""


# -------
# Imports
# -------


import warnings

import pandas as pd
import numpy as np

from ..constants import (
    ZERO_TOLERANCE,
    AMU_TO_KG,
    PLANCK_CONSTANT_J,
    SPEED_OF_LIGHT,
)

from ..distributions import lorentzian, phonon_occupation_number
from ..spectrum import GammaPhononSpectrumBase
from ..units import nm_to_thz

from ..utility.numpy_helper import (
    np_asarray_copy,
    np_readonly_view,
    np_check_shape,
    np_expand_dims,
)


# ---------
# Constants
# ---------


_RAMAN_INTENSITY_PREFACTOR = (
    (1.0e10**2)
    * (2.0 * (np.pi**2))
    * (PLANCK_CONSTANT_J / (SPEED_OF_LIGHT**4))
    * ((1.0e-10**4) / AMU_TO_KG)
)


# ---------
# Functions
# ---------


def adjust_gamma_phonons_for_spectrum_type(
    freqs, ints, lws, irrep_syms=None, spectrum_type="stokes"
):
    """Adjust a set of band frequencies, intensities, linewidths and,
    optionally, irreps for different types of Raman spectrum.

    Parameters
    ----------
    freqs : array_like
        Frequencies (shape: `(N,)`).
    ints : array_like
        Intensities for `N` bands (shape: `(N,)` or `N` modes and `M`
        calculations (shape: `(N, M)`).
    lws : array_like or None, optional
        Linewidths (shape: `(N,)`).
    irrep_syms : array_like or None, optional
        Irreducible representation (irrep) symbols (shape: `(N,)`).
    spectrum_type : {'stokes', 'anti-stokes', 'both'}
        Type of Raman spectrum (default "stokes").

    Returns
    -------
    phonons : tuple of (numpy.ndarray or None)
        `(freqs, ints, lws, irreps)` tuple with input data adjusted for
        `spectrum_type`.
    """

    freqs = np.asarray(freqs)

    if not np_check_shape(freqs, (None,)):
        raise ValueError("freqs must be an array_like with shape (N,).")

    ints, n_dim_add = np_expand_dims(np.asarray(ints), (len(freqs), None))

    if lws is not None:
        lws = np.asarray(lws)

        if not np_check_shape(lws, (len(freqs),)):
            raise ValueError("lws must be an array_like with shape (N,).")

    if irrep_syms is not None:
        irrep_syms = np.asarray(irrep_syms)

        if not np_check_shape(irrep_syms, (len(freqs),)):
            raise ValueError(
                "irrep_syms must be an array_like with shape (N,)."
            )

    spectrum_type = str(spectrum_type).lower()

    if spectrum_type == "stokes":
        # Nothing to do - return input data.
        return (
            freqs,
            ints if n_dim_add == 0 else ints.reshape((-1,)),
            lws,
            irrep_syms,
        )

    if spectrum_type == "anti-stokes" or spectrum_type == "both":
        # Reverse data for anti-stokes spectrum.

        freqs_as = (-1.0 * freqs)[::-1]
        ints_as = ints[::-1, :]
        lws_as = lws[::-1] if lws is not None else None
        irrep_syms_as = irrep_syms[::-1] if irrep_syms is not None else None

        if spectrum_type == "anti-stokes":
            return (
                freqs_as,
                ints_as if n_dim_add == 0 else ints_as.reshape((-1,)),
                lws_as,
                irrep_syms_as,
            )

        else:
            ints_both = np.concatenate((ints_as, ints), axis=0)

            return (
                np.concatenate((freqs_as, freqs), axis=0),
                ints_both if n_dim_add == 0 else ints_both.reshape((-1,)),
                np.concatenate((lws_as, lws), axis=0),
                np.concatenate((irrep_syms_as, irrep_syms), axis=0),
            )

    raise ValueError("Unsupported spectrum type: '{0}'.".format(spectrum_type))


def modulate_intensities(freqs, ints, w=None, t=None):
    """Modulate Raman band intensities for a measurement wavelength
    and/or temperature.

    Parameters
    ----------
    freqs : array_like
        Frequencies (shape: `(N,)`).
    ints : array_like
        Intensities for `N` bands (shape: `(N,)` or `N` modes and `M`
        calculations (shape: `(N, M)`).
    w, t : float, optional
        Measurement wavelength in nm and temperature in K.

    Returns
    -------
    mod_ints : numpy.ndarray
        Modulated intensities (same shape as `ints`).
    """

    freqs = np.asarray(freqs)

    if not np_check_shape(freqs, (None,)):
        raise ValueError("freqs must be an array_like with shape (N,).")

    ints, n_dim_add = np_expand_dims(np.asarray(ints), (len(freqs), None))

    if w is not None:
        if w <= 0.0:
            raise ValueError("If supplied, w must be non-zero and positive.")

        f_i = nm_to_thz(w)

        scale = _RAMAN_INTENSITY_PREFACTOR * (
            ((1.0e12 * (f_i - freqs)) ** 4) / (1.0e12 * f_i)
        )

        ints *= scale[:, np.newaxis]

    if t is not None:
        if t <= 0.0:
            raise ValueError("If supplied, t must be non-zero and positive.")

        occ_nums = phonon_occupation_number(np.abs(freqs), t)

        mask = freqs >= 0.0
        inv_mask = np.logical_not(mask)

        # The weighting factors are different for Stokes and
        # anti-Stokes branches.

        if mask.any():
            ints[mask, :] *= occ_nums[mask, np.newaxis] + 1.0

        if inv_mask.any():
            ints[inv_mask, :] *= occ_nums[inv_mask, np.newaxis]

    return ints if n_dim_add == 0 else ints.reshape((-1,))


# -------------------
# RamanSpectrum class
# -------------------


class RamanSpectrum(GammaPhononSpectrumBase):
    """Simulate 1D or 2D Raman spectra (e.g. for an angle rotation) from
    sets of frequencies, band intensities and linewidths."""

    def __init__(
        self,
        freqs,
        ints,
        lws,
        irreps=None,
        w=None,
        t=None,
        spectrum_type="stokes",
        x_range=None,
        x_res=None,
        x_units="thz",
    ):
        r"""Create a new instance of the `RamanSpectrum` class.

        Parameters
        ----------
        freqs : array_like
            List of frequencies in THz.
        ints : array_like
            List of `N` scalar Raman band intensities, or 2D array of
            `N` intensities for `M` calculations (shape: `(N`,)` or
            `(N, M)`). Units are Ang^4 / sqrt(amu).
        lws : array_like
            List of linewidths in THz.
        irreps : Irreps or None
            An Irreps object assigning bands to irrep groups.
        w : float or None
            Measurement wavelength in nm to calculate scattered photon
            energies for band intensity modulation envelope (default:
            None).
        t : float or None
            Temperature in K to calculate phonon occupation numbers for
            band intensity modulation envelope (default: None)
        x_range : tuple of float or None
            Range of spectrum as a `(min, max)` tuple or `None` to
            automatically determine a suitable range (default: `None`).
        x_res : float or None
            Resolution of spectrum (default: automatically determined).
        spectrum_type : {'stokes', 'anti-stokes', 'both'}
            Type of spectrum (default: 'stokes').
        x_units : str
            Frequency units of spectrum (default: 'thz').
        """

        freqs = np_asarray_copy(freqs, dtype=np.float64)

        if not np_check_shape(freqs, (None,)):
            raise ValueError("freqs must be an array_like with shape `(N,)`.")

        ints, n_dim_add = np_expand_dims(
            np_asarray_copy(ints, dtype=np.float64), (len(freqs), None)
        )

        # Keep track of dimensionality of ints for returning results.

        ints_2d = n_dim_add == 0

        lws = np_asarray_copy(lws, dtype=np.float64)

        if not np_check_shape(lws, (len(freqs),)):
            raise ValueError("lws must be an array_like with shape `(N,)`.")

        # If we have any imaginary modes with non-zero intensity, it
        # not cause issues in the calculations but it may give
        # unphysical results in some cases.

        mask = np.logical_and(freqs < 0.0, np.abs(freqs) > ZERO_TOLERANCE)

        if (np.abs(ints[mask, :]) > ZERO_TOLERANCE).any():
            warnings.warn(
                "Imaginary modes with non-zero intensity may lead to "
                "unphysical results, particularly if applying "
                "temperature modulation to the intensities and/or "
                "simulating anti-Stokes spectra.",
                UserWarning,
            )

        # If irreps are supplied, average frequencies/linewidths and sum
        # intensities.

        irrep_syms = None

        if irreps is not None:
            ir_band_inds = irreps.irrep_band_indices

            freqs = np.array(
                [np.mean(freqs[inds]) for inds in ir_band_inds],
                dtype=np.float64,
            )

            ints = np.array(
                [np.sum(ints[inds, :], axis=0) for inds in ir_band_inds],
                dtype=np.float64,
            )

            lws = np.array(
                [np.mean(lws[inds]) for inds in ir_band_inds], dtype=np.float64
            )

            irrep_syms = np_asarray_copy(irreps.irrep_symbols, dtype=object)

        # Adjust freqs, ints, lws and irrep_syms for spectrum type.

        freqs, ints, lws, irrep_syms = adjust_gamma_phonons_for_spectrum_type(
            freqs,
            ints,
            lws,
            irrep_syms=irrep_syms,
            spectrum_type=spectrum_type,
        )

        # Apply intensity modulation.

        ints = modulate_intensities(freqs, ints, w, t)

        # Call the SpectrumBase constructor to handle "x-axis"-related
        # intialisation. This also validates freqs, lws and irreps, and
        # the optional x_range, x_res and units parameters.

        super(RamanSpectrum, self).__init__(
            freqs,
            lws,
            irrep_syms=irrep_syms,
            x_range=x_range,
            x_res=x_res,
            x_units=x_units,
        )

        # Store remaining parameters.

        self._ints = ints
        self._ints_2d = ints_2d

        self._w = w
        self._t = t

        self._spectrum_type = spectrum_type

        # Set _y to default value.

        self._y = None

    def _lazy_init_y(self):
        """Simulate spectrum on first call to `y` or `spectrum()`."""

        if self._y is None:
            x = self.x

            _, num_sp = self._ints.shape

            y = np.zeros((len(x), num_sp), dtype=np.float64)

            for sp_idx in range(self._ints.shape[1]):
                for f, i, lw in zip(
                    self._freqs, self._ints[:, sp_idx], self._lws
                ):
                    y[:, sp_idx] += lorentzian(x, i, f, lw)

            self._y = y

    @property
    def frequencies(self):
        """numpy.ndarray : Phonon frequencies."""
        # If the spectrum includes anti-Stokes branches, the frequencies
        # are actually Raman shifts and may include negative values. To
        # avoid confusion, we hide the base class property and intercept
        # it with a warning.

        if (
            self._spectrum_type == "anti-stokes"
            or self._spectrum_type == "both"
        ):
            warnings.warn(
                "The values returned by frequencies are Raman shifts "
                "and not phonon frequencies (to avoid this warning, "
                "use the raman_shifts property alias instead).",
                RuntimeWarning,
            )

        return super(RamanSpectrum, self).frequencies

    @property
    def raman_shifts(self):
        """numpy.ndarray : Raman shifts."""
        return super(RamanSpectrum, self).frequencies

    @property
    def intensities(self):
        """numpy.ndarray : Band intensities."""
        return np_readonly_view(
            self._ints if self._ints_2d else self._ints.reshape((-1,))
        )

    @property
    def spectrum_type(self):
        """{'stokes', 'anti-stokes', 'both'} : Type of spectrum."""
        return self._spectrum_type

    @property
    def measurement_wavelength(self):
        """float or None : Measurement wavelength."""
        return self._w

    @property
    def measurement_temperature(self):
        """float or None : Measurement temperature."""
        return self._t

    @property
    def is_2d_spectrum(self):
        """bool : `True` if the spectrum is 2D, otherwise `False`."""
        return self._ints_2d

    @property
    def y(self):
        """array_like : y values for simulated spectrum."""

        self._lazy_init_y()

        return np_readonly_view(
            self._y if self._ints_2d else self._y.reshape((-1,))
        )

    @property
    def y_unit_text_label(self):
        """str : y-axis label suitable for plain-text output."""

        # If a measurement wavelength was set, the intensities are
        # converted to differential cross sections. If not, the
        # intensities are "raw" values.

        return (
            "d(sigma)/d(omega) / (Ang^2 sterad^-1)"
            if self._w is not None
            else "I^Raman / (Ang^4 amu^-1)"
        )

    @property
    def y_unit_plot_label(self):
        """str: y-axis label suitable for plotting (contains TeX
        strings.)
        """

        # (See comment on y_unit_text_label.)

        return (
            r"$d \sigma / d \Omega$ / ($\mathrm{\AA}^2$ sterad$^{-1}$)"
            if self._w is not None
            else r"$I^\mathrm{Raman}$ / ($\mathrm{AA}^4$ amu$^{-1}$)"
        )

    def _get_int_col_hdrs(self, int_col_hdrs=None):
        """Get column headers for the intensity columns in peak tables
        and simulated spectra in Pandas `DataFrame` objects.

        Parameters
        ----------
        int_col_hdrs : array_like or None, optional
            Column header(s).

        Returns
        -------
        int_col_hdrs : array_like
            Column header(s) if `int_col_hdrs` is specified, otherwise
            a default set of headers depending on the type of
            intensities calculated.
        """

        num_hdrs = self._ints.shape[1]

        if int_col_hdrs is not None:
            int_col_hdrs = np.array(
                [str(h) for h in int_col_hdrs], dtype=object
            )

            if not np_check_shape(int_col_hdrs, (num_hdrs,)):
                raise ValueError(
                    "int_col_hdrs must specify a header for each set of "
                    "intensities in the simulated spectrum."
                )
        else:
            label = "cross_sect" if self._w is not None else "int"

            if num_hdrs == 1:
                int_col_hdrs = [label]
            else:
                int_col_hdrs = [
                    "{0}_{1}".format(label, i + 1) for i in range(num_hdrs)
                ]

        return np.asarray(int_col_hdrs, dtype=object)

    def peak_table(self, int_col_hdrs=None):
        """Return the peak table as a Pandas `DataFrame`.

        Parameters
        ----------
        int_col_hdrs : array_like or None, optional
            Column header(s) for the intensity column(s) in the
            `DataFrame` (default: automatically determined).

        Returns
        -------
        df : pandas.DataFrame
            `DataFrame` containing the peak table.
        """

        d = {"freq_energy": self._freqs, "linewidth": self._lws}

        if self._irrep_syms is not None:
            d["irrep"] = self._irrep_syms
        else:
            d["irrep"] = ["None"] * len(self._freqs)

        for i, h in enumerate(
            self._get_int_col_hdrs(int_col_hdrs=int_col_hdrs)
        ):
            d[h] = self._ints[:, i]

        return pd.DataFrame(d)

    def spectrum(self, int_col_hdrs=None):
        """Return the spectrum as a Pandas `DataFrame`.

        Parameters
        ----------
        int_col_hdrs : array_like or None, optional
            Column header(s) for the intensity column(s) in the
            `DataFrame` (default: automatically determined).

        Returns
        -------
        df : pandas.DataFrame
            `DataFrame` containing the spectrum.
        """

        self._lazy_init_y()

        d = {"freq_energy": self.x}

        for i, h in enumerate(
            self._get_int_col_hdrs(int_col_hdrs=int_col_hdrs)
        ):
            d[h] = self._y[:, i]

        return pd.DataFrame(d)
