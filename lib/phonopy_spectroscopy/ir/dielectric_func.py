# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Routines implementing single-crystal and powder infrared dielectric
function simulations."""


# -------
# Imports
# -------


import numpy as np
import pandas as pd

from ..constants import (
    INFRARED_DIELECTIC_TO_RELATIVE_PERMITTIVITY,
    SPEED_OF_LIGHT,
)

from ..distributions import dielectric_function

from ..spectrum_base import GammaPhononSpectrumBase

from ..units import convert_frequency_units

from ..utility.numpy_helper import (
    np_readonly_view,
    np_asarray_copy,
    np_check_shape,
)


# ------------------------------------
# InfraredDielectricFunctionBase class
# ------------------------------------


class InfraredDielectricFunctionBase(GammaPhononSpectrumBase):
    """Base class for generating simulated infrared dielectric functions
    from sets of frequencies, mode oscillator strengths and linewidths.
    """

    def __init__(
        self,
        freqs,
        osc_strs,
        lws,
        cell_volume,
        irreps=None,
        eps_inf=None,
        **kwargs
    ):
        r"""Create a new instance of the
        `InfraredDielectricFunctionBase` class.

        Parameters
        ----------
        freqs : array_like
            Frequencies in THz (shape: `(N,)`).
        osc_strs : array_like
            Scalar or tensor oscillator strengths in e^2 / amu (must be
            at least 1D).
        lws : array_like
            Linewidths in THz (shape: `(N,)`).
        cell_volume : float
            Unit-cell volume in Ang^3.
        irreps : Irreps or None, optional
            `Irreps` object assigning bands to irrep groups.
        eps_inf : float, array_like or None, optional
            High-frequency dielectric constant in units of relative
            permittivity to add to the calculated dielectric function
            (same shape as elements in `osc_strs`).
        **kwargs
            Keyword arguments to the `GammaPhononSpectrumBase`
            constructor.
        """

        osc_strs = np_asarray_copy(osc_strs, dtype=np.float64)

        if not (
            np_check_shape(osc_strs, (None,))
            or np_check_shape(osc_strs, (None, 3, 3))
        ):
            raise ValueError(
                "osc_strs must be be an array_like with shape (N,) or "
                "(N, 3, 3)."
            )

        # If irreps are supplied, average frequencies/linewidths and sum
        # mode oscillator strengths.

        irrep_syms = None

        if irreps is not None:
            ir_band_inds = irreps.irrep_band_indices

            freqs = np.array(
                [np.mean(freqs[inds]) for inds in ir_band_inds],
                dtype=np.float64,
            )

            osc_strs = np.array(
                [np.sum(osc_strs[inds], axis=0) for inds in ir_band_inds],
                dtype=np.float64,
            )

            lws = np.array(
                [np.mean(lws[inds]) for inds in ir_band_inds], dtype=np.float64
            )

            irrep_syms = np_asarray_copy(irreps.irrep_symbols, dtype=object)

        if cell_volume <= 0.0:
            raise ValueError("cell_volume must be positive and non-zero.")

        if eps_inf is not None:
            eps_inf = np.asarray(eps_inf)

            if not (
                (osc_strs.ndim == 1 and eps_inf.ndim > 0)
                or np_check_shape(eps_inf, osc_strs.shape[1:])
            ):
                raise ValueError(
                    "If supplied, eps_inf must have the same "
                    "dimensions as the elements in osc_strs."
                )

            # The high-frequency dielectric constant should be real.

            if np.iscomplex(eps_inf).any():
                raise ValueError("If supplied, eps_inf must be real.")

            if eps_inf.ndim == 0:
                # Scalar -> store as a "regular" Python float.
                eps_inf = float(eps_inf)
            else:
                # Tensor -> store as a NumPy array.
                eps_inf = np_asarray_copy(eps_inf, dtype=np.float64)

        # Call the GammaPhononSpectrumBase constructor to handle
        # "x-axis"-related intialisation.

        super(InfraredDielectricFunctionBase, self).__init__(
            freqs, lws, irrep_syms=irrep_syms, **kwargs
        )

        # Store parameters.

        self._osc_strs = osc_strs
        self._cell_volume = cell_volume

        self._eps_inf = eps_inf

        self._dielectric_func = None

    def _lazy_init_dielectric_func(self):
        """Calculate the dielectric function when required."""

        if self._dielectric_func is None:
            # The conversion factor to relative permittivity assumes
            # oscillator strengths in e^2 / amu, volumes in Ang^2 and
            # frequencies in THz.

            x, freqs, lws = self.x, self.frequencies, self.linewidths

            if self._x_units != "thz":
                x = convert_frequency_units(x, self._x_units, "thz")
                freqs = convert_frequency_units(freqs, self._x_units, "thz")
                lws = convert_frequency_units(lws, self._x_units, "thz")

            dielectric_func = dielectric_function(
                x, self._osc_strs[0], freqs[0], lws[0]
            )

            for osc_str, freq, lw in zip(
                self._osc_strs[1:], freqs[1:], lws[1:]
            ):
                dielectric_func += dielectric_function(x, osc_str, freq, lw)

            # Convert to relative permittivity.

            dielectric_func *= (
                INFRARED_DIELECTIC_TO_RELATIVE_PERMITTIVITY / self._cell_volume
            )

            # If a high-frequency dielectric constant was supplied
            # during initialisation, add it to the dielectric function.

            if self._eps_inf is not None:
                # Note that _eps_inf may be a ("regular") float.
                dielectric_func += np.asarray(self._eps_inf)[np.newaxis]

            self._dielectric_func = dielectric_func

    @property
    def mode_oscillator_strength_unit_text_label(self):
        """str : Mode oscillator strength unit label suitable for
        plain-text output."""

        return "e^2 / amu"


# --------------------------------------
# TensorInfraredDielectricFunction class
# --------------------------------------


class TensorInfraredDielectricFunction(InfraredDielectricFunctionBase):
    """Class for generating simulated tensor infrared dielectric
    functions from sets of frequencies, mode oscillator strengths and
    linewidths."""

    def __init__(
        self,
        freqs,
        osc_strs,
        lws,
        cell_volume,
        irreps=None,
        eps_inf=None,
        x_range=None,
        x_res=None,
        x_units="thz",
        **kwargs
    ):
        r"""Create a new instance of the
        `TensorInfraredDielectricFunctionBase` class.

        Parameters
        ----------
        struct : Structure
            Crystal structure.
        freqs : array_like
            Frequencies in THz (shape: `(N,)`).
        osc_strs : array_like
            Tensor oscillator strengths in e^2 / amu (shape:
            `(N, 3, 3)`) .
        lws : array_like
            Linewidths in THz (shape: `(N,)`).
        cell_volume : float
            Unit-cell volume in Ang^3.
        irreps : Irreps or None, optional
            `Irreps` object assigning bands to irrep groups.
        eps_inf : float or None, optional
            High-frequency dielectric constant in units of relative
            permittivity to add to the calculated dielectric function.
        x_range : tuple of float or None, optional
            `(min, max)` range of x-axis of simulated spectrum in
            `x_units` (default: automatically determined).
        x_res : float or None, optional
            Resolution of x-axis in `x_units` (default: automatically
            determined).
        x_units : str or None, optional
            x-axis units of simulated spectrum (default: `"thz"`).
        **kwargs : any
            Keyword arguments to parent class constructor(s).
        """

        osc_strs = np.asarray(osc_strs, dtype=np.float64)

        if not np_check_shape(osc_strs, (None, 3, 3)):
            raise ValueError(
                "osc_strs must be an array_like with shape (N, 3, 3)."
            )

        # Call the TensorInfraredDielectricFunctionBase constrictor to
        # perform additional initialisation.

        super(TensorInfraredDielectricFunction, self).__init__(
            freqs,
            osc_strs,
            lws,
            cell_volume,
            irreps=irreps,
            eps_inf=eps_inf,
            x_range=x_range,
            x_res=x_res,
            x_units=x_units,
            **kwargs
        )

    @property
    def oscillator_strengths(self):
        """numpy.ndarray : Mode oscillator strengths (shape: '(N, 3, 3)')."""
        return np_readonly_view(self._osc_strs)

    @property
    def epsilon_inf(self):
        """numpy.ndarray or None : High-frequency dielectric constant
        (shape: `(3, 3)`)."""

        if self._eps_inf is not None:
            return np_readonly_view(self._eps_inf)

        return None

    @property
    def epsilon(self):
        """numpy.ndarray : Complex dielectric function (shape:
        `(O, 3, 3)`)."""

        self._lazy_init_dielectric_func()
        return np_readonly_view(self._dielectric_func)

    @property
    def epsilon_unit_text_label(self):
        """str : Dielectric function unit label suitable for plain-text
        output."""

        return r"\eps / \eps_0"

    @property
    def epsilon_unit_plot_label(self):
        """str : Dielectric function unit label suitable for plotting
        (contains TeX strings)."""

        return r"$\epsilon$ / $\epsilon_0$"

    def peak_table(self):
        """Return the peak table as a Pandas `DataFrame`.

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

        for i, j, label in TensorInfraredDielectricFunction._DF_COL_INDS_HDRS:
            d["osc_str_{0}".format(label)] = self._osc_strs[:, i, j]

        return pd.DataFrame(d)

    def spectrum(self):
        """Return the simulated dielectric function as a Pandas
        `DataFrame`.

        Returns
        -------
        df : pandas.DataFrame
            `DataFrame` containing the dielectric function.
        """

        self._lazy_init_dielectric_func()

        d = {"freq_energy": self.x}

        for i, j, label in TensorInfraredDielectricFunction._DF_COL_INDS_HDRS:
            d["eps_re_{0}".format(label)] = self._dielectric_func.real[:, i, j]

        for i, j, label in TensorInfraredDielectricFunction._DF_COL_INDS_HDRS:
            d["eps_im_{0}".format(label)] = self._dielectric_func.imag[:, i, j]

        return pd.DataFrame(d)

    _DF_COL_INDS_HDRS = [
        (0, 0, "xx"),
        (1, 1, "yy"),
        (2, 2, "zz"),
        (0, 1, "xy"),
        (0, 2, "xz"),
        (1, 2, "yz"),
    ]

    """list of tuples of (int, int, str) : Indices and column labels
    for constructing Pandas `DataFrame` objects from the unique
    components of the mode oscillator strengths/tensor dielectric
    function."""


# --------------------------------------
# ScalarInfraredDielectricFunction class
# --------------------------------------


class ScalarInfraredDielectricFunction(InfraredDielectricFunctionBase):
    """Class for generating simulated scalar infrared dielectric
    functions from sets of frequencies, mode oscillator strengths and
    linewidths."""

    def __init__(
        self,
        freqs,
        osc_strs,
        lws,
        cell_volume,
        irreps=None,
        eps_inf=None,
        x_range=None,
        x_res=None,
        x_units="thz",
        **kwargs
    ):
        r"""Create a new instance of the
        `ScalarInfraredDielectricFunctionBase` class.

        Parameters
        ----------
        struct : Structure
            Crystal structure.
        freqs : array_like
            Frequencies in THz (shape: `(N,)`).
        osc_strs : array_like
            Scalar oscillator strengths in e^2 / amu (shape: `(N,)`) .
        lws : array_like
            Linewidths in THz (shape: `(N,)`).
        cell_volume : float
            Unit-cell volume in Ang^3.
        irreps : Irreps or None, optional
            `Irreps` object assigning bands to irrep groups.
        eps_inf: array_like or None, optional
            High-frequency dielectric constant in units of relative
            permittivity to add to the calculated dielectric function
            (shape: `(3, 3)`).
        x_range : tuple of float or None, optional
            `(min, max)` range of x-axis of simulated spectrum in
            `x_units` (default: automatically determined).
        x_res : float or None, optional
            Resolution of x-axis in `x_units` (default: automatically
            determined).
        x_units : str or None, optional
            x-axis units of simulated spectrum (default: `"thz"`).
        **kwargs : any
            Keyword arguments to parent class constructor(s).
        """

        osc_strs = np.asarray(osc_strs, dtype=np.float64)

        if not np_check_shape(osc_strs, (None,)):
            raise ValueError("osc_strs must be an array_like with shape (N,).")

        super(ScalarInfraredDielectricFunction, self).__init__(
            freqs,
            osc_strs,
            lws,
            cell_volume,
            irreps=irreps,
            eps_inf=eps_inf,
            x_range=x_range,
            x_res=x_res,
            x_units=x_units,
            **kwargs
        )

        self._dielectric_func = None

        self._n_tilde = None

        self._abs_coeff = None
        self._ref_coeff = None
        self._loss_func = None

    def _lazy_init_derived(self):
        """Calculate derived quantities on first call to
        `absorption_coefficient`, `reflection_coefficient`,
        `loss_function` or `spectrum`.
        """

        self._lazy_init_dielectric_func()

        if (
            self._abs_coeff is None
            or self._ref_coeff is None
            or self._loss_func is None
        ):
            # Complex refractive index:
            #     \tilde{n} = sqrt(\epsilon) = n + ik

            n_tilde = np.sqrt(self._dielectric_func)

            self._n_tilde = n_tilde

            # Absorption coefficient: \alpha = (2 \omega / c) * k

            # Using frequencies in THz (converted to Hz) and c in m/s
            # (converted to cm/s) should give \alpha in cm^-1.

            v = convert_frequency_units(self.x, self._x_units, "thz")

            self._abs_coeff = (2.0 * 1.0e12 * v * n_tilde.imag) / (
                1.0e2 * SPEED_OF_LIGHT
            )

            # Reflection coefficient (normal incidence):
            # | (\tilde{n} - 1) / (\tilde{n} + 1) |^2

            self._ref_coeff = np.abs((n_tilde - 1) / (n_tilde + 1)) ** 2

            # Loss function: Im[1 / \epsilon]

            self._loss_func = (-1.0 / self._dielectric_func).imag

    @property
    def oscillator_strengths(self):
        """numpy.ndarray : Mode oscillator strengths (shape: '(N,)')."""
        return np_readonly_view(self._osc_strs)

    @property
    def epsilon_inf(self):
        """float or None : High-frequency dielectric constant."""

        if self._eps_inf is not None:
            return self._eps_inf

        return None

    @property
    def epsilon(self):
        """numpy.ndarray : Complex dielectric function (shape: `(O,)`)."""

        self._lazy_init_dielectric_func()
        return np_readonly_view(self._dielectric_func)

    @property
    def complex_refractive_index(self):
        """numpy.ndarray : Complex refractive index (shape: `(O,)`)."""

        self._lazy_init_derived()
        return np_readonly_view(self._n_tilde)

    @property
    def refractive_index(self):
        """numpy.ndarray : Refractive index (shape: `(O,)`)."""

        self._lazy_init_derived()
        return np_readonly_view(self._n_tilde.real)

    @property
    def extinction_coefficient(self):
        """numpy.ndarray : Extinction coefficient (shape: `(O,)`)."""

        self._lazy_init_derived()
        return np_readonly_view(self._n_tilde.imag)

    @property
    def absorption_coefficient(self):
        """numpy.ndarray : Absorption coefficient (shape: `(O,)`)."""

        self._lazy_init_derived()
        return np_readonly_view(self._abs_coeff)

    @property
    def reflectance_coefficient(self):
        """numpy.ndarray : Reflection coefficient at normal incidence
        (shape: `(O,)`)."""

        self._lazy_init_derived()
        return np_readonly_view(self._ref_coeff)

    @property
    def loss_function(self):
        """numpy.ndarray : Loss function (shape: `(O,)`)."""

        self._lazy_init_derived()
        return np_readonly_view(self._loss_func)

    @property
    def epsilon_unit_text_label(self):
        """str : Dielectric function unit label suitable for plain-text
        output."""

        return r"\eps / \eps_0"

    @property
    def epsilon_unit_plot_label(self):
        """str : Dielectric function unit label suitable for plotting
        (contains TeX strings)."""

        return r"$\epsilon$ / $\epsilon_0$"

    @property
    def absorption_coefficient_unit_text_label(self):
        """str : Absorption coefficient unit label suitable for
        plain-text output."""

        return r"\alpha / cm^-1"

    @property
    def absorption_coefficient_unit_plot_label(self):
        """str : Absorption coefficient unit label suitable for plotting
        (contains TeX strings)."""

        return r"$\alpha$ / cm$^{-1}$"

    def peak_table(self):
        """Return the peak table as a Pandas `DataFrame`.

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

        d["osc_str"] = self._osc_strs

        return pd.DataFrame(d)

    def spectrum(self, sp_type="epsilon"):
        """Return the simulated dielectric function and derived
        quantities as a Pandas `DataFrame`.

        Returns
        -------
        df : pandas.DataFrame
            `DataFrame` containing the dielectric function and derived
            quantities.
        """

        self._lazy_init_derived()

        d = {
            "freq_energy": self.x,
            "eps_re": self._dielectric_func.real,
            "eps_im": self._dielectric_func.imag,
            "abs_coeff": self._abs_coeff,
            "ref_coeff": self._ref_coeff,
            "loss_func": self._loss_func,
        }

        return pd.DataFrame(d)
