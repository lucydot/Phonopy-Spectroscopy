# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Routines implementing single-crystal and powder Raman simulations."""


# -------
# Imports
# -------


import warnings

import numpy as np

from ..constants import ZERO_TOLERANCE

from ..distributions import march_dollase

from ..utility.geometry import direction_cosine

from ..utility.numpy_helper import np_expand_dims

from ..intensity_base import (
    calculate_single_crystal_intensities,
    calculate_powder_intensities,
)


# ---------
# Constants
# ---------

_EIGHT_PI_SQUARED = 8.0 * np.pi**2

"""Value of 8 * pi^2. """


# ----------------------
# Intensity calculations
# ----------------------


def _raman_intensity(v_i, t, v_s):
    """Calculate the scalar Raman intensity for a Raman tensor and
    incident/scattered polarisation.

    Parameters
    ----------
    t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered light polarisation vectors.

    Returns
    -------
    int : float
        Scalar Raman intensity.
    """

    return np.abs(np.dot(v_i, np.dot(t, v_s))).real ** 2


def _powder_nquad_int(phi, theta, psi, v_i, t, v_s):
    """Integrand function for calculating powder-averaged Raman
    intensities using the SciPy `nquad()` routine.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles.
    t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered light polarisation vectors.

    Returns
    -------
    int : float
        Scalar Raman intensity.

    See Also
    --------
    intensity_base.calculate_powder_intensities
    intensity_base.calculate_powder_intensities_nquad
        "Base" functions using this integrand.
    """

    r = direction_cosine(phi, theta, psi)

    return (
        np.abs(np.dot(v_i, np.dot(r, np.dot(t, np.dot(r.T, v_s))))).real ** 2
    ) * (np.sin(theta) / _EIGHT_PI_SQUARED)


def _powder_nquad_odf_int(
    phi, theta, psi, v_i, t, v_s, po_r, po_norm, po_axis
):
    """Integrand function for calculating powder-averaged Raman
    intensities with the March-Dollase orientation distribution function
    using the SciPy `nquad()` routine.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles.
    t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered light polarisation vectors.
    po_r : float
        r parameter in the March-Dollase distribution.
    po_norm, po_axis : array_like
        Surface normal and reference axis for preferred orientation
        (shape: `(3,)`).

    Returns
    -------
    int : float
        Scalar Raman intensity.

    See Also
    --------
    intensity_base.calculate_powder_intensities
    intensity_base.calculate_powder_intensities_odf_nquad
        "Base" functions using this integrand.
    """

    r = direction_cosine(phi, theta, psi)
    po_alpha = np.arccos(np.dot(np.dot(r, po_norm), po_axis))

    return (
        march_dollase(po_alpha, po_r)
        * np.abs(np.dot(v_i, np.dot(r, np.dot(t, np.dot(r.T, v_s))))).real ** 2
        * (np.sin(theta) / _EIGHT_PI_SQUARED)
    )


def _powder_lebedev_int(phi, theta, psi, v_i, t, v_s):
    """Integrand function for calculating powder-averaged Raman
    intensities using Lebedev + circle quadrature.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles.
    t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered light polarisation vectors.

    Returns
    -------
    int : float
        Scalar Raman intensity.

    See Also
    --------
    intensity_base.calculate_powder_intensities
    intensity_base.calculate_powder_intensities_lebedev_circle
        "Base" functions using this integrand.
    """

    r = direction_cosine(phi, theta, psi)

    return (
        np.abs(np.dot(v_i, np.dot(r, np.dot(t, np.dot(r.T, v_s))))).real ** 2
    )


def _powder_lebedev_odf_int(
    phi, theta, psi, v_i, t, v_s, po_r, po_norm, po_axis
):
    """Integrand function for calculating powder-averaged Raman
    intensities with the March-Dollase orientation distribution function
    using Lebedev + circle quadrature.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles.
    t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered light polarisation vectors.
    po_r : float
        r parameter in the March-Dollase distribution.
    po_norm, po_axis : array_like
        Surface normal and reference axis for preferred orientation
        (shape: `(3,)`).

    Returns
    -------
    int : float
        Scalar Raman intensity.

    See Also
    --------
    intensity_base.calculate_powder_intensities
    intensity_base.calculate_powder_intensities_odf_lebedev_circle
        "Base" functions using this integrand.
    """

    r = direction_cosine(phi, theta, psi)
    po_alpha = np.arccos(np.dot(np.dot(r, po_norm), po_axis))

    return (
        march_dollase(po_alpha, po_r)
        * np.abs(np.dot(v_i, np.dot(r, np.dot(t, np.dot(r.T, v_s))))).real ** 2
    )


# --------------------
# Single-crystal Raman
# --------------------


def calculate_single_crystal_raman_intensities(
    r_t, geom, i_pol, s_pol, rot=None
):
    """Calculate the scalar Raman intensities for a polarised Raman
    measurement on a single crystal.

    Parameters
    ----------
    r_t : array_like
        Raman tensors (shape: `(3, 3)` or `(N, 3, 3)`).
    geom : Geometry
        Measurement geometry.
    i_pol, s_pol : Polarisation
        Polarisations of incident and scattered light.
    rot : array_like
        Rotation matrix to apply to Raman tensors.

    Returns
    -------
    ints : numpy.ndarray
        Calcuated intensity (scalar) or set of intensities
        (shape: `(N,)`).

    See Also
    --------
    intensity_base.single_crystal_intensities : Low-level
        implementation called by this function.
    """

    return calculate_single_crystal_intensities(
        _raman_intensity, r_t, geom, i_pol, s_pol, rot=rot
    )


# ------------
# Powder Raman
# ------------


def calculate_powder_raman_intensities_analytical(r_t, geom, i_pol, s_pol):
    """Calculate the scalar Raman intensities for a polarised Raman
    measurement with powder averaging, using the analytical formula.

    Parameters
    ----------
    r_t : array_like
        Raman tensors (shape: `(3, 3)` or `(N, 3, 3)`).
    geom : Geometry
        Measurement geometry.
    i_pol, s_pol : Polarisation
        Polarisations of incident and scattered light.


    Returns
    -------
    ints : float or numpy.ndarray
        Calcuated intensity or array of intensities (shape: `(N,)`).

    Notes
    -----
    The formula calculates the intensity from the angle between the
    incident and scattered polarisation, and is only valid if:

    * The Raman tensors are real; and
    * The laser polarisation is perpendicular to the collection axis.
    """

    if not (
        geom.check_incident_polarisations(i_pol)
        and geom.check_scattered_polarisations(s_pol)
    ):
        raise ValueError(
            "The supplied incident and scattered polarisation are "
            "not possible with the supplied geometry."
        )

    r_t, n_dim_add = np_expand_dims(np.asarray(r_t), (None, 3, 3))

    # The analytical formula is only valid if the incident polarisation
    # is perpendicular to the collection axis.

    if not i_pol.check_perpendicular(geom.collection_direction):
        raise RuntimeError(
            "The analytical formula can only be used when the "
            "incident polarisation is perpendicular to the collection "
            "axis - use the numerical routines instead."
        )

    # Raman tensors calculated without using the far from resonance
    # (FFR) approximation can in general be complex. Since the
    # analytical formula was derived under the FFR, we only use it
    # for real Raman tensors.

    if np.iscomplex(r_t).any():
        raise RuntimeError(
            "The analytical formula can only be used for real Raman "
            "tensors - use the numerical routines instead."
        )

    ints = np.zeros((r_t.shape[0],), dtype=np.float64)

    for i, t in enumerate(r_t.real):
        # \alpha^\prime in Porezag and Pederson's notation.

        a_p = np.abs(np.trace(t) / 3.0)

        # (\beta^\prime)^2.

        b_p_2 = (
            (t[0, 0] - t[1, 1]) ** 2
            + (t[0, 0] - t[2, 2]) ** 2
            + (t[1, 1] - t[2, 2]) ** 2
            + 6.0 * (t[0, 1] ** 2 + t[0, 2] ** 2 + t[1, 2] ** 2)
        ) / 2.0

        i_par = a_p**2 + (4.0 / 45.0) * b_p_2
        i_per = (3.0 / 45.0) * b_p_2

        for v_i, v_s, w in i_pol.combine_with_iter(s_pol):
            cos_chi = np.dot(v_i, v_s) / (
                np.linalg.norm(v_i) * np.linalg.norm(v_s)
            )

            ints[i] += w * (i_per + (i_par - i_per) * cos_chi**2)

    return ints if n_dim_add == 0 else ints[0]


def calculate_powder_raman_intensities(
    r_t,
    geom,
    i_pol,
    s_pol,
    po_eta=0.0,
    po_surf_norm=None,
    method="best",
    lc_prec=5,
):
    """Calculate the scalar Raman intensities for a polarised Raman
    measurement on a powder.

    Parameters
    ----------
    r_t : array_like
        Raman tensors (shape: `(3, 3)` or `(N, 3, 3)`).
    geom : Geometry
        Measurement geometry.
    i_pol, s_pol : Polarisation
        Polarisation of incident and scattered light.
    po_eta : float, optional
        Fractional excess of crystalites in preferred orientation
        (default: 0.0).
    po_surf_norm : array_like or str, optional
        Surface normal for preferred orientation (default: `None`,
        required if `po_eta` > 0).
    method : {"nquad", "lebedev+circle", "best"}
        Method for calculating intensnties.
    lc_prec : int
        Specifies the precision of the Lebedev/circle quadrature scheme
        for `method="lebedev+circle"`.

    Returns
    -------
    ints : float or numpy.ndarray
        Calcuated intensity (scalar) or set of intensities (shape:
        `(N,)`).

    See Also
    --------
    intensity_base.calculate_powder_intensities : Low-level
        implementation of numerical powder averaging called by this
        function.
    """

    r_t, _ = np_expand_dims(np.asarray(r_t), (None, 3, 3))

    # Determine whether an energy-dependent Raman calculation and/or
    # a calculation with a preferred orientation are required.

    complex_rt = np.iscomplex(r_t).any()
    pref_orient = po_eta > ZERO_TOLERANCE

    # If the Raman tensors are real, there is no preferred orientation,
    # and the incident polarisation is perpendicular to the collection
    # direction, we can use the analytical formula.

    if (
        not complex_rt
        and not pref_orient
        and method == "best"
        and i_pol.check_perpendicular(geom.collection_direction)
    ):
        return calculate_powder_raman_intensities_analytical(
            r_t, geom, i_pol, s_pol
        )

    # If a preferred orientation is specified, or if the Lebedev +
    # circle precision is set to the minimum value, the SciPy nquad()
    # routine is the "safe" option.

    if method == "best":
        if not pref_orient and lc_prec >= 5:
            method = "lebedev+circle"
        else:
            method = "nquad"

    if pref_orient and method == "lebedev+circle":
        warnings.warn(
            "Powder Raman simulations with preferred orientation using "
            "the Lebedev + circle quadrature scheme require careful "
            "testing of the precision and should ideally be verified "
            'against the results with method="nquad".',
            UserWarning,
        )

    return calculate_powder_intensities(
        _raman_intensity,
        r_t,
        geom,
        i_pol,
        s_pol,
        po_eta=po_eta,
        po_norm=po_surf_norm,
        po_axis=-1.0 * geom.incident_direction,
        method=method,
        lc_prec=lc_prec,
        nquad_int_f=_powder_nquad_int,
        nquad_odf_int_f=_powder_nquad_odf_int,
        # lc_int_f=_powder_lebedev_int,
        # lc_odf_int_f=_powder_lebedev_odf_int,
    )
