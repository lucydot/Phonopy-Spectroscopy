# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Routines implementing single-crystal and powder Raman simulations.
"""


# -------
# Imports
# -------


import warnings

import numpy as np

from itertools import chain, product

from scipy.integrate import nquad

from ..distributions import march_dollase, march_dollase_eta_to_r

from ..utility.numpy_helper import np_expand_dims

from ..utility.geometry import (
    parse_direction,
    rotate_tensors,
    direction_cosine,
)

from ..utility.quadrature import lebedev_circle_quad

from ..constants import ZERO_TOLERANCE


# --------------------
# Single-crystal Raman
# --------------------


def calculate_single_crystal_raman_intensities(
    r_t, geom, i_pol, s_pol, rot=None
):
    """Calculate the scalar Raman intensities for a polarised Raman
    measurement.

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
    """

    if not (
        geom.check_incident_polarisations(i_pol)
        and geom.check_scattered_polarisations(s_pol)
    ):
        raise ValueError(
            "The supplied incident and scattered polarisation(s) are "
            "not possible with the supplied geometry."
        )

    r_t, n_dim_add = np_expand_dims(np.asarray(r_t), (None, 3, 3))

    # Rotate tensors if required.

    if rot is not None:
        r_t = rotate_tensors(r_t, rot)

    ints = np.zeros((r_t.shape[0],), dtype=np.float64)

    for v_i, v_s, w in i_pol.combine_with_iter(s_pol):
        for i, t in enumerate(r_t):
            ints[i] += w * np.abs(np.dot(v_i, np.dot(t, v_s))).real ** 2

    return ints if n_dim_add == 0 else ints[0]


# ------------------------------------------------------
# Powder raman: numerical integration with SciPy nquad()
# ------------------------------------------------------


def _powder_raman_integrand_nquad(phi, theta, psi, r_t, v_i, v_s):
    """Integrand for calculating Raman intensities of ideal powders
    using general numerical quadrature.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles in Radians.
    r_t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered polarisation vectors (shape: `(3, 3)`).

    Notes
    -----
    This routine is part of the implementation of numerical powder Raman
    simulations and should not be used directly. In particular, the
    input parameters are not checked.
    """

    r = direction_cosine(phi, theta, psi)

    return (
        (1.0 / (8.0 * np.pi**2))
        * (
            np.abs(np.dot(v_s, np.dot(r.T, np.dot(r_t, np.dot(r, v_i))))).real
            ** 2
        )
        * np.sin(theta)
    )


def _powder_raman_intensities_nquad(r_t, i_pol, s_pol):
    """Calculate powder Raman intensities with numerical integration
    using the general `scipy.integrate.nquad()` routine.

    Parameters
    ----------
    r_t : array_like
        Raman tensor (shape: `(3, 3)`).
    i_pol, s_pol : Polarisation
        Polarisations of incident and scattered light.

    Notes
    -----
    This routine is part of the implementation of numerical powder Raman
    simulations and should not be used directly. In particular, the
    input parameters arenot checked.
    """

    int_sum = 0.0

    for v_i, v_s, w in i_pol.combine_with_iter(s_pol):
        i, _ = nquad(
            _powder_raman_integrand_nquad,
            [(0.0, 2.0 * np.pi), (0.0, np.pi), (0.0, 2.0 * np.pi)],
            [r_t, v_i, v_s],
        )

        int_sum += w * i

    return int_sum


def _powder_raman_integrand_with_odf_nquad(
    phi, theta, psi, r_t, v_i, v_s, po_r, po_surf_norm, po_ref_axis
):
    """Integrand for calculating Raman intensities of powders with
    preferred orientation using general numerical quadrature.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles in Radians.
    r_t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered polarisation vectors (shape: `(3, 3)`).
    po_r : float
        March parameter for the March-Dollase distribution function.
    po_surf_norm, po_ref_axis : array_like
        Surface normal and reference axis used to determine the angle
        `alpha` in the March-Dollase function.

    Notes
    -----
    This routine is part of the implementation of numerical powder Raman
    simulations and should not be used directly. In particular, the
    input parameters are not checked.
    """

    r = direction_cosine(phi, theta, psi)
    po_alpha = np.arccos(np.dot(np.dot(r, po_surf_norm), po_ref_axis))

    return (
        march_dollase(po_alpha, po_r)
        * (1.0 / (8.0 * np.pi**2))
        * (
            np.abs(np.dot(v_s, np.dot(r.T, np.dot(r_t, np.dot(r, v_i))))).real
            ** 2
        )
        * np.sin(theta)
    )


def _powder_raman_intensities_with_odf_nquad(
    r_t, i_pol, s_pol, po_r, po_surf_norm, po_ref_axis
):
    """Calculate powder Raman intensities with numerical integration,
    including an orientation-distribution function, using the general
    `scipy.integrate.nquad()` routine.

    Parameters
    ----------
    r_t : array_like
        Raman tensor (shape: `(3, 3)`).
    i_pol, s_pol : Polarisation
        Polarisations of incident and scattered light.
    po_r : float
        March parameter for the March-Dollase function.
    po_surf_norm, po_ref_axis : array_like
        Surface normal and reference axis for preferred orientation
        (shape: `(3,)`).

    Notes
    -----
    This routine is part of the implementation of numerical powder Raman
    simulations and should not be used directly. In particular, the
    input parameters are not checked.
    """

    int_sum = 0.0

    for v_i, v_s, w in i_pol.combine_with_iter(s_pol):
        i, _ = nquad(
            _powder_raman_integrand_with_odf_nquad,
            [(0.0, 2.0 * np.pi), (0.0, np.pi), (0.0, 2.0 * np.pi)],
            [r_t, v_i, v_s, po_r, po_surf_norm, po_ref_axis],
        )

        int_sum += w * i

    return int_sum


# --------------------------------------------------------------------
# Powder raman: numerical integration with Levedev + circle quadrature
# --------------------------------------------------------------------


def _powder_raman_integrand_lebedev_circle(phi, theta, psi, r_t, v_i, v_s):
    """Integrand for calculating Raman intensities of ideal powders
    using Lebedev + circle quadrature.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles in Radians.
    r_t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered polarisation vectors (shape: `(3, 3)`).

    Notes
    -----
    This routine is part of the implementation of numerical powder Raman
    simulations and should not be used directly. In particular, the
    input parameters are not checked.
    """

    r = direction_cosine(phi, theta, psi)

    return (
        np.abs(np.dot(v_s, np.dot(r.T, np.dot(r_t, np.dot(r, v_i))))).real ** 2
    )


def _powder_raman_intensities_lebedev_circle(r_t, i_pol, s_pol, p):
    """Calculate powder Raman intensities with numerical integration
    using Lebedev + circle quadrature.

    Parameters
    ----------
    r_t : array_like
        Raman tensor (shape: `(3, 3)`).
    i_pol, s_pol : Polarisation
        Polarisations of incident and scattered light.
    p : int
        Precision of the Lebedev + circle quadrature.

    Notes
    -----
    This routine is part of the implementation of numerical powder Raman
    simulations and should not be used directly. In particular, the
    input parameters are not checked.
    """

    int_sum = 0.0

    for v_i, v_s, w in i_pol.combine_with_iter(s_pol):
        i = lebedev_circle_quad(
            _powder_raman_integrand_lebedev_circle,
            p,
            n=None,
            args=[r_t, v_i, v_s],
        )

        int_sum += w * i

    return int_sum


def _powder_raman_integrand_with_odf_lebedev_circle(
    phi, theta, psi, r_t, v_i, v_s, po_r, po_surf_norm, po_ref_axis
):
    """Integrand for calculating Raman intensities of powders with
    preferred orientation using Lebedev + circle quadrature.

    Parameters
    ----------
    phi, theta, psi : float
        Euler angles in Radians.
    r_t : array_like
        Raman tensor (shape: `(3, 3)`).
    v_i, v_s : array_like
        Incident and scattered polarisation vectors (shape: `(3, 3)`).
    po_r : float
        March parameter for the March-Dollase distribution function.
    po_surf_norm, po_ref_axis : array_like
        Surface normal and reference axis used to determine the angle
        `alpha` in the March-Dollase function.

    Notes
    -----
    This routine is part of the implementation of numerical powder Raman
    simulations and should not be used directly. In particular, the
    input parameters are not checked.
    """

    r = direction_cosine(phi, theta, psi)
    po_alpha = np.arccos(np.dot(np.dot(r, po_surf_norm), po_ref_axis))

    return march_dollase(po_alpha, po_r) * (
        np.abs(np.dot(v_s, np.dot(r.T, np.dot(r_t, np.dot(r, v_i))))).real ** 2
    )


def _powder_raman_intensities_with_odf_lebedev_circle(
    r_t, i_pol, s_pol, po_r, po_surf_norm, po_ref_axis, p
):
    """Calculate powder Raman intensities with numerical integration
    using Lebedev + circle quadrature.

    Parameters
    ----------
    r_t : array_like
        Raman tensor (shape: `(3, 3)`).
    i_pol, s_pol : Polarisation
        Polarisations of incident and scattered light.
    po_r : float
        March parameter for the March-Dollase function.
    po_surf_norm, po_ref_axis : array_like
        Surface normal and reference axis for preferred orientation
        (shape: `(3,)`).
    p : int
        Precision of the Lebedev + circle quadrature.


    Notes
    -----
    This routine is part of the implementation of numerical powder Raman
    simulations and should not be used directly. In particular, the
    input parameters are not checked.
    """

    int_sum = 0.0

    for v_i, v_s, w in i_pol.combine_with_iter(s_pol):
        i = lebedev_circle_quad(
            _powder_raman_integrand_with_odf_lebedev_circle,
            p,
            args=[r_t, v_i, v_s, po_r, po_surf_norm, po_ref_axis],
        )

        int_sum += w * i

    return int_sum


# -------------------------------------
# Powder Raman: "public" implementation
# -------------------------------------


def calculate_powder_raman_intensities(
    r_t,
    geom,
    i_pol,
    s_pol,
    pref_orient_eta=0.0,
    pref_orient_surf_norm=None,
    method="best",
    lebedev_prec=5,
):
    """Calculate the scalar Raman intensities for a polarised Raman
    measurement with powder averaging using the chosen method.

    Parameters
    ----------
    r_t : array_like
        Raman tensors (shape: `(3, 3)` or `(N, 3, 3)`).
    geom : Geometry
        Measurement geometry.
    i_pol, s_pol : Polarisation
        Polarisation of incident and scattered light.
    pref_orient_eta : float, optional
        Fractional excess of crystalites in preferred orientation
        (default: 0.0).
    pref_orient_surf_norm : array_like or str, optional
        Surface normal for preferred orientation (default: `None`,
        required if `pref_orient_eta` > 0).
    method : {"nquad", "lebedev+circle", "best"}
        Method for calculating intensnties.
    lebedev_prec : int
        Specifies the precision of the Lebedev/circle quadrature scheme
        for `method="lebedev+circle"`.

    Returns
    -------
    ints : float or numpy.ndarray
        Calcuated intensity or array of intensities (shape: `(N,)`).

    See Also
    --------
    calculate_powder_raman_intensities_analytical : Calculate powder
        Raman intensities using an analytical formula.
    calculate_powder_raman_intensities_numerical : Calculate powder
        Raman intensities using numerical integration.
    """

    r_t, _ = np_expand_dims(np.asarray(r_t), (None, 3, 3))

    # Determine whether an energy-dependent Raman calculation and/or
    # a calculation with a preferred orientation are required.

    complex_rt = np.iscomplex(r_t).any()
    pref_orient = np.abs(pref_orient_eta) > ZERO_TOLERANCE

    # If a preferred orientation is specified, the only guaranteed
    # safe method is numerical integration with the general SciPy
    # nquad() routine.

    if method == "nquad" or (pref_orient and method == "best"):
        return calculate_powder_raman_intensities_numerical(
            r_t,
            geom,
            i_pol,
            s_pol,
            pref_orient_eta=pref_orient_eta,
            pref_orient_surf_norm=pref_orient_surf_norm,
            method="nquad",
        )

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

    # If not, we will need to use numerical integration, but we
    # should be able to use more efficient Lebedev quadrature with a
    # suitable precision.

    if method == "lebedev+circle" or method == "best":
        return calculate_powder_raman_intensities_numerical(
            r_t,
            geom,
            i_pol,
            s_pol,
            pref_orient_surf_norm=pref_orient_surf_norm,
            pref_orient_eta=pref_orient_eta,
            method="lebedev+circle",
            lebedev_prec=lebedev_prec,
        )

    raise ValueError('Unknown method="{0}".'.format(method))


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
        raise Exception(
            "The analytical formula can only be used when the "
            "incident polarisation is perpendicular to the collection "
            "axis - use the numerical routines instead."
        )

    # Raman tensors calculated without using the far from resonance
    # (FFR) approximation can in general be complex. Since the
    # analytical formula was derived under the FFR, it can only be used
    # for real Raman tensors.

    if np.iscomplex(r_t).any():
        raise Exception(
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


def calculate_powder_raman_intensities_numerical(
    r_t,
    geom,
    i_pol,
    s_pol,
    pref_orient_eta=0.0,
    pref_orient_surf_norm=None,
    method="nquad",
    lebedev_prec=5,
):
    """_Calculate the scalar Raman intensities for a polarised Raman
    measurement with powder averaging using numerical integration.

    Parameters
    ----------
    r_t : array_like
        Raman tensors (shape: `(3, 3)` or `(N, 3, 3)`).
    geom : Geometry
        Measurement geometry.
    i_pol, s_pol : Polarisation
        Polarisations of incident and scattered light.
    pref_orient_eta : float, optional
        Fractional excess of crystalites in preferred orientation
        (default: 0.0).
    pref_orient_surf_norm : array_like or str, optional
        Surface normal for preferred orientation (default: `None`,
        required if `pref_orient_eta` > 0).
    method : {"nquad", "lebedev+circle", "best"}
        Method for calculating intensnties.
    lebedev_prec : int
        Specifies the precision of the Lebedev/circle quadrature scheme
        for `method="lebedev+circle"`.

    Returns
    -------
    ints : float or numpy.ndarray
        Calcuated intensity or array of intensities (shape: `(N,)`).
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

    # Different code paths depending on whether a preferred orientation
    # specified.

    pref_orient = np.abs(pref_orient_eta) > ZERO_TOLERANCE

    if pref_orient:
        if pref_orient_surf_norm is None:
            raise Exception(
                "For calculations with eta > 0, pref_orient_surf_norm "
                "must be supplied."
            )

        pref_orient_surf_norm = parse_direction(pref_orient_surf_norm)

    pref_orient_r = march_dollase_eta_to_r(pref_orient_eta)

    ints = np.zeros((r_t.shape[0],), dtype=np.float64)

    if method == "nquad":
        if pref_orient:
            for i, t in enumerate(r_t):
                ints[i] = _powder_raman_intensities_with_odf_nquad(
                    t,
                    i_pol,
                    s_pol,
                    pref_orient_r,
                    pref_orient_surf_norm,
                    -1.0 * geom.incident_direction,
                )
        else:
            for i, t in enumerate(r_t):
                ints[i] = _powder_raman_intensities_nquad(t, i_pol, s_pol)

    elif method == "lebedev+circle":
        if pref_orient:
            warnings.warn(
                "Powder Raman simulations with preferred orientation "
                "using the Lebedev + circle quadrature scheme require "
                "careful testing of the precision and should ideally "
                'be verified against the results with method="nquad".',
                UserWarning,
            )

            for i, t in enumerate(r_t):
                ints[i] = _powder_raman_intensities_with_odf_lebedev_circle(
                    t,
                    i_pol,
                    s_pol,
                    pref_orient_r,
                    pref_orient_surf_norm,
                    -1.0 * geom.incident_direction,
                    lebedev_prec,
                )
        else:
            for i, t in enumerate(r_t):
                ints[i] = _powder_raman_intensities_lebedev_circle(
                    t, i_pol, s_pol, lebedev_prec
                )

    else:
        raise Exception('Unknown method="{0}".'.format(method))

    return ints if n_dim_add == 0 else ints
