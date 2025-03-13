# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Defines (mathematical) distributions."""


# -------
# Imports
# -------


import numpy as np

from .constants import BOLTZMANN_CONSTANT_EV, PLANCK_CONSTANT_EV
from .utility.numpy_helper import np_expand_dims


# ----------
# Lineshapes
# ----------


def gaussian(x, i, mu, sigma):
    r"""Evaluate the Gaussian lineshape :math:`G(x, i, \mu, \sigma)`
    with the supplied intensity (peak area) `i`, mean `mu` (centre) and
    standard deviation `sigma` (width).

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the function.
    i, mu, sigma : float
        Intensity, mean and standard deviation.

    Returns
    -------
    g_x : numpy.ndarray
        Gaussian function evaluated at `x`.

    Notes
    -----
    The definition of :math:`G(x, i, \mu, \sigma)` is taken from:
    http://mathworld.wolfram.com/GaussianFunction.html.

    .. math::

        G(x) = \frac{i}{\sqrt{2\pi} \sigma} \exp{-\frac{(x - \mu)^2}{2 \sigma^2}}
    """

    x = np.asarray(x)

    return (i / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -1.0 * (x - mu) ** 2 / (2 * sigma**2)
    )


def lorentzian(x, i, x0, gamma):
    r"""Evaluate the Lorentzian lineshape :math:`L(x, i, x_0, \Gamma)`
    with intensity (peak area) `i`, central value `x0` and width
    `gamma`.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the function.
    i, x0, gamma : float
        Intensity, central frequency and width.

    Returns
    -------
    l_x : numpy.ndarray
        Lorentzian function evaluated at `x`.

    Notes
    -----
    The definition of :math:`L(x, i, x_0, \Gamma)` is taken from:
    http://mathworld.wolfram.com/LorentzianFunction.html.

    .. math::

        L(x) = \frac{i}{\pi} \sigma} \frac{\frac{1}{2} \Gamma}{(x - x_0)^2 + (\frac{1}{2} \Gamma)^2}
    """

    x = np.asarray(x)

    return (i / np.pi) * ((0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2))


# -------------
# Bose-Einstein
# -------------


def phonon_occupation_number(nu, t):
    """Evaluate the phonon occupation number :math:`n(\nu, t)` for
    frequencies `nu` and temperature `t` using the Bose-Einstein
    distribution.

    Parameters
    ----------
    nu : float or array_like
        Phonon frequency or frequencies in THz.
    t : float
        Temperature in K.

    Returns
    -------
    n : float or array_like
        Phonon occupation number(s) (same shape as `nu`).
    """

    nu, n_dim_add = np_expand_dims(np.asarray(nu), (None,))

    e = PLANCK_CONSTANT_EV * 1.0e12 * nu
    n = 1.0 / (np.exp(e / (BOLTZMANN_CONSTANT_EV * t)) - 1.0)

    return n if n_dim_add == 0 else n[0]


# ---------------------
# Preferred orientation
# ---------------------


def march_dollase(alpha, r):
    r"""Evaluate the March-Dollase distribution function `TODO` with the
    supplied angle `alpha` and March parameter `r`.

    Parameters
    ----------
    alpha : float
        Angle (radians).
    r : float
        March parameter.

    Returns
    -------
    v : float
        Value of the March-Dollase function.

    Notes
    -----
    TODO
    """

    return (r**2 * np.cos(alpha) ** 2 + (1.0 / r) * np.sin(alpha) ** 2) ** (
        -3.0 / 2.0
    )


def march_dollase_eta_to_r(eta):
    """Convert a crystallte excess fraction `eta` to the corresponding
     March parameter `r` in the March-Dollase distribution function.

    Parameters
    ----------
    eta : float
        Crystallite fraction.

    Returns
    -------
    r : float
        March parameter.
    """

    if eta < 0.0 or eta >= 1.0:
        raise ValueError(
            "eta must be >= 0 and < 1. eta = 1 causes the "
            "March-Dollase function to diverge."
        )

    return (
        (-0.5 * eta**2)
        + (np.sqrt(3.0) * eta * np.sqrt(1.0 - 0.25 * eta**2))
        - 1.0
    ) / (eta**2 - 1.0)
