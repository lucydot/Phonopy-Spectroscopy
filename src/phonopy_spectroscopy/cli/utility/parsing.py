# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Helper routines for parsing command-line arguments for implementing
the command-line interface."""


# -------
# Imports
# -------


import numpy as np


# ---------
# Functions
# ---------


def parse_frac(inp):
    """Parse a value specified as a fraction using the `/` character to
    sepearate the numerator and denominator.

    Parameters
    ----------
    inp : str
        Input string to parse.

    Returns
    -------
    val : float
        Parsed value.
    """

    if "/" in inp:
        num, denom = inp.strip().split("/")
        return float(num) / float(denom)

    return float(inp.strip())


def parse_3x3_matrix(inp):
    """Parse a 3x3 matrix from an input string.

    Parameters
    ----------
    inp : str
        Input string to parse.

    Returns
    -------
    mat : numpy.ndarray
        Parsed matrix (shape: (3, 3)).

    See Also
    --------
    parse_frac : Parse a value specified as a fraction.

    Notes
    -----
    inp may contain one, three, six or nine values, any of which may be
    specified as fractions, which are interpreted as follows:
      * 1 value: `xx = yy = zz = vals[0]`
      * 3 values: `xx = vals[0]`z`, `yy = vals[1]`, `zz = vals[2]`
      * 6 values: `xx = vals[0]`, `xy = yx = vals[1]`,
        `xz = zx = vals[2]`, `yy = vals[3]`, `yz = zy = vals[4]`,
        `zz = vals[5]`
      * 9 values: `xx, xy, xz, yx, yy, yz, zx, zy, zz = vals[:]`

    If fractional values are used, whitespace between the numerator and
    denominator will result in incorrect parsing and/or an error.
    """

    vals = [parse_frac(val) for val in inp.strip().split()]

    val_inds = None

    if len(vals) == 1:
        # Single value specified -> assume equal diagonals and
        # zero off-diagonals.

        val_inds = [(0, i, i) for i in range(3)]

    elif len(vals) == 3:
        # Diagonals specified -> assume zero off-diagonals.
        val_inds = [3 * (i,) for i in range(3)]

    elif len(vals) == 6:
        # Upper triangle specified -> assume symmetric about diagonal.
        # xx, xy, xz, yy, yz, zz

        val_inds = (
            [(0, 0, 0), (1, 0, 1), (2, 0, 2)]
            + [(1, 1, 0), (3, 1, 1), (4, 1, 2)]
            + [(2, 2, 0), (4, 2, 1), (5, 2, 2)]
        )

    elif len(vals) == 9:
        # All elements specified.
        val_inds = [(i, i // 3, i % 3) for i in range(9)]

    if val_inds is not None:
        mat = np.zeros((3, 3), dtype=np.float64)

        for i, idx, idy in val_inds:
            mat[idx, idy] = vals[i]

        return mat

    raise ValueError("inp must specify one, three, six or nine values.")
