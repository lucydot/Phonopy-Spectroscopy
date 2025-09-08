# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Routines for reading and writing files not associated with specific
codes."""


# -------
# Imports
# -------


import numpy as np

from ..structure import Structure


# ---------
# XYZ Files
# ---------


def structure_from_xyz(file_path, cubic=True, pad=15.0):
    r"""Read coordinates from an XYZ-format file and return a `Structure`
    object with the molecule placed at the centre of a large unit cell.

    Parameters
    ----------
    file_path : str
        Path to input file.
    cubic : bool, optional
        Selects a cubic (`True`) or rectangular cell (`False`) (default:
        `True`).
    pad : float, optional
        Minimum spacing between atoms in periodic images (default: 15
        Ang).

    Returns
    -------
    struct : Structure
        `Structure` object with the molecule placed at the centre of a
        cubic (`cubic=True`) or rectangular cell (`cubic=False`) with a
        minimum distance of `pad` between atoms in periodic images.
    """

    if pad < 0.0:
        raise ValueError("pad cannot be less than zero.")

    with open(file_path, "r") as f:
        # Read atom count.

        n_at = int(next(f).strip())

        # Skip title line.

        next(f)

        # Read atom data.

        at_syms, at_pos = [], []

        for _ in range(n_at):
            vals = next(f).strip().split()

            at_syms.append(vals[0])
            at_pos.append(np.array([float(v) for v in vals[1:4]]))

        # Determine "bounding box" for molecular structure.

        at_pos = np.array(at_pos, dtype=np.float64)

        p_min = at_pos.min(axis=0)
        p_max = at_pos.max(axis=0)

        m_box = p_max - p_min

        # Determine cell box and lattice parameters.

        c_box = None

        if cubic:
            c_box = (m_box.max() + pad) * np.ones((3,), dtype=np.float64)
        else:
            c_box = m_box + pad

        a, b, c = c_box

        v_latt = np.array(
            [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]], dtype=np.float64
        )

        # Shift atoms to place molecule at the centre of the cell.

        at_pos += ((c_box / 2.0) - p_min)[np.newaxis, :]

        # Build and return a Structure object.

        return Structure(v_latt, at_pos, at_syms, cart_to_frac=True)
