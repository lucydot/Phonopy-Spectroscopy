# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""This module contains helper routines for input/output."""


# -------
# Imports
# -------


import numpy as np

from scipy.interpolate import interp1d

from ..constants import ZERO_TOLERANCE
from ..interfaces.vasp_interface import dielectric_from_vasprun_xml


# ----------------
# Dielectric input
# ----------------


def raman_fd_read_dielectrics_vasp(file_list, num_bands, num_steps, e_tol=0.1):
    """Read dielectric constants or functions for a finite-differences
    Raman calculation, using linear interpolation to obtain a common
    energy axis if required.

    Parameters
    ----------
    file_list : array_like
        Files to read input data from.
    num_bands, num_steps : int
        Number of bands and displacement steps in the calculation.
    e_tol : float, optional
        Allowed tolerance on differences in average energy axis spacing.

    Returns
    -------
    dielectrics : tuple of ndarray
        `(e, eps_e)` tuple with photon energies (shape: `(O,)`) and
        dielectric tensors (shape: `(N, M, O, 3, 3)`) for N bands,
        M displacement steps and O photon energies.

    Notes
    -----
    The files in `file_list` should be ordered as:
        <Band00001-Step01>
        <Band00001-Step02>
        ...
        <Band00001-StepMM>
        ...
        <BandNNNNN-Step01>
        <BandMMMMM-StepMM>
    """

    if len(file_list) != num_bands * num_steps:
        raise ValueError(
            "Number of files in file_list is not consistent with the "
            "number of bands and displacement steps."
        )

    data = [dielectric_from_vasprun_xml(f) for f in file_list]

    # Find the smallest energy range for interpolating to a common
    # energy axis.

    e_ref = None

    for e, _ in data:
        if e_ref is None or e.max() < e_ref.max():
            e_ref = e

    num_pts = len(e_ref)

    d_e_ref = np.mean(e_ref[1:] - e_ref[:-1]) if num_pts > 1 else None

    for f, (e, _) in zip(file_list, data):
        if len(e) != num_pts:
            # This almost certainly indicates that the individual
            # calculations were performed with different parameters and
            # is likely a (user) mistake.

            raise Exception(
                "Input file {0} : Number of energies {1:,} differs "
                "from reference {2:,}.".format(f, len(e), num_pts)
            )

        if num_pts > 1:
            d_e = np.mean(e[1:] - e[:-1])

            if np.abs(d_e - d_e_ref) > e_tol:
                raise Exception(
                    "Input file {0} : Difference in energy spacing from "
                    "reference is {1:.3f} eV > {2:.3f} eV."
                    "".format(f, d_e, e_tol)
                )

    eps_e_interp = np.zeros(
        (num_bands, num_steps, len(e_ref), 3, 3), dtype=np.complex128
    )

    for i in range(num_bands):
        for j in range(num_steps):
            e, eps_e = data[i * num_steps + j]

            if np.allclose(e, e_ref, atol=ZERO_TOLERANCE):
                # Interpolation not required.
                eps_e_interp[i, j, :, :, :] = eps_e

            else:
                for k in range(3):
                    for l in range(3):
                        f = interp1d(e, eps_e[:, k, l], kind="linear")
                        eps_e_interp[i, j, :, k, l] = f(e_ref)

    return (e_ref, eps_e_interp)


# -----------
# Text export
# -----------


def df_to_txt(df, file_path, col_fmts, preamble_lines=None):
    """Write a Pandas `DataFrame` to a text file.

    Parameters
    ----------
    df : pandas.DataFrame
        `DataFrame` to write.
    preamble_lines : list of str
        Line(s) to write to the start of the output file (will be
        prepended with '# ').
    col_fmts : list of str
        Format specifiers for columns.
    file_path : str
        File to write to.
    """

    if len(col_fmts) != len(df.columns):
        raise Exception(
            "col_fmts must contain a format specifier for each column in df."
        )

    # Determine column widths as the maximum of the lengths of the
    # DataFrame headers and the lengths of the values in the first row
    # after formatting according to col_fmts.

    row_test = [fmt.format(v) for fmt, v in zip(col_fmts, df.iloc[0].values)]
    col_widths = [max(len(s), len(h)) for s, h in zip(row_test, df.columns)]

    with open(file_path, "w") as f:
        if preamble_lines is not None:
            for l in preamble_lines:
                f.write("# {0}\n".format(l))

        headers = [
            "{{0: >{0}}}".format(w).format(h)
            for w, h in zip(col_widths, df.columns)
        ]

        f.write("# " + "  ".join(headers) + "\n")

        for row in df.itertuples(index=False, name=None):
            row = [
                "{{0: >{0}}}".format(w).format(fmt.format(v))
                for w, fmt, v in zip(col_widths, col_fmts, row)
            ]

            f.write("  " + "  ".join(row) + "\n")


def raman_peak_table_to_txt(df, x_units, y_units, file_path):
    """Write a peak table produced by one of the `Spectrum` classes
    to a plain-text file.

    Parameters
    ----------
    df : pandas.DataFrame
        `DataFrame` produced by the `Spectrum` object.
    file_path : str
        File to write data to.
    """

    col_fmts = ["{0: >12.5f}", "{0: >10.5f}", "{0: >5}"]
    col_fmts += ["{0: >12.5e}"] * (len(df.columns) - 3)

    df_to_txt(df, file_path, col_fmts, preamble_lines=[x_units, y_units])


def raman_spectrum_to_txt(df, x_units, y_units, file_path):
    """Write a spectrum produced by one of the `Spectrum` classes
    to a plain-text file.

    Parameters
    ----------
    df : pandas.DataFrame
        `DataFrame` produced by the `Spectrum` object.
    file_path : str
        File to write data to.
    """

    col_fmts = ["{0: >12.5f}"] + ["{0: >10.5e}"] * (len(df.columns) - 1)

    df_to_txt(df, file_path, col_fmts, preamble_lines=[x_units, y_units])
