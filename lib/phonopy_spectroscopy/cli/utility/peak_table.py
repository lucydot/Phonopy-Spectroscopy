# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Helper routines for outputting peak tables and spectra to plain-text
files, for implementing the command-line interface."""


# -------
# General
# -------


def df_to_txt(df, file_path, col_fmts, preamble_lines=None):
    """Write a Pandas `DataFrame` to a text file.

    Parameters
    ----------
    df : pandas.DataFrame
        `DataFrame` to write.
    preamble_lines : list of str
        Line(s) to write to the start of the output file (will be
        prepended with "# ").
    col_fmts : list of str
        Format specifiers for columns.
    file_path : str
        File to write to.
    """

    if len(col_fmts) != len(df.columns):
        raise ValueError(
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


# ----------------
# Infrared spectra
# ----------------


def infrared_peak_table_to_txt(dielectric_func, file_path):
    """Write a peak table from a `ScalarInfraredDielectricFunction` or
    `TensorInfraredDielectricFunction` object to a plain-text file.

    Parameters
    ----------
    dielectric_func : ScalarInfraredDielectricFunction or TensorInfraredDielectricFunction
        Dielectric function.
    file_path : str
        File to write data to.
    """

    df = dielectric_func.peak_table()

    preamble_lines = [
        "x : {0}".format(dielectric_func.x_unit_text_label),
        "y : {0}".format(
            dielectric_func.mode_oscillator_strength_unit_text_label
        ),
    ]

    col_fmts = ["{0: >12.5f}", "{0: >10.5f}", "{0: >5}"]
    col_fmts += ["{0: >12.5e}"] * (len(df.columns) - 3)

    df_to_txt(df, file_path, col_fmts, preamble_lines=preamble_lines)


def tensor_infrared_spectrum_to_txt(dielectric_func, file_path):
    """Write the dielectric function from a
    `TensorInfraredDielectricFunction` object to a plain-text file.

    Parameters
    ----------
    dielectric_func : TensorInfraredDielectricFunction
        Dielectric function.
    file_path : str
        File to write data to.
    """

    df = dielectric_func.spectrum()

    preamble_lines = [
        "x : {0}".format(dielectric_func.x_unit_text_label),
        "y : {0}".format(dielectric_func.epsilon_unit_text_label),
    ]

    col_fmts = ["{0: >12.5f}"] + ["{0: >12.5e}"] * (len(df.columns) - 1)

    df_to_txt(df, file_path, col_fmts, preamble_lines=preamble_lines)


def scalar_infrared_spectrum_to_txt(dielectric_func, file_path):
    """Write the dielectric function and derived quantities from a
    `ScalarInfraredDielectricFunction` object to a plain-text file.

    Parameters
    ----------
    dielectric_func : ScalarInfraredDielectricFunction
        Dielectric function.
    file_path : str
        File to write data to.
    """

    df = dielectric_func.spectrum()

    preamble_lines = [
        "x : {0}".format(dielectric_func.x_unit_text_label),
        "y (eps) : {0}".format(dielectric_func.epsilon_unit_text_label),
        "y (alpha) : {0}".format(
            dielectric_func.absorption_coefficient_unit_text_label
        ),
        "y (R) : dimensionless",
        "y (L) : dimensionless",
    ]

    col_fmts = ["{0: >12.5f}"] + ["{0: >12.5e}"] * (len(df.columns) - 1)

    df_to_txt(df, file_path, col_fmts, preamble_lines=preamble_lines)


# -------------
# Raman spectra
# -------------


def _raman_get_preamble_lines(sp):
    """Return a list of lines to be written to the "preamble" of a
    plain-text file with data from a `RamanSpectrum1D` or
    `RamanSpectrum2D` object.

    Parameters
    ----------
    sp : RamanSpectrum1D or RamanSpectrum2D
        Raman spectrum.

    Returns
    -------
    preamble_lines : list of str
        Preamble lines to be written to output file.
    """

    preamble_lines = [
        "x: {0}".format(sp.x_unit_text_label),
        "y: {0}".format(sp.y_unit_text_label),
    ]

    if hasattr(sp, "z"):
        preamble_lines.append("z: {0}".format(sp.z_unit_text_label))

    return preamble_lines


def raman_peak_table_to_txt(sp, file_path):
    """Write a peak table from a  a `RamanSpectrum1D` or
    `RamanSpectrum2D` object to a plain-text file.

    Parameters
    ----------
    sp : RamanSpectrum1D or RamanSpectrum2D
        Raman spectrum.
    file_path : str
        File to write data to.
    """

    df = sp.peak_table()

    col_fmts = ["{0: >12.5f}", "{0: >10.5f}", "{0: >5}"]
    col_fmts += ["{0: >12.5e}"] * (len(df.columns) - 3)

    df_to_txt(
        df, file_path, col_fmts, preamble_lines=_raman_get_preamble_lines(sp)
    )


def raman_spectrum_to_txt(sp, file_path):
    """Write a spectrum from a  a `RamanSpectrum1D` or `RamanSpectrum2D`
    object to a plain-text file.

    Parameters
    ----------
    sp : RamanSpectrum1D or RamanSpectrum2D
        Raman spectrum.
    file_path : str
        File to write data to.
    """

    df = sp.spectrum()

    col_fmts = ["{0: >12.5f}"] + ["{0: >10.5e}"] * (len(df.columns) - 1)

    df_to_txt(
        df, file_path, col_fmts, preamble_lines=_raman_get_preamble_lines(sp)
    )
