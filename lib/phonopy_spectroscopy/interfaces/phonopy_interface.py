# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Routines for interfacing with the Phono(3)py code."""


# -------
# Imports
# -------


import os
import warnings

import h5py

import numpy as np

_PHONOPY_AVAILABLE = False

try:
    from phonopy.file_IO import parse_BORN

    _PHONOPY_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Imports from phonopy failed - some functions require phonopy and "
        "will raise exceptions if it is not installed.",
        RuntimeWarning,
    )


from ..constants import ZERO_TOLERANCE
from ..irreps import Irreps
from ..phonon import GammaPhonons
from ..structure import Structure
from ..utility.io_helper import load_yaml

from .vasp_interface import structure_from_poscar


# -------------------
# High-level "loader"
# -------------------


def gamma_phonons_from_phono3py(
    cell_file,
    freqs_evecs_file,
    lws_file=None,
    lws_t=300.0,
    irreps_file=None,
    conv_trans=None,
):
    """Read a complete Phono(3)py calculation and return a
    `GammaPhonons` object.

    Parameters
    ----------
    cell_file : str
        VASP POSCAR or phonopy.yaml file to read structure from.
    freqs_evecs_file : str
        mesh.yaml, band.yaml, mesh.hdf5 or band.hdf5 file to read phonon
        frequencies/eigenvectors from.
    lws_file : str, optional
        kappa-m*.hdf5 file to read phonon linewidths from (default:
        `None`).
    lws_t : float, optional
        Temperature to read linewidths at (default: 300 K)
    irreps_file : str, optional
        irreps.yaml file to read irreps from (default: `None`).
    conv_trans : array_like, optional
        Transformation matrix to convert the structure to its
        conventional cell (shape: `(3, 3)`, default: `None`).

    Returns
    -------
    gamma_ph : `GammaPhonons`
        `GammaPhonons` object containing the calculation data.
    """

    # Read a structure. If cell_file has a .yaml extension, assume it
    # is a phonopy.yaml file; otherwise, assume it is a VASP POSCAR
    # file.

    struct = None

    _, ext = os.path.splitext(cell_file)

    if ext.lower() == ".yaml":
        struct = structure_from_phonopy_yaml(cell_file)
    else:
        struct = structure_from_poscar(cell_file)

    # If a transformation matrix is specified, create a new structure
    # with the conv_trans keyword set.

    if conv_trans is not None:
        struct = Structure(
            struct.lattice_vectors,
            struct.atom_positions,
            struct.atom_types,
            struct.atomic_masses,
            conv_trans=conv_trans,
        )

    # Read a set of frequencies/eigenvectors. If freqs_evecs_file has a
    # .yaml extension, assume it is a mesh.yaml/band.yaml file. If the
    # file has a .hdf5 extension, assume it is a mesh.hdf5/band.hdf5
    # file. Otherwise, raise an error and defer responsibility to the
    # calling code.

    freqs, evecs = None, None

    _, ext = os.path.splitext(freqs_evecs_file)

    if ext.lower() == ".yaml":
        freqs, evecs = gamma_freqs_evecs_from_mesh_or_band_yaml(
            freqs_evecs_file
        )
    elif ext.lower() == ".hdf5":
        freqs, evecs = gamma_freqs_evecs_from_mesh_or_band_hdf5(
            freqs_evecs_file
        )
    else:
        raise Exception(
            "Frequencies/eigenvectors file {0}: unknown format.".format(
                freqs_evecs_file
            )
        )

    # If a lws_file is specified, read linewidths; otherwise, set a
    # uniform linewidth of lw.

    lws = None

    if lws_file is not None:
        lws = gamma_linewidths_from_kappa_hdf5(lws_file, lws_t)

    # If irreps_file is specified, read irrep data.

    irreps = None

    if irreps_file is not None:
        irreps = irreps_from_irreps_yaml(irreps_file)

    # Construct and return a GammaPhonons object with the data. (The
    # GammaPhonons constructor will handle validation and consistency
    # checking.)

    return GammaPhonons(struct, freqs, evecs, lws=lws, irreps=irreps)


# ----------
# YAML files
# ----------


def structure_from_phonopy_yaml(file_path):
    """Read a structure from a phonopy.yaml file and return a
    `Structure` object.

    Parameters
    ----------
    file_path : str
        Input file.

    Returns
    -------
    struct : Structure
        `Structure` object containing the structure.
    """

    data = load_yaml(file_path)

    cell = data["primitive_cell"]

    return Structure(
        cell["lattice"],
        [atom["coordinates"] for atom in cell["points"]],
        [atom["symbol"] for atom in cell["points"]],
        [atom["mass"] for atom in cell["points"]],
    )


def gamma_freqs_evecs_from_mesh_or_band_yaml(file_path):
    r"""Read Gamma-point phonon frequencies and eigenvectors
    from a mesh.yaml or band.yaml file.

    Parameters
    ----------
    file_path : str
        Input file.

    Returns
    -------
    freqs_evecs : tuple of numpy.ndarray
        A `(freqs, evecs)` tuple of arrays with shapes `(3N,)` and
        `(3N, N, 3)`.
    """

    data = load_yaml(file_path)

    # Get Gamma-point frequencies and eigenvectors.

    for qpt in data["phonon"]:
        if np.allclose(qpt["q-position"], 0.0, atol=ZERO_TOLERANCE):
            # q = (0, 0, 0) = \Gamma.

            if "eigenvector" not in qpt["band"][0]:
                raise Exception(
                    "mesh.yaml/band.yaml file {0}: Eigenvectors not found."
                    "".format(file_path)
                )

            freqs = np.array(
                [mode["frequency"] for mode in qpt["band"]], dtype=np.float64
            )

            evecs = np.array(
                [mode["eigenvector"] for mode in qpt["band"]], dtype=np.float64
            )

            # The YAML files store the eigenvectors as complex numbers
            # so the initial list will have shape (3N, N, 3, 2). For
            # Gamma-point calculations, the imaginary part should be
            # zero, and we can drop the last dimension.

            if not np.allclose(evecs[:, :, :, 1], 0.0, atol=ZERO_TOLERANCE):
                raise Exception(
                    "mesh.yaml/band.yaml file {0}: One or more "
                    "Gamma-point eigenvectors has a non-zero "
                    "imaginary part.".format(file_path)
                )

            return (freqs, evecs[:, :, :, 0])

    raise Exception(
        "mesh.yaml/band.yaml file {0}: Gamma-point "
        "frequencies/eigenvectors not found.".format(file_path)
    )


def irreps_from_irreps_yaml(file_path):
    """Read Gamma-point mode irreducible representations
    (irreps) from an irreps.yaml file and return an `Irreps` object.

    Parameters
    ----------
    file_path : str
        File path.

    Returns
    -------
    irrep_data : Irreps
        `Irreps` object containing the irreps.
    """

    data = load_yaml(file_path)

    if not np.allclose(data["q-position"], 0.0, atol=ZERO_TOLERANCE):
        raise Exception(
            "irreps.yaml file {0}: Irreps are for a non-Gamma q."
            "".format(file_path)
        )

    return Irreps(
        str(data["point_group"]),
        [mode["ir_label"] for mode in data["normal_modes"]],
        [
            [idx - 1 for idx in mode["band_indices"]]
            for mode in data["normal_modes"]
        ],
    )


# ----------
# HDF5 files
# ----------


def gamma_freqs_evecs_from_mesh_or_band_hdf5(file_path):
    """Read Gamma-point phonon frequencies and eigenvectors from a
    mesh.hdf5 or band.hdf5 file.

    Parameters
    ----------
    file_path : str
        File path.

    Returns
    -------
    freqs_evecs : tuple of numpy.ndarray
        A `(freqs, evecs)` tuple of arrays with shapes `(3N,)` and
        `(3N, N, 3)`.
    """

    with h5py.File(file_path, "r") as f:
        if "eigenvector" not in f:
            raise Exception(
                "mesh.hdf5/band.hdf5 file {0}: Eigenvectors not found."
                "".format(file_path)
            )

        # mesh.hdf5 and band.hdf5 files have slightly different layouts.

        q_pts, freqs, evecs = None, None, None

        if "qpoint" in f:
            # mesh.hdf5 file.

            q_pts = f["qpoint"][:]
            freqs, evecs = f["frequency"][:], f["eigenvector"][:]
        elif "nqpoint" in f:
            path = f["path"][:]
            frequency = f["frequency"][:]

            n_seg, n_qpts, _ = path.shape
            _, _, n_bnd = frequency.shape

            q_pts = path.reshape((n_seg * n_qpts, 3))
            freqs = frequency.reshape((n_seg * n_qpts, n_bnd))

            evecs = f["eigenvector"][:].reshape((n_seg * n_qpts, n_bnd, n_bnd))
        else:
            raise Exception(
                "mesh.hdf5/band.hdf5 file {0}: Unknown data format."
                "".format(file_path)
            )

        for idx, q_pos in enumerate(q_pts):
            if np.allclose(q_pos, 0.0, atol=ZERO_TOLERANCE):
                freqs = freqs[idx]

                if not np.allclose(evecs[idx].imag, 0.0, atol=ZERO_TOLERANCE):
                    raise Exception(
                        "mesh.hdf5/band.hdf5 file {0}: One or more "
                        "Gamma-point eigenvectors has a non-zero "
                        "imaginary part.".format(file_path)
                    )

                n_at = len(freqs) // 3

                evecs = evecs[idx].real

                evecs = np.array(
                    [evecs[:, i].reshape(n_at, 3) for i in range(len(freqs))],
                    dtype=np.float64,
                )

                return (freqs, evecs)

    raise Exception(
        "mesh.hdf5/band.hdf5 file {0}: Gamma-point "
        "frequencies/eigenvectors not found.".format(file_path)
    )


def gamma_linewidths_from_kappa_hdf5(file_path, t=300.0):
    """Read Gamma-point linewidths at the specified temperature from a
    kappa-m*.hdf5 file.

    Parameters
    ----------
    file_path : str
        File path.

    Returns
    -------
    lws : numpy.ndarray
        Linewidths (shape: `(3N,)`).
    """

    with h5py.File(file_path, "r") as f:
        cond = f["temperature"][:] == t

        if cond.sum() != 1:
            raise Exception(
                "kappa-m*.hdf5 file {0}: Requested t = {1:.2f} not "
                "found.".format(file_path, t)
            )

        ((t_idx,),) = np.where(cond)

        if "qpoint" in f:
            # gamma has shape (n_t, n_q, 3 n_a).

            for q_idx, q_pos in enumerate(f["qpoint"]):
                if np.allclose(q_pos, 0.0, atol=ZERO_TOLERANCE):
                    return f["gamma"][t_idx, q_idx]
        else:
            # gamma has shape (n_t, 3 n_a).
            return f["gamma"][t_idx]

        raise Exception(
            "kappa-m*.hdf5 file {0}: Gamma-point linewidths not found."
            "".format(file_path)
        )


# ---------
# BORN file
# ---------


def hf_dielectric_and_born_from_born(file_path, struct):
    """Read the high-frequency dielectric constant and Born effective
    charges from a Phonopy BORN file and expand the charges for the
    supplied structure.

    Parameters
    ----------
    file_path : str
        File path.
    struct : Structure
        Crystal structure as a `Structure` object.

    Returns
    -------
    eps_born : tuple of numpt.ndarray
        A `(eps, born)` tuple with the high-frequency dielectric
        constant (shape: `(3, 3)`) and Born charges (shape:
        `(N, 3, 3)`).
    """

    if not _PHONOPY_AVAILABLE:
        raise Exception(
            "read_hf_dielectric_and_born_from_born() requires the "
            "phonopy.file.IO.parse_born function."
        )

    born_data = parse_BORN(struct.to_phonopy_atoms(), filename=file_path)
    return (born_data["dielectric"], born_data["born"])
