# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Class for storing and working with phonon calculations."""


# -------
# Imports
# -------


import warnings

import numpy as np

from .constants import VASP_TO_THZ, ZERO_TOLERANCE
from .irreps import Irreps
from .structure import Structure

from .utility.numpy_helper import (
    np_asarray_copy,
    np_readonly_view,
    np_check_shape,
)

_PHONOPY_AVAILABLE = False

try:
    from phonopy import Phonopy
    from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants

    _PHONOPY_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Imports from phonopy failed - some functions require phonopy "
        "and will raise exceptions if it is not installed.",
        RuntimeWarning,
    )


# ------------------
# GammaPhonons class
# ------------------


class GammaPhonons:
    """Class for storing and working with a Gamma-point phonon
    calculation."""

    def __init__(self, struct, freqs, evecs, lws=None, irreps=None):
        r"""Create a new instance of the `GammaPhonons` class.

        Parameters
        ----------
        struct : Structure
            Crystal structure.
        freqs : array_like
            Phonon frequencies in THz (shape: `(3N,)`).
        evecs : array_like
            Phonon eigenvectors in mass-weighted units of sqrt(amu)
            (shape: `(3N, N, 3)`).
        lws : array_like or None, optional
            Phonon linewidths in THz (shape: `(3N,)`) or `None`.
        irreps : Irreps or None, optional
            `Irreps` object specifying the point group and assigning
            bands to irrep groups.
        """

        n_a = struct.num_atoms

        if n_a == 0:
            raise ValueError(
                'struct cannot be "empty" and must contain at least one atom.'
            )

        freqs = np_asarray_copy(freqs, dtype=np.float64)

        if not np_check_shape(freqs, (3 * n_a,)):
            raise ValueError("freqs must be an array_like with shape (3N,).")

        evecs = np_asarray_copy(evecs, dtype=np.float64)

        if not np_check_shape(evecs, (3 * n_a, n_a, 3)):
            raise ValueError(
                "evecs must be an array_like with shape (3N, N, 3)."
            )

        # Eigenvectors are in general complex, but Gamma-point
        # eigenvectors must be real.

        if np.iscomplex(evecs).any():
            if (np.abs(evecs.imag) > ZERO_TOLERANCE).any():
                raise ValueError("Gamma-point eigenvectors should be real.")

            evecs = evecs.real

        if lws is not None:
            lws = np_asarray_copy(lws, dtype=np.float64)

            if not np_check_shape(lws, (3 * n_a,)):
                raise ValueError(
                    "If supplied, lws must be an array_like with shape (3N,)."
                )

            if (lws < 0.0).any():
                raise ValueError("Linewidths cannot be negative.")

        if irreps is not None:
            ir_band_inds_flat = irreps.band_indices_flat()

            if len(ir_band_inds_flat) != 3 * n_a:
                raise ValueError(
                    "If supplied, irreps must assign all bands to "
                    "irrep groups."
                )

            if ir_band_inds_flat.max() >= (3 * n_a):
                raise RuntimeError(
                    "One or more band indices in irreps are not "
                    "compatible with the number of modes in the "
                    "phonon calculation."
                )

        self._struct = struct
        self._freqs = freqs
        self._evecs = evecs
        self._lws = lws
        self._irreps = irreps

    @property
    def structure(self):
        """Structure : Crystal structure."""
        return self._struct

    @property
    def frequencies(self):
        """numpy.ndarray : Phonon frequencies (shape: `(3N,)`)."""
        return np_readonly_view(self._freqs)

    @property
    def eigenvectors(self):
        """numpy.ndarray : Phonon eigenvectors (shape: `(3N, N, 3)`)."""
        return np_readonly_view(self._evecs)

    @property
    def linewidths(self):
        """numpy.ndarray or None : Phonon linewidths (shape: `(3N,)`)."""
        return np_readonly_view(self._lws) if self._lws is not None else None

    @property
    def num_modes(self):
        """int : Number of modes."""
        return len(self._freqs)

    @property
    def has_linewidths(self):
        """bool : `True` if linewidths are available, otherwise `False`."""
        return self._lws is not None

    @property
    def has_irreps(self):
        """bool : `True` if irredicuble representations (irreps) are
        available, otherwise `False`.
        """
        return self._irreps is not None

    @property
    def irreps(self):
        """Irreps or None : `Irreps` object with the point group and
        irrep symbols and indices of band groups."""
        return self._irreps

    def get_acoustic_mode_indices(self):
        """Return the band indices of the acoustic modes.

        Returns
        -------
        band_inds : numpy.ndarray
            Band indices of the acoustic modes.
        """

        # If we have irreps, select irrep group(s) for which the
        # average frequency is closest to f = 0 until we have chosen
        # sufficient groups to cover three modes.

        if self._irreps is not None:
            ir_ave_freqs = [
                np.mean(self._freqs[band_inds])
                for band_inds in self._irreps.irrep_band_indices
            ]

            subset_band_inds = []

            for _, band_inds in sorted(
                zip(ir_ave_freqs, self._irreps.irrep_band_indices)
            ):
                subset_band_inds.extend(band_inds)

                if len(subset_band_inds) == 3:
                    return np.array(subset_band_inds, dtype=int)

            raise RuntimeError(
                "Unable to select a set of acoustic modes spanning "
                "complete irrep groups. This may indicate an issue "
                "with the phonon calculation."
            )

        # If not, find the three modes with frequencies closest to
        # f = 0.

        return np.argsort(np.abs(self._freqs))[:3]

    def eigendisplacements(self):
        r"""Return the phonon eigendisplacements (eigenvectors divided
        by sqrt(mass)).

        Returns
        -------
        edisps : numpy.ndarray
            Eigendisplacements (shape: `(3N, N, 3)`).
        """

        sqrt_m = np.sqrt(self._struct.atomic_masses)
        return self._evecs / sqrt_m[np.newaxis, :, np.newaxis]

    def hessian(self):
        r"""Calculate the Hessian matrix (force constants) for the
        Gamma-point phonon modes.

        Returns
        -------
        h : numpy.ndarray
            Hessian in eV / Ang^2 (shape: `(3N, 3N)`).

        Notes
        -----
        This function requires the `phonopy` package.
        """

        if not _PHONOPY_AVAILABLE:
            raise RuntimeError(
                "GammaPhonons.hessian() requires the "
                "phonopy.Phonopy and phonopy.phonon.DynmatToFc class."
            )

        n_a = self._struct.num_atoms

        evals = np.copysign((self._freqs / VASP_TO_THZ) ** 2, self._freqs)

        evec_mat = np.zeros((3 * n_a, 3 * n_a), dtype=np.complex128)

        for i, evec in enumerate(self._evecs):
            evec_mat[:, i].real = evec.flat

        # Construct a Phonopy object to obtain the primitive cell and
        # "supercell".

        phonopy = Phonopy(self._struct.to_phonopy_atoms(), np.eye(3, 3))

        # Construct a DynmatToForceConstants object to reverse transform
        # the dynamical matrix to the corresponding force constants.

        d2f = DynmatToForceConstants(phonopy.primitive, phonopy.supercell)

        d2f.create_dynamical_matrices(
            eigenvalues=[evals], eigenvectors=[evec_mat]
        )

        d2f.run()

        fc2 = d2f.force_constants

        # "Flatten" fc2 to a (3 n_a) x (3 n_a) matrix.

        h = np.zeros((3 * n_a, 3 * n_a), dtype=np.float64)

        for i_at in range(n_a):
            i_fc2 = 3 * i_at

            for j_at in range(n_a):
                j_fc2 = 3 * j_at

                h[i_fc2 : i_fc2 + 3, j_fc2 : j_fc2 + 3] = fc2[i_at, j_at]

        return h

    def to_dict(self):
        """Return the internal data as a dictionary of native Python
        types for serialisation.

        Returns
        -------
        d : dict
            Dictionary structure containing internal data as native
            Python types.
        """

        lws = self._lws.tolist() if self._lws is not None else None
        irreps = self._irreps.to_dict() if self._irreps is not None else None

        return {
            "structure": self._struct.to_dict(),
            "frequencies": self._freqs.tolist(),
            "eigenvectors": self._evecs.tolist(),
            "linewidths": lws,
            "irreps": irreps,
        }

    @staticmethod
    def from_dict(d):
        """Create a new `GammaPhonons` instance from a dictionary
        generated by `GammaPhonons.to_dict()`.

        Parameters
        ----------
        d : dict
            Dictionary generated by `to_dict()`.

        Returns
        -------
        gamma_phonons : GammaPhonons
            `GammaPhonons` object constructed from the data in `d`.
        """

        irreps = None

        if d["irreps"] is not None:
            irreps = Irreps.from_dict(d["irreps"])

        return GammaPhonons(
            Structure.from_dict(d["structure"]),
            d["frequencies"],
            d["eigenvectors"],
            lws=d["linewidths"],
            irreps=irreps,
        )
