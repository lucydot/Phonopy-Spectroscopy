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

from .constants import ZERO_TOLERANCE
from .irreps import Irreps
from .structure import Structure

from .utility.numpy_helper import (
    np_asarray_copy,
    np_readonly_view,
    np_check_shape,
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
            if irreps.band_indices_flat().max() >= (3 * n_a):
                raise Exception(
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

            raise Exception(
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

    def to_dict(self):
        """Return the internal data as a dictionary of native Python
        types for serialisation.

        Returns
        -------
        d : dict
            Dictionary structure containing internal data as native
            Python types.
        """

        return {
            "structure": self._struct.to_dict(),
            "frequencies": self._freqs.tolist(),
            "eigenvectors": self._evecs.tolist(),
            "linewidths": (
                self._lws.tolist() if self._lws is not None else None
            ),
            "irreps": (
                self._irreps.to_dict() if self._irreps is not None else None
            ),
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

        return GammaPhonons(
            Structure.from_dict(d["structure"]),
            d["frequencies"],
            d["eigenvectors"],
            lws=d["linewidths"],
            irreps=Irreps.from_dict(d["irreps"]),
        )
