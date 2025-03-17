# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Class and routines for storing and working with crystal structures.
"""


# -------
# Imports
# -------

import warnings

import numpy as np

from .constants import ZERO_TOLERANCE

from .utility.numpy_helper import (
    np_asarray_copy,
    np_readonly_view,
    np_check_shape,
    np_expand_dims,
)

_PHONOPY_AVAILABLE = False

try:
    from phonopy.structure.atoms import PhonopyAtoms, atom_data

    _PHONOPY_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Imports from phonopy failed - some functions require phonopy "
        "and will raise exceptions if it is not installed.",
        RuntimeWarning,
    )


# ---------
# Functions
# ---------


def cartesian_to_fractional_coordinates(cart_pos, latt_vecs):
    """Convert positions from Cartesian to fractional coordinates.

    Parameters
    ----------
    cart_pos : array_like
        Atom position or set of positions in Cartesian coordinates
        (shape: `(3,)` or `(N, 3)`).
    latt_vecs : array_like
        Lattice vectors (shape: `(3, 3)`).

    Returns
    -------
    frac_pos : numpy.ndarray
        Atom positions in fractional coordinates (same shape as
        `cart_pos`).
    """

    cart_pos, n_dim_add = np_expand_dims(np.asarray(cart_pos), (None, 3))

    latt_vecs = np.asarray(latt_vecs)

    if not np_check_shape(latt_vecs, (3, 3)):
        raise ValueError("latt_vecs must be an array_like with shape (3, 3).")

    trans_mat = np.linalg.inv(latt_vecs)

    frac_pos = np.zeros_like(cart_pos)

    for i, p in enumerate(cart_pos):
        frac_pos[i] = np.dot(p, trans_mat) % 1.0

    return frac_pos if n_dim_add == 0 else frac_pos[0]


def fractional_to_cartesian_coordinates(frac_pos, latt_vecs):
    """Convert positions from fractional to Cartesian coordinates.

    Parameters
    ----------
    frac_pos : array_like
        Atom position or set of positions in fractional coordinates
        (shape: `(3,)` or `(N, 3)`).
    latt_vecs : array_like
        Lattice vectors (shape: `(3, 3)`).

    Returns
    -------
    cart_pos : numpy.ndarray
        Atom positions in Cartesian coordinates (same shape as
        `frac_pos`).
    """

    frac_pos, n_dim_add = np_expand_dims(np.asarray(frac_pos), (None, 3))

    latt_vecs = np.asarray(latt_vecs)

    if not np_check_shape(latt_vecs, (3, 3)):
        raise ValueError("latt_vecs must be an array_like with shape (3, 3).")

    cart_pos = np.zeros_like(frac_pos)

    for i, p in enumerate(frac_pos):
        cart_pos[i] = np.dot(latt_vecs, p)

    return cart_pos if n_dim_add == 0 else cart_pos[0]


def lookup_atomic_mass(symbol):
    """Lookup an atomic mass from an atomic symbol.

    Parameters
    ----------
    symbol : str
        Atomic symbol.

    Returns
    -------
    m : float
        Atomic mass (amu).

    Notes
    -----
    This function requires the `phonopy` package.
    """

    if not _PHONOPY_AVAILABLE:
        raise Exception(
            "lookup_atomic_mass() requires the "
            "phonopy.atoms.atom_data attribute."
        )

    symbol = str(symbol).title()

    for _, db_symbol, _, db_mass in atom_data:
        if symbol == db_symbol:
            return db_mass

    raise ValueError(
        'Data for symbol="{0}" not available in '
        "phonopy.atoms.atom_data.".format(symbol)
    )


# ---------------
# Structure class
# ---------------


class Structure:
    def __init__(
        self,
        latt_vecs,
        at_pos,
        at_typ,
        at_m=None,
        prim_trans=None,
        cart_to_frac=False,
    ):
        """Create a new instance of the `Structure` class.

        Parameters
        ----------
        latt_vecs : array_like
            Lattice vectors (shape: `(3, 3)`).
        at_pos : array_like
            Atomic positions (shape: `(N, 3)`).
        at_typ : array_like
            Atom types (shape: `(N,)`).
        at_m : array_like, optional
            Atomic masses (optional, shape: `(N,)`).
        prim_trans : array_like, optional
            Specifies a transformation from a conventional cell to this
            structure.
        cart_to_frac : bool, optional
            If `True`, convert `at_pos` from Cartesian to fractional
            coordinates (default: `False`).
        """

        latt_vecs = np_asarray_copy(latt_vecs, dtype=np.float64)

        if not np_check_shape(latt_vecs, (3, 3)):
            raise ValueError(
                "latt_vecs must be an array_like with shape (3, 3)."
            )

        at_pos = np_asarray_copy(at_pos, dtype=np.float64)
        at_typ = np.array([str(typ) for typ in at_typ], dtype=object)

        if len(at_pos) > 0 and not np_check_shape(at_pos, (None, 3)):
            raise ValueError("at_pos must be an array_like with shape (N, 3).")

        n_a = len(at_pos)

        if not np_check_shape(at_typ, (n_a,)):
            raise ValueError("at_typ must be an array_like with shape (N,).")

        if at_m is None:
            at_m = np.array(
                [lookup_atomic_mass(sym) for sym in at_typ], dtype=np.float64
            )
        else:
            at_m = np_asarray_copy(at_m, dtype=np.float64)

            if not np_check_shape(at_m, (n_a,)):
                raise ValueError(
                    "If supplied, at_m must be an array_like with shape (N,)."
                )

        if prim_trans is not None:
            prim_trans = np_asarray_copy(prim_trans, dtype=np.float64)

            if not np_check_shape(prim_trans, (3, 3)):
                raise ValueError(
                    "If supplied, prim_trans must be an array_like "
                    "with shape (3, 3)."
                )

        if n_a > 0:
            if cart_to_frac:
                at_pos = cartesian_to_fractional_coordinates(at_pos, latt_vecs)
            else:
                if (np.abs(at_pos) > 1.0).any():
                    warnings.warn(
                        "One or more of at_pos are outside the range "
                        "[-1, 1] expected for fractional coordinates - "
                        "use cart_to_frac=True to convert if needed.",
                        UserWarning,
                    )

        self._v_latt = latt_vecs
        self._at_pos = at_pos
        self._at_typ = at_typ
        self._at_m = at_m
        self._prim_trans = prim_trans

    @property
    def lattice_vectors(self):
        """numpy.ndarray : Lattice vectors (shape: `(3, 3)`)."""
        return np_readonly_view(self._v_latt)

    @property
    def atom_positions(self):
        """numpy.ndarray : Atomic positions (shape: `(N, 3)`)."""
        return np_readonly_view(self._at_pos)

    @property
    def atom_types(self):
        """numpy.ndarray : Atom types (shape: `(N,)`)."""
        return np_readonly_view(self._at_typ)

    @property
    def atomic_masses(self):
        """numpy.ndarray : Atomic masses (shape: `(N,)`)."""
        return np_readonly_view(self._at_m)

    @property
    def primitive_transformation_matrix(self):
        """numpy.ndarray : Transformation matrix from the conventional
        cell to this structure."""
        return np_readonly_view(self._prim_trans)

    @property
    def num_atoms(self):
        """int : Number of atoms in the structure."""
        return self._at_pos.shape[0]

    @property
    def conventional_cell_defined(self):
        """bool : `True` if the structure has a conventional cell
        defined, otherwise `False`."""
        return self._prim_trans is not None

    def _get_v_latt_for_cell_definition(self, cell="prim"):
        """Return the lattice vectors for the cell definition `cell`.

        Returns
        -------
        v_latt : numpy.ndarray
            Lattice vectors for cell definition (shape: `(3, 3)`).
        """

        cell = cell.lower()

        if cell == "prim":
            return self._v_latt.view()

        elif cell == "conv":
            if self._prim_trans is None:
                raise Exception(
                    "The conventional cell can only be used if a "
                    "primitive transformation matrix was specified at "
                    "initialisation."
                )

            conv_trans = np.linalg.inv(self._prim_trans)

            # A valid primitive transformation should have an inverse
            # with integer values.

            abs_diff = np.abs(np.rint(conv_trans) - conv_trans)

            if (abs_diff > ZERO_TOLERANCE).any():
                warnings.warn(
                    "One or more elements in the inverse of the "
                    "primitive transformation matrix deviates from "
                    "integer values by up to {0:.3e}. This could "
                    "indicate an invalid tranformation matrix or "
                    "inadequate precision.".format(abs_diff.max())
                )

            return np.dot(conv_trans, self._v_latt)

        else:
            raise ValueError(
                'Unknown cell definition cell="{0}".'.format(cell)
            )

    def volume(self, cell="prim"):
        """Calculate the unit-cell volume.

        Parameters
        ----------
        cell : {"prim", "conv"}, optional
            Compute volume for the primitive ("prim") or conventional
            unit cell ("conv") (default: "prim"). `cell="conv"` is only
            valid if a primitive transformation matrix was supplied at
            construction.

        Returns
        -------
        v : float
            Unit-cell volume.
        """

        v_1, v_2, v_3 = self._get_v_latt_for_cell_definition(cell=cell)
        return np.dot(v_1, np.cross(v_2, v_3))

    def reciprocal_lattice_vectors(self, cell="prim"):
        """Calculate and return the reciprocal lattice vectors.

        Parameters
        ----------
        cell : {"prim", "conv"}, optional
            Compute vectors for the primitive ("prim") or conventional
            unit cell ("conv") (default: "prim"). `cell="conv"` is only
            valid if a primitive transformation matrix was supplied at
            construction.

        Returns
        -------
        recip_latt_vec : numpy.ndarray
            Recipocal lattice vectors (shape: `(3, 3)`).
        """

        a_1, a_2, a_3 = self._get_v_latt_for_cell_definition(cell=cell)
        v = self.volume(cell=cell)

        return np.array(
            [
                np.cross(a_2, a_3) / v,
                np.cross(a_3, a_1) / v,
                np.cross(a_1, a_2) / v,
            ],
            dtype=np.float64,
        )

    def real_space_normal(self, hkl, cell="prim"):
        """Calculate the real-space normal to the surface with Miller
        index `hkl`.

        Parameters
        ----------
        hkl : array_like
            Integer Miller indices of the surface (shape: `(3,)`).,
        cell : {"prim", "conv"}, optional
            Specify the Miller indices in terms of the primitive
            ("prim") or conventional cell ("conv") (default: "prim"").
            `cell="conv"` is only valid if a primitive transformation
            matrix was supplied at construction.

        Returns
        -------
        norm : numpy.ndarray
            Real-space sufrace normal in Cartesian coordinates (shape:
            `(3,)`).
        """

        hkl = np.asarray(hkl)

        if not np_check_shape(hkl, (3,)):
            raise ValueError("hkl must be an array_like with shape `(3,)`.")

        b_1, b_2, b_3 = self.reciprocal_lattice_vectors(cell=cell)

        # Use the reciprocal metric tensor to obtain the real-space
        # normal in fractional coordinates.

        recip_metric = np.array(
            [
                [np.dot(b_1, b_1), np.dot(b_1, b_2), np.dot(b_1, b_3)],
                [np.dot(b_2, b_1), np.dot(b_2, b_2), np.dot(b_2, b_3)],
                [np.dot(b_3, b_1), np.dot(b_3, b_2), np.dot(b_3, b_3)],
            ]
        )

        norm_frac = np.dot(recip_metric, hkl)

        norm_cart = fractional_to_cartesian_coordinates(
            norm_frac, self._get_v_latt_for_cell_definition(cell=cell)
        )

        return norm_cart / np.linalg.norm(norm_cart)

    def cartesian_positions(self):
        """Return the atomic positions converted to Cartesian
        coordinates.

        Returns
        -------
        pos_cart : numpy.ndarray
            Atom positions in Cartesian coordinates (shape: `(N, 3)`).
        """

        return fractional_to_cartesian_coordinates(self._at_pos, self._v_latt)

    def to_phonopy_atoms(self):
        """Return the structure as a `PhonopyAtoms` instance.

        Returns
        -------
        atoms : PhonopyAtoms
            `PhonopyAtoms` object containing the structure data.

        Notes
        -----
        This function requires the `phonopy` package.
        """

        if not _PHONOPY_AVAILABLE:
            raise Exception(
                "Structure.to_phonopy_atoms() requires the "
                "phonopy.structure.PhonopyAtoms class."
            )

        return PhonopyAtoms(
            cell=self.lattice_vectors,
            scaled_positions=self.atom_positions,
            symbols=self.atom_types,
            masses=self.atomic_masses,
        )

    def to_dict(self):
        """Return the internal data as a dictionary of native Python
        types for serialisation.

        Returns
        -------
        d : dict
            Dictionary structure containing internal data as native
            Python types.
        """

        prim_trans = (
            self._prim_trans.tolist() if self._prim_trans is not None else None
        )

        return {
            "lattice_vectors": self._v_latt.tolist(),
            "atom_positions": self._at_pos.tolist(),
            "atom_types": list(self._at_typ),
            "atomic_masses": self._at_m.tolist(),
            "primitive_transformation_matrix": prim_trans,
        }

    @staticmethod
    def from_dict(d):
        """Create a new `Structure` instance from a dictionary
        generated by `Structure.to_dict()`.

        Parameters
        ----------
        d : dict
            Dictionary generated by `to_dict()`.

        Returns
        -------
        struct : Structure
            `Structure` object constructed from the data in `d`.
        """

        return Structure(
            d["lattice_vectors"],
            d["atom_positions"],
            d["atom_types"],
            at_m=d["atomic_masses"],
            prim_trans=d["primitive_transformation_matrix"],
        )
