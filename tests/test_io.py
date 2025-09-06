# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""This file contains test routines for the core I/O routines."""


# -------
# Imports
# -------


import os
import unittest

import numpy as np

from phonopy_spectroscopy.phonon import GammaPhonons
from phonopy_spectroscopy.structure import Structure

from phonopy_spectroscopy.utility.io_helper import load_json, save_json

from phonopy_spectroscopy.interfaces.vasp_interface import (
    structure_from_poscar,
    structure_to_poscar,
    dielectric_from_vasprun_xml,
)

from phonopy_spectroscopy.interfaces.phonopy_interface import (
    gamma_phonons_from_phono3py,
    structure_from_phonopy_yaml,
    gamma_freqs_evecs_from_mesh_or_band_yaml,
    gamma_freqs_evecs_from_mesh_or_band_hdf5,
)

from comparison_helper import compare_structures, compare_gamma_phonons


# ---------
# Constants
# ---------


_EXAMPLE_BASE_DIR = r"../example/Si"


# -------------
# Tests for I/O
# -------------


class TestIO(unittest.TestCase):
    def test_structure_io(self):
        """Test "core" structure input/output routines."""

        # Load reference structures from VASP POSCAR and Phonopy
        # phonopy.yaml files and compare.

        struct_ref_1 = structure_from_poscar(
            os.path.join(_EXAMPLE_BASE_DIR, r"POSCAR.Opt.Prim")
        )

        struct_ref_2 = structure_from_phonopy_yaml(
            os.path.join(_EXAMPLE_BASE_DIR, r"phonopy.yaml")
        )

        self.assertTrue(compare_structures(struct_ref_2, struct_ref_1))

        # Write a structure to a VASP POSCAR file, read it back, and
        # ensure the two Structure objects are equivalent.

        structure_to_poscar(struct_ref_1, r"POSCAR.vasp.tmp")
        struct_cmp = structure_from_poscar(r"POSCAR.vasp.tmp")

        self.assertTrue(compare_structures(struct_cmp, struct_ref_1))

        os.remove(r"POSCAR.vasp.tmp")

        # Serialise the structure to a dictionary, write it to a JSON
        # file, reload and recreate it, and check the Structure objects
        # are equivalent.

        save_json(struct_ref_1.to_dict(), r"structure.json.tmp")
        struct_cmp = Structure.from_dict(load_json(r"structure.json.tmp"))

        self.assertTrue(compare_structures(struct_cmp, struct_ref_1))

        os.remove(r"structure.json.tmp")

    def test_freqs_evecs_io(self):
        """Test routines for reading phonon frequencies and eigenvectors
        from Phonopy calculations."""

        # Load frequencies and eigenvectors from mesh/band YAML and HDF5
        # files and check the data are eqivalent.

        freqs_evecs_1 = gamma_freqs_evecs_from_mesh_or_band_yaml(
            os.path.join(_EXAMPLE_BASE_DIR, r"mesh.yaml")
        )

        freqs_evecs_2 = gamma_freqs_evecs_from_mesh_or_band_yaml(
            os.path.join(_EXAMPLE_BASE_DIR, r"band.yaml")
        )

        freqs_evecs_3 = gamma_freqs_evecs_from_mesh_or_band_hdf5(
            os.path.join(_EXAMPLE_BASE_DIR, r"mesh.hdf5")
        )

        freqs_evecs_4 = gamma_freqs_evecs_from_mesh_or_band_hdf5(
            os.path.join(_EXAMPLE_BASE_DIR, r"band.hdf5")
        )

        freqs_ref, evecs_ref = freqs_evecs_1

        for freqs_cmp, evecs_cmp in (
            freqs_evecs_2,
            freqs_evecs_3,
            freqs_evecs_4,
        ):
            self.assertTrue(np.allclose(freqs_cmp, freqs_ref))
            self.assertTrue(np.allclose(evecs_cmp, evecs_ref))

    def test_gamma_phonons_io(self):
        """Test high-level Phono(3)py "loader" and
        serialisation/deserialisation of `GammaPhonons` objects."""

        # Test the construction of a complete GammaPhonons object
        # including a structure, frequencies/eigenvectors, linewidths
        # and irreps.

        gamma_ph = gamma_phonons_from_phono3py(
            os.path.join(_EXAMPLE_BASE_DIR, r"phonopy.yaml"),
            os.path.join(_EXAMPLE_BASE_DIR, r"mesh.hdf5"),
            lws_file=os.path.join(_EXAMPLE_BASE_DIR, r"kappa-m646464-g0.hdf5"),
            irreps_file=os.path.join(_EXAMPLE_BASE_DIR, r"irreps.yaml"),
        )

        # Serialise the GammaPhonons to a dictionary, write it to a JSON
        # file, reload and recreate it, and check the two objects
        # are equivalent.

        save_json(gamma_ph.to_dict(), r"gamma_phonons.json.tmp")

        gamma_ph_cmp = GammaPhonons.from_dict(
            load_json(r"gamma_phonons.json.tmp")
        )

        self.assertTrue(compare_gamma_phonons(gamma_ph_cmp, gamma_ph))

        os.remove(r"gamma_phonons.json.tmp")

    def test_dielectric_io(self):
        """Test routines for reading dielectric data from vasprun.xml
        files."""

        input_files = [
            r"vasprun-PBEsol-DFPT.xml",
            r"vasprun-PBEsol-FiniteField.xml",
            r"vasprun-PBEsol-LinearOptics.xml",
            r"vasprun-r2SCAN-FiniteField.xml",
            r"vasprun-r2SCAN-LinearOptics.xml",
            r"vasprun-mBJ-LinearOptics.xml",
            r"vasprun-HSE06-FiniteField.xml",
            r"vasprun-HSE06-LinearOptics.xml",
        ]

        ref_eps_hf = [
            13.49657154,
            13.27554306,
            14.2169,
            11.78495469,
            11.8552,
            10.3019,
            11.09544159,
            10.3564,
        ]

        for f, ref_eps in zip(input_files, ref_eps_hf):
            e, eps_e = dielectric_from_vasprun_xml(
                os.path.join(_EXAMPLE_BASE_DIR, r"raman_ref", f)
            )

            # First energy should be E = 0.

            self.assertTrue(np.isclose(e[0], 0.0))

            # Dielectric constant at E = 0 should be real.

            self.assertFalse(np.iscomplex(eps_e[0]).any())

            # Si is isotropic so the three diagonal elements should be
            # equal by symmetry.

            self.assertTrue((np.diag(eps_e[0]) == ref_eps).all())


# ----
# Main
# ----


if __name__ == "__main__":
    unittest.main()
