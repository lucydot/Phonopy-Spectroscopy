# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------


"""Test routines for the Raman calculation workflow."""


# -------
# Imports
# -------


import os
import unittest

import numpy as np

from phonopy_spectroscopy.cli.utility.raman_io import (
    fd_read_dielectrics_vasp,
)

from phonopy_spectroscopy.interfaces.phonopy_interface import (
    gamma_phonons_from_phono3py,
)

from phonopy_spectroscopy.interfaces.vasp_interface import (
    structure_from_poscar,
)

from phonopy_spectroscopy.raman.calculation import RamanCalculation

from phonopy_spectroscopy.raman.finite_diff import (
    FiniteDisplacementRamanTensorCalculator,
)

from phonopy_spectroscopy.utility.io_helper import load_json, save_json

from comparison_helper import (
    compare_structures,
    compare_finite_displacement_raman_tensor_calculators,
    compare_raman_calculations,
)

from io_helper import generate_fd_raman_dielectric_input_file_list


# ---------
# Constants
# ---------


_EXAMPLE_BASE_DIR = r"../example/si"


# ------------------------------------------------
# Tests for finite-displacement Raman calculations
# ------------------------------------------------


class TestFiniteDisplacementRamanCalculations(unittest.TestCase):
    def setUp(self):
        """Perform setup."""

        # Load a Gamma-point phonon calculation using the high-level
        # Phono(3)py "loader" function.

        gamma_ph = gamma_phonons_from_phono3py(
            os.path.join(_EXAMPLE_BASE_DIR, r"phonopy.yaml"),
            os.path.join(_EXAMPLE_BASE_DIR, r"mesh.hdf5"),
            lws_file=os.path.join(_EXAMPLE_BASE_DIR, r"kappa-m646464-g0.hdf5"),
            irreps_file=os.path.join(_EXAMPLE_BASE_DIR, r"irreps.yaml"),
        )

        # Construct a FiniteDisplacementRamanTensorCalculator object.

        self._calc = FiniteDisplacementRamanTensorCalculator(gamma_ph)

    def test_finite_diff_struct_ser_des(self):
        """Test serialisation/deserialisation of the
        `FiniteDisplacementRamanTensorCalculator` class."""

        # Serialise the calculator to a dictionary, write it to a JSON
        # file, reload and recreate it, and check the two objects
        # are equivalent.

        save_json(self._calc.to_dict(), r"raman_calculator.json.tmp")

        calc_cmp = FiniteDisplacementRamanTensorCalculator.from_dict(
            load_json(r"raman_calculator.json.tmp")
        )

        self.assertTrue(
            compare_finite_displacement_raman_tensor_calculators(
                self._calc, calc_cmp
            )
        )

        os.remove(r"raman_calculator.json.tmp")

    def test_finite_diff_struct_gen(self):
        """Test generation of displaced structures using the
        `FiniteDisplacementRamanTensorCalculator` class."""

        # Generate the displaced structures and check against reference
        # structures.

        disp_struct_sets = self._calc.generate_displaced_structures()

        for band_idx, disp_structs in zip(
            self._calc.band_indices, disp_struct_sets
        ):
            for step_idx, (_, disp_struct) in enumerate(
                zip(self._calc.displacement_steps, disp_structs)
            ):
                file_path = os.path.join(
                    _EXAMPLE_BASE_DIR,
                    r"raman_ref",
                    r"POSCAR-{0:0>4}-{1:0>2}".format(
                        band_idx + 1, step_idx + 1
                    ),
                )

                struct_ref = structure_from_poscar(file_path)
                self.assertTrue(compare_structures(disp_struct, struct_ref))

    def test_finite_diff_raman_tensor_calc_1(self):
        """Test calculation of Raman tensors with high-frequency
        dielectric constants ("far from resonance" approximation)."""

        # Load dielectric constant calculations.

        file_path_template = os.path.join(
            _EXAMPLE_BASE_DIR,
            r"raman_ref",
            r"vasprun-PBEsol-DFPT-{0:0>4}-{1:0>2}.xml",
        )

        file_list = generate_fd_raman_dielectric_input_file_list(
            self._calc.band_indices, self._calc.num_steps, file_path_template
        )

        e, eps_e = fd_read_dielectrics_vasp(
            file_list, self._calc.num_bands, self._calc.num_steps
        )

        # Calculate and check Raman tensors.

        raman_calc = self._calc.calculate_raman_tensors(eps_e, e)

        r_t = raman_calc.raman_tensors

        self.assertTrue(np.allclose(r_t.energies, e))

        exp_shape = (self._calc.num_bands, len(e), 3, 3)
        self.assertTrue(np.equal(r_t.raman_tensors.shape, exp_shape).all())

        self.assertFalse(r_t.is_energy_dependent)

        # Serialise the RamanCalcuator object to a dictionary, write it
        # to a JSON file, reload it, and check the two objects are
        # equivalent.

        save_json(raman_calc.to_dict(), r"raman_calculation.json.tmp")

        raman_calc_cmp = RamanCalculation.from_dict(
            load_json(r"raman_calculation.json.tmp")
        )
        self.assertTrue(compare_raman_calculations(raman_calc_cmp, raman_calc))

        os.remove(r"raman_calculation.json.tmp")

    def test_finite_diff_raman_tensor_calc_2(self):
        """Test calculation of Raman tensors with energy-dependent
        dielectric functions."""

        file_path_template = os.path.join(
            _EXAMPLE_BASE_DIR,
            r"raman_ref",
            r"vasprun-PBEsol-LinearOptics-{0:0>4}-{1:0>2}.xml",
        )

        file_list = generate_fd_raman_dielectric_input_file_list(
            self._calc.band_indices, self._calc.num_steps, file_path_template
        )

        e, eps_e = fd_read_dielectrics_vasp(
            file_list, self._calc.num_bands, self._calc.num_steps
        )

        raman_calc = self._calc.calculate_raman_tensors(eps_e, e)

        r_t = raman_calc.raman_tensors

        self.assertTrue(np.allclose(r_t.energies, e))

        exp_shape = (self._calc.num_bands, len(e), 3, 3)
        self.assertTrue(np.equal(r_t.raman_tensors.shape, exp_shape).all())

        self.assertTrue(r_t.is_energy_dependent)

        # The first set of Raman tensors should have E = 0 and should
        # be real.

        self.assertTrue(np.isclose(r_t.energies[0], 0.0))
        self.assertTrue(np.allclose(r_t.raman_tensors[:, 0, :, :].imag, 0.0))

        # Test serialising/deserialising the RamanTensors object again.
        # This is slightly different to the previous test, as the
        # RamanTensors object contains energy-dependent tensors that
        # are complex-valued.

        save_json(raman_calc.to_dict(), r"raman_calculation.json.tmp")

        raman_calc_cmp = RamanCalculation.from_dict(
            load_json(r"raman_calculation.json.tmp")
        )

        self.assertTrue(compare_raman_calculations(raman_calc_cmp, raman_calc))

        os.remove(r"raman_calculation.json.tmp")


# ----
# Main
# ----


if __name__ == "__main__":
    unittest.main()
