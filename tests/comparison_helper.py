# -*- coding: utf-8 -*-


# ---------
# Docstring
# ---------

"""This module contains routines for comparing objects as part of unit
tests."""


# -------
# Imports
# -------

import numpy as np


# ---------
# Functions
# ---------


def compare_structures(struct_cmp, struct_ref):
    """Compare two `Structure` objects and determine whether they hold
    identical data.

    Parameters
    ----------
    struct_cmp, struct_ref : Structure
        `Structure` objects to compare.

    Returns
    -------
    equiv : bool
        `True` if `struct_cmp` is equivalent to `struct_ref`, otherwise
        `False`.
    """

    return (
        np.allclose(struct_cmp.lattice_vectors, struct_ref.lattice_vectors)
        and np.allclose(struct_cmp.atom_positions, struct_ref.atom_positions)
        and np.equal(struct_cmp.atom_types, struct_ref.atom_types).all()
        and np.allclose(struct_cmp.atomic_masses, struct_ref.atomic_masses)
        and np.allclose(
            struct_cmp.conventional_transformation_matrix,
            struct_ref.conventional_transformation_matrix,
        )
    )


def compare_irreps(irreps_cmp, irreps_ref):
    """Compare two `Irreps` objects and determine
    whether they hold identical data.

    Parameters
    ----------
    irreps_cmp, irreps_ref : Irreps
        `Irreps` objects to compare.

    Returns
    -------
    equiv : bool
        `True` if `irreps_cmp` is equivalent to `irreps_ref`,
        otherwise `False`.
    """

    if irreps_cmp.point_group != irreps_ref.point_group:
        return False

    if len(irreps_cmp.irrep_symbols) != len(irreps_ref.irrep_symbols):
        return False

    # IrreducibleRepresentations constructor should enforce that
    # the irrep symbols and irrep_band_indices properties are the same
    # length.

    for sym_cmp, sym_ref in zip(
        irreps_cmp.irrep_symbols, irreps_ref.irrep_symbols
    ):
        if sym_ref != sym_cmp:
            return False

    for band_inds_cmp, band_inds_ref in zip(
        irreps_cmp.irrep_band_indices, irreps_ref.irrep_band_indices
    ):
        if (band_inds_cmp != band_inds_ref).any():
            return False

    return True


def compare_gamma_phonons(gamma_ph_cmp, gamma_ph_ref):
    """Compare two `GammaPhonons` objects and determine whether they
    hold identical data.

    Parameters
    ----------
    gamma_ph_cmp, gamma_ph_ref : GammaPhonons
        `GammaPhonons` objects to compare.

    Returns
    -------
    equiv : bool
        `True` if `gamma_ph_cmp` is equivalent to `gamma_ph_ref`,
        otherwise `False`.
    """

    if not compare_structures(gamma_ph_cmp.structure, gamma_ph_ref.structure):
        return False

    return (
        np.allclose(gamma_ph_cmp.frequencies, gamma_ph_ref.frequencies)
        and np.allclose(gamma_ph_cmp.eigenvectors, gamma_ph_ref.eigenvectors)
        and np.allclose(gamma_ph_cmp.linewidths, gamma_ph_ref.linewidths)
        and compare_irreps(gamma_ph_cmp.irreps, gamma_ph_ref.irreps)
    )


def compare_infrared_calculators(calc_cmp, calc_ref):
    """Compare two `InfraredCalculator` objects and determine whether
    they hold identical data.

    Parameters
    ----------
    calc_cmp, calc_ref : InfraredCalculator
        `InfraredCalculator` objects to compare.

    Returns
    -------
    equiv : bool
        `True` if `calc_cmp` is equivalant to `calc_ref`, otherwise
        `False`.
    """

    if not compare_gamma_phonons(
        calc_ref.gamma_phonons, calc_cmp.gamma_phonons
    ):
        return False

    if not np.allclose(
        calc_cmp.born_effective_charges, calc_ref.born_effective_charges
    ):
        return False

    if calc_cmp.epsilon_inf is not None:
        return calc_ref.epsilon_inf is not None and np.allclose(
            calc_cmp.epsilon_inf, calc_ref.epsilon_inf
        )
    else:
        return calc_ref.born_charges is None


def compare_finite_displacement_raman_tensor_calculators(
    fd_calc_cmp, fd_calc_ref
):
    """Compare two `FiniteDisplacementRamanTensorCalculator` objects and
    determine whether they hold identical data.

    Parameters
    ----------
    fd_calc_cmp, fd_calc_ref : FiniteDisplacementRamanTensorCalculator
        `FiniteDisplacementRamanTensorCalculator` objects to compare.

    Returns
    -------
    equiv : bool
        `True` if `fd_calc_cmp` is equivalant to `fd_calc_ref`,
        otherwise `False`.
    """

    if not compare_gamma_phonons(
        fd_calc_ref.gamma_phonons, fd_calc_cmp.gamma_phonons
    ):
        return False

    return (
        np.equal(fd_calc_ref.band_indices, fd_calc_cmp.band_indices).all()
        and np.allclose(
            fd_calc_ref.displacement_steps, fd_calc_cmp.displacement_steps
        )
        and np.allclose(
            fd_calc_ref.step_coefficients, fd_calc_cmp.step_coefficients
        )
    )


def compare_raman_tensors(r_t_cmp, r_t_ref):
    """Compare two `RamanTensors` objects and determine whether they
    hold identical data.

    Parameters
    ----------
    r_t_cmp, r_t_ref : RamanTensors
        `RamanTensors` objects to compare.

    Returns
    -------
    equiv : bool
        `True` if `r_t_cmp` is equivalant to `r_t_ref`, otherwise
        `False`.
    """

    return (
        np.allclose(r_t_cmp.energies, r_t_ref.energies),
        np.allclose(r_t_cmp.raman_tensors, r_t_ref.raman_tensors),
    )


def compare_raman_calculations(calc_cmp, calc_ref):
    """Compare two `RamanCalculation` objects and determine whether they
    hold identical data.

    Parameters
    ----------
    calc_cmp, calc_ref : RamanCalculation
        `RamanCalculation` objects to compare.

    Returns
    -------
    equiv : bool
        `True` if `calc_cmp` is equivalant to `calc_ref`, otherwise
        `False`.
    """

    return (
        compare_gamma_phonons(calc_cmp.gamma_phonons, calc_ref.gamma_phonons)
        and compare_raman_tensors(
            calc_cmp.raman_tensors, calc_ref.raman_tensors
        )
        and np.equal(calc_cmp.band_indices, calc_ref.band_indices).all()
        and compare_irreps(calc_cmp.irreps, calc_ref.irreps)
    )
