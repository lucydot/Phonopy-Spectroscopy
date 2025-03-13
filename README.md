# Phonopy-Spectroscopy

Phonopy-Spectroscopy adds the capability to simulate vibrational spectra to the Phonopy code.[[1](#Ref1)]

The software package consists of a Python package, `phonopy_spectroscopy`, and command-line scripts for performing typical calculations.


## Upgrade

The current status of the development code is as follows:

* The "backend" Python API has been completely rewritten with a new object model.
* The Raman capability has been expanded to include complete functionality for simulating single-crystal and powder Raman experiments, including with energy-dependent Raman tensors.
* Unit tests have been implemented with reasonable code coverage.
* The infrared (IR) capability has been temporarily removed pending a rewrite.
* The command-line programs and associated API have been temporarily removed pending a rewrite.


## Installation

The code requires a typical scientific computing Python stack with the `NumPy`[[2](#Ref3)], `SciPy`[[3](#Ref3)], `Pandas`[[4](#Ref4)], `PyYaml`[[5](#Ref5)] and `H5py`[[6](#Ref6)] packages.
The Phonopy interface additionally requires the `Phonopy` Python library[[1](#Ref1)].

We recommend using Anaconda (or Miniconda)[[7](#Ref7)] and installing the required packages in a virtual environment:

```bash
% conda create -n phonopy-spectroscopy
% conda activate phonopy-spectroscopy
% conda install -c conda-forge phonopy pandas
```

This code does not currently ship with a `setup.py` script or as a PIP/Conda package.
After cloning or downloading and unpacking the repository, add the `lib` subfolder to your `PYTHONPATH`, e.g.:

`export PYTHONPATH=${PYTHONPATH}:/Users/user/Desktop/Repositories/Phonopy-Spectroscopy/lib`

If you wish, you can run some of the `test_*.py` files in the `/tests` subfolder to check your installation is working.


## Citation

Paper(s) on the new implementation and code are planned in the very near future.
For now, the citation for the previous version of Phonopy-Spectroscopy is:

J. M. Skelton, L. A. Burton, A. J. Jackson, F. Oba, S. C. Parker and A. Walsh, "Lattice dynamics of the tin sulphides SnS<sub>2</sub>, SnS and Sn<sub>2</sub>S<sub>3</sub>: vibrational spectra and thermal transport", *Physical Chemistry Chemical Physics* **19**, 12452 (**2017**), DOI: [10.1039/C7CP01680H](https://doi.org/10.1039/C7CP01680H) (open access)

If you use Phonopy-Spectroscopy in your work, please consider citing this paper and/or including a link to this GitHub repository when you publish your results.


## References

1. <a name="Ref1"></a>[https://atztogo.github.io/phonopy](https://atztogo.github.io/phonopy)
2. <a name="Ref2"></a>[https://www.numpy.org](https://www.numpy.org)
3. <a name="Ref3"></a>[https://scipy.org](https://scipy.org)
4. <a name="Ref4"></a>[https://pandas.pydata.org](https://pandas.pydata.org)
5. <a name="Ref5"></a>[https://pyyaml.org/wiki/PyYAML](https://pyyaml.org/wiki/PyYAML)
6. <a name="Ref6"></a>[https://www.h5py.org](https://www.h5py.org)
7. <a name="Ref7"></a>[https://www.anaconda.com](https://www.anaconda.com)
