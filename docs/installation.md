# Installation

### Dependencies

The code requires a typical scientific computing Python stack with the [`Numpy`](https://www.numpy.org), [`SciPy`](https://scipy.org), [`Pandas`](https://pandas.pydata.org), [`PyYaml`](https://pyyaml.org/wiki/PyYAML) and [`H5py`](https://www.h5py.org) packages.
The Phonopy interface additionally requires the [`Phonopy`](https://github.com/phonopy/phonopy) Python library.

If you are new to Python, we recommend using [Anaconda]((https://www.anaconda.com) (or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)) and installing the required packages in a virtual environment:

```bash
% conda create -n phonopy-spectroscopy
% conda activate phonopy-spectroscopy
% conda install -c conda-forge phonopy pandas
```

### Phonopy-Spectroscopy

This code does not currently ship with a `setup.py` script or as a PIP/Conda package. This is planned for the near future.

After cloning or downloading and unpacking the repository, add the `lib` subfolder to your `PYTHONPATH`, e.g.:

`export PYTHONPATH=${PYTHONPATH}:/Users/user/Desktop/Repositories/Phonopy-Spectroscopy/lib`

### Verifying your installation

If you wish, you can run some of the `test_*.py` files in the `/tests` subfolder to check your installation is working.