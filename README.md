# About

pyebsd is an python implemented open-source tool for processing Electron Backscattered Diffraction (EBSD) data. The main implemented features are:

- pole figures
- inverse pole figures for cubic crystals
- accurate orientation relationship for cubic crystals
- misorientation

pyebsd is in development stage and, for now, is only able to process phases of cubic symmetry. Its main use case is steel microstructures.

pyebsd is a "pure" Python package, meaning that it does not depend on building extension modules. For computational expensive calculations, such as manipulation of matrices, it relies on the clever vectorized operations with NumPy.

# Installation

pyebsd runs in Python 3 (>= 3.5). Earlier versions of pyebsd still work on python 2, but python 2 support has been deprecated.

pyebsd is not available yet in the Python Package Index. In order to install the library, first download the repository:

```bash
git clone https://github.com/arthursn/pyebsd
```

Then install pyebsd by running pip with the cloned `./pyebsd` folder (local repository) as argument:

```bash
pip install ./pyebsd
```

# Basic usage

Examples can be found in the [examples](https://github.com/arthursn/pyebsd/tree/master/examples) folder. A jupyter notebook with interactive examples is provided [here](https://github.com/arthursn/pyebsd/blob/master/examples/plot_EBSD_maps.ipynb).

Load EBSD data:

```python
import matplotlib.pyplot as plt
import pyebsd

# So far, pyebsd only supports loading .ang files generated
# by the TSL OIM software
scan = pyebsd.load_scandata('path/to/ang/file')
```

Plot Inverse Pole Figure (IPF) map:

```python
# gray="IQ" is used to set the quality index as grayscale
scan.plot_IPF(gray="IQ")
plt.show()
```

Plot phase map:

```python
scan.plot_phase(gray="IQ")
plt.show()
```

Plot pole figure:

```python
scan.plot_PF()
plt.show()
```
