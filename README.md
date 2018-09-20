# About

pyebsd is an python implemented open-source tool for processing Electron Backscattered Diffraction (EBSD) data. The main implemented features are:

- pole figures
- inverse pole figures for cubic crystals
- accurate orientation relationship for cubic crystals
- misorientation

# Installation and requirements

pyebsd runs in python 2 (>=2.7) and python 3 (>= 3.5) and uses the following non-standard python libraries:

- numpy
- matplotlib
- scipy
- pillow

pyebsd is not available yet at PyPI. In order to install the library, first download the repository:

```bash
git clone https://github.com/arthursn/pyebsd
```

Then install pyebsd by running the `setup.py` file:

```bash
python setup.py install
```

Please be aware that administrator permissions might be necessary.

Use the `--user` option to install pyebsd in the user folder:

```bash
python setup.py install --user
```

Please notice that `setuptools` must be installed beforehand.

When pyebsd is installed using `setup.py`, all dependencies should be automatically solved.

If the dependencies are not solved, the required libraries can be installed from the [Python Package Index](https://pypi.org) using pip:

```bash
pip install numpy matplotlib scipy pillow
```
# Basic usage

Load EBSD data:

```python
import matplotlib.pyplot as plt
import pyebsd

# So far, pyebsd only supports loading .ang files generated
# by the TSL OIM software
scan = pyebsd.load_scandata('path/to/ang/file')
plt.show()
```

Plot Inverse Pole Figure (IPF) map:

```python
# gray=scan.IQ is used to set the quality index as grayscale
scan.plot_IPF(gray=scan.IQ)
plt.show()
```

Plot phase map:

```python
scan.plot_phase(gray=scan.IQ)
plt.show()
```

Plot pole figure:

```python
scan.plot_PF()
plt.show()
```