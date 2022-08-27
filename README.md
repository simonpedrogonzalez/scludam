# SCLUDAM (**S**tar **CLU**ster **D**etection **A**nd **M**embership estimation)

[![Build Status](https://travis-ci.com/simonpedrogonzalez/scludam.svg?branch=main)](https://travis-ci.com/simonpedrogonzalez/scludam)
[![Documentation Status](https://readthedocs.org/projects/scludam/badge/?version=latest)](https://simonpedrogonzalez.github.io/scludam-docs/index.html)
[![PyPI](https://img.shields.io/pypi/v/scludam)](https://pypi.org/project/scludam/)
[![Python 3.7.6+](https://img.shields.io/badge/python-3.7.6+-blue.svg)](https://github.com/simonpedrogonzalez/scludam)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://github.com/simonpedrogonzalez/scludam)
[![License](https://img.shields.io/badge/License-GNU-blue.svg)](https://tldrlegal.com/license/gnu-lesser-general-public-license-v3-(lgpl-3))


**scludam** (**S**tar **CLU**ster **D**etection **A**nd **M**embership estimation) is a Python package for GAIA catalogues **data fetching**, **star cluster detection** and **star cluster membership estimation**.

### Repository and issues
[https://github.com/simonpedrogonzalez/scludam](https://github.com/simonpedrogonzalez/scludam)

### Authors
- Simón Pedro González
email: [simon.pedro.g@gmail.com](simon.pedro.g@gmail.com)

### Features
Currently **scludam** is a work in progress. Modules and features already included are:

- **fetcher**: simple query builder to get data from the GAIA catalogue more easily, and some extra useful functions.

- **stat_tests**: set of 3 clusterability tests that can be used to detect the presence of a cluster in a sample.

- **synthetic**: classes that can be used to generate synthetic astrometric samples by specifying the distributions to use and parameter values.

- **detection**: classes that can be used to detect the presence of a cluster in a sample.

- **shdbscan**: soft clustering based on the **HDBSCAN** algorithm.

- **hkde**: kernel density estimation with variable bandwidth.

- **membership**: membership calculation based on **hkde** smoothing.

- **pipeline**: pipeline for the detection and membership estimation, with default values and convenience functions.

--------------------------------------------------------------------------------

### Requirements
You need **Python 3.7.6+** and **R 3.6.3+** to run scludam. It is recommended to install scludam in a separate environment created with pyenv or conda, to avoid dependencies issues with other preinstalled packages you may have in the base environment. The following dependencies will be installed with SCLUDAM:

- numpy>=1.21.6
- matplotlib>=3.4.1
- scipy>=1.7.3
- astropy>=4.3.1
- astroquery>=0.4.6
- pandas>=1.3.5
- hdbscan==0.8.27
- scikit-learn>=1.0.2
- scikit-image>=0.18.1
- rpy2>=3.5.2
- seaborn>=0.11.0
- attrs>=21.4.0
- beartype>=0.10.0
- ordered_set>=4.0.2
- statsmodels>=0.12.2
- diptest>=0.4.2
- typing_extensions>=4.2.0

### User install in a Conda environment
Create a conda environment named ``myscludamenv`` with python3.8 and ``scludam`` installed
```
conda create --name myscludamenv python=3.8 pip --yes
conda activate myscludamenv
python -m pip install scludam
```

### User install
Install from PyPi:
```python -m pip install scludam```

### Dev install
Clone the repo and run the following command in the cloned directory (with your environment activated):
```python -m pip install -e .[dev]```