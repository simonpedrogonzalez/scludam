
SCLUDAM (\ **S**\ tar **CLU**\ ster **D**\ etection **A**\ nd **M**\ embership estimation)
==================================================================================================


.. image:: https://travis-ci.com/simonpedrogonzalez/scludam.svg?branch=main
   :target: https://travis-ci.com/simonpedrogonzalez/scludam
   :alt: Build Status


.. image:: https://img.shields.io/badge/docs-passing-success
   :target: https://simonpedrogonzalez.github.io/scludam-docs/index.html
   :alt: Documentation Status


.. image:: https://img.shields.io/pypi/v/scludam
   :target: https://pypi.org/project/scludam/
   :alt: PyPI


.. image:: https://img.shields.io/badge/python-3.9.21+-blue.svg
   :target: https://github.com/simonpedrogonzalez/scludam
   :alt: Python 3.9.21+


.. image:: https://img.shields.io/badge/python-3.10.16+-blue.svg
   :target: https://github.com/simonpedrogonzalez/scludam
   :alt: Python  3.10.16+


.. image:: https://img.shields.io/badge/python-3.11.11+-blue.svg
   :target: https://github.com/simonpedrogonzalez/scludam
   :alt: Python  3.11.11+


.. image:: https://img.shields.io/badge/License-GNU-blue.svg
   :target: https://tldrlegal.com/license/gnu-lesser-general-public-license-v3-(lgpl-3)
   :alt: License


**SCLUDAM** (\ **S**\ tar **CLU**\ ster **D**\ etection **A**\ nd **M**\ embership estimation) is a Python package for GAIA catalogues **data fetching**\ , **star cluster detection** and **star cluster membership estimation**.

Repository and issues
^^^^^^^^^^^^^^^^^^^^^

`https://github.com/simonpedrogonzalez/scludam <https://github.com/simonpedrogonzalez/scludam>`_

Authors
^^^^^^^


* Simón Pedro González. 
  Email: `simon.pedro.g@gmail.com <simon.pedro.g@gmail.com>`_

Features
^^^^^^^^

Included modules and features are:


* 
  **fetcher**\ : Query builder for easy access to GAIA catalogues data, functions to get catalogues and SIMBAD objects information.

* 
  **stat_tests**\ : Set of three clusterability tests that can be used to detect the presence of a cluster in a sample.

* 
  **synthetic**\ : Classes that can be used to generate synthetic astrometric samples by specifying the distributions and parameter values.

* 
  **detection**\ : Detection of star clusters in a sample using an improved version of the Star Counts algorithm.

* 
  **shdbscan**\ : Soft clustering based on the **HDBSCAN** algorithm.

* 
  **hkde**\ : Kernel density estimation with per-observation or per-dimension variable bandwidth.

* 
  **membership**\ : Membership probability estimation based on **hkde** smoothing.

* 
  **pipeline**\ : Pipeline for the detection and membership estimation, with default values and convenience functions.

* 
  **plots**\ : Plot detection and membership estimation results alongside SIMBAD objects for better result interpretation.

----

Requirements
^^^^^^^^^^^^

**Python 3.9+**  is needed to run SCLUDAM. It is recommended to install scludam in a separate environment created to avoid dependencies issues with other preinstalled packages in the base environment. The following dependencies will be installed along with SCLUDAM:


* numpy>=1.26.4,<2.0
* matplotlib>=3.9.4,<4.0
* scipy>=1.13.1,<2.0
* astropy>=6.0.1,<7.0
* astroquery==0.4.6
* pandas>=2.3.1,< 3.0
* hdbscan>=0.8.40
* joblib>=1.1.0
* scikit-learn>=1.1.3
* scikit-image>=0.24.0
* seaborn>=0.13.2,<0.14
* statsmodels>=0.12.2
* diptest>=0.10.0,<0.11.0

User install in a Conda environment (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a conda environment named ``myscludamenv`` with python3.11 and ``scludam`` installed

.. code-block::

   conda create --name myscludamenv python=3.11 pip --yes
   conda activate myscludamenv
   python -m pip install scludam

Update scludam in a Conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

   conda activate myscludamenv
   python -m pip install -U scludam
   python -m pip show scludam

Simple user install
^^^^^^^^^^^^^^^^^^^

Install from PyPi:
``python -m pip install scludam``

Simple user update
^^^^^^^^^^^^^^^^^^

Update from PyPi:
``python -m pip install -U scludam``

Dev install
^^^^^^^^^^^

Clone the repo and run the following command in the cloned directory (with your environment activated):
``python -m pip install -e .[dev]``
