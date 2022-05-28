import os

from setuptools import setup


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="scludam",
    version="0.0.1",
    author="Simón Pedro González",
    author_email="simon.pedro.g@gmail.com",
    description="star cluster detection and membership estimation",
    license="GPL-3",
    keywords="star cluster detection membership probabilities",
    url="http://packages.python.org/scludam",
    packages=["scludam", "tests"],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    # dependencies
    install_requires=[
        "numpy>=1.19.3",
        "matplotlib>=3.4.1",
        "scipy>=1.4.1",
        "astropy>=4.2.1",
        "astroquery>=0.4.1",
        "pandas>=1.1.4",
        "hdbscan==0.8.27",
        "scikit-learn>=0.23.1",
        "scikit-image>=0.18.1",
        # "rpy2>=3.1.0",
        "seaborn>=0.11.0",
        "attrs>=21.4.0",
        "beartype>=0.10.0",
        # "statmodels==0.12.2",
        # "unidip==0.1.1"
    ],
    extras_require={
        # dev dependencies
        "dev": [
            "pytest",
            "pytest-pep8",
            "pytest-cov",
            "pytest-mock",
            "flake8",
            "black>=22.3.0",
            "isort",
            "flake8-black",
            "flake8-import-order",
        ]
    },
)
