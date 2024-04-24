import os

from setuptools import setup


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="scludam",
    version="1.0.8",
    author="Simón Pedro González",
    author_email="simon.pedro.g@gmail.com",
    description="Star cluster detection and membership estimation based on GAIA data.",
    license="GPL-3",
    keywords="star cluster detection membership probabilities",
    url="http://packages.python.org/scludam",
    packages=["scludam", "tests"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    # dependencies
    install_requires=[
        "numpy>=1.21.6",  # "numpy>=1.19.3",
        "matplotlib>=3.4.1",
        "scipy>=1.7.3",  # "scipy>=1.4.1",
        "astropy>=4.3.1",  # "astropy>=5.1",  # "astropy>=4.2.1",
        "astroquery>=0.4.6",  # "astroquery>=0.4.7",  # "astroquery>=0.4.1",
        "pandas>=1.3.5",  # "pandas>=1.4.2",  # "pandas>=1.1.4",
        "hdbscan==0.8.28",
        "joblib==1.1.0",  # in order to hdbscan to work
        "scikit-learn>=1.0.2",  # "scikit-learn>=0.23.1",
        "scikit-image>=0.18.1",
        # "rpy2>=3.5.2",  # "rpy2>=3.1.0",
        "seaborn>=0.11.0",
        "attrs>=21.4.0",
        "beartype>=0.10.0",
        "ordered_set>=4.0.2",
        "statsmodels>=0.12.2",
        "diptest>=0.4.2",  # "unidip>=0.1.1",
        "typing_extensions>=4.2.0",
    ],
    extras_require={
        # dev dependencies
        "dev": [
            "pytest",
            "pytest-pep8",
            "pytest-cov",
            "pytest-mock",
            "pytest-mpl",
            "flake8",
            "black>=22.3.0",
            "isort",
            "flake8-black",
            "flake8-import-order",
        ]
    },
)
