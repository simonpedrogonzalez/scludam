# scludam, Star CLUster Detection And Membership estimation package
# Copyright (C) 2022  Simón Pedro González

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Module for R helper functions."""
import warnings
from typing import List, Union

import numpy as np
import rpy2.rinterface_lib.callbacks as rcallbacks
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects import packages as rpackages

# from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

utils = None


def disable_r_warnings():
    """Disable R runtime warnings."""
    warnings.filterwarnings("ignore", category=RRuntimeWarning)


def disable_r_console_output():
    """Disable R console output in python."""

    def add_to_stdout(line):
        pass

    def add_to_stderr(line):
        pass

    rcallbacks.consolewrite_print = add_to_stdout
    rcallbacks.consolewrite_warnerror = add_to_stderr


def clean_r_session(rsession, mode: str = "all"):
    """Remove saved elements from R session.

    Parameters
    ----------
    rsession : rpy2.robjects.R
        Active R session object.
    mode : str, optional
        Elements to remove, by default "all".
        Possible values are "all" to remove
        variables, functions and imported packages,
        "vars" to remove only variables.

    Returns
    -------
    rpy2.robjects.R
        Active R session object, the same provided.

    """
    if mode == "all":
        rsession("rm(list=ls())")
    elif mode == "var":
        rsession("rm(list=setdiff(ls(), lsf.str()))")
    elif mode == "fun":
        rsession("rm(list=lsf.str())")
    return rsession


def load_r_packages(rsession, packages: Union[List[str], str]):
    """Load R packages into R session.

    If the package is not installed, it will try to install
    it from the first CRAN mirror.

    Parameters
    ----------
    rsession : rpy2.robjects.R
        Active R session.
    packages : Union[List[str], str]
        A package name or list of package names

    Returns
    -------
    rpy2.robjects.R
        R session object with loaded packages, the same
        R session provided.

    """
    if isinstance(packages, str):
        packages = [packages]
    for package in packages:
        if not rsession(f'"{package}" %in% (.packages())')[0]:
            if not rpackages.isinstalled(package):
                global utils
                if utils is None:
                    # so its done on demand just the first time
                    utils = rpackages.importr("utils")
                    utils.chooseCRANmirror(ind=1)
                utils.install_packages(package)
            rpackages.importr(package)
    return rsession


def assign_r_args(rsession, **kwargs):
    """Assign R arguments to R session.

    Loads function kwargs as R variables
    into the provided R session. The kwargs
    supported can have any name, but must be
    of type str, int, float, bool or np.ndarray.

    Parameters
    ----------
    rsession : rpy2.robjects.R
        Active R session.

    Returns
    -------
    rpy2.robjects.R
        Provided R session with assigned arguments.
    str
        A string containing all the assigned arguments, useful
        for calling R commands, assuming vars are named as they
        are used in the R command. For example:
        ``_, my_r_args = assing_r_args(rsession, a=1, b="foo")`` returns
        ``my_r_args="a=a, b=b"``. Then the a command can be run as:
        ``r(f"myRfunction({my_r_args})")``.

    Raises
    ------
    TypeError
        If kwarg is not of type str, int, float, bool or np.ndarray.

    """
    params = ""
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, (int, float, str, bool, np.ndarray)):
                with localconverter(default_converter + numpy2ri.converter):
                    rsession.assign(key, value)
            else:
                raise TypeError(
                    f"Unsupported python to r argument conversion for {key} of type"
                    f" {type(value).__name__}"
                )
            params += f"{key}={key},"
    return rsession, params[:-1]
