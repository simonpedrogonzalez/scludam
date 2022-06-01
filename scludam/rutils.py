import numpy as np
from rpy2.robjects import packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from typing import Optional, Tuple, List, Union, Callable
from rpy2.robjects import default_converter, conversion

# from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import conversion
import rpy2.rinterface_lib.callbacks as rcallbacks
import warnings
from rpy2.rinterface import RRuntimeWarning

utils = None

def disable_r_warnings():
    warnings.filterwarnings("ignore", category=RRuntimeWarning)

def disable_r_console_output():
    # flush any R console output
    def add_to_stdout(line):
        pass
    def add_to_stderr(line):
        pass

    rcallbacks.consolewrite_print = add_to_stdout
    rcallbacks.consolewrite_warnerror = add_to_stderr

def clean_r_session(rsession, mode: str = "all"):
    if mode == "all":
        rsession("rm(list=ls())")
    elif mode == "var":
        rsession("rm(list=setdiff(ls(), lsf.str()))")
    elif mode == "fun":
        rsession("rm(list=lsf.str())")
    return rsession

def load_r_packages(rsession, packages: Union[list, str]):
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
    params = ""
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, (int, float, str, bool)):
                rsession.assign(key, value)
            elif isinstance(value, np.ndarray):
                with localconverter(default_converter + numpy2ri.converter):
                    rsession.assign(key, value)
            else:
                raise TypeError("Unsuported py to r variable conversion.")
            params += f"{key}={key},"
    return rsession, params[:-1]
