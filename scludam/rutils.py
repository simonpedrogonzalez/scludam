import numpy as np
from rpy2.robjects import packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from typing import Optional, Tuple, List, Union, Callable


utils = None

def rclean(rsession, mode: str = 'all'):
    if mode == 'all':
        rsession('rm(list=ls())')
    elif mode == 'var':
        rsession('rm(list=setdiff(ls(), lsf.str()))')
    elif mode == 'fun':
        rsession('rm(list=lsf.str())')
    return rsession

def pyargs2r(rsession, **kwargs):
    numpy2ri.activate()
    params = ''
    for key, value in kwargs.items():
        if value is not None:   
            if isinstance(value, (int, float, np.ndarray, str, bool)):
                rsession.assign(key, value)
                params += f'{key}={key},'
            else:
                raise TypeError('Unsuported py to r variable conversion.')
    numpy2ri.deactivate()
    return rsession, params[:-1]

def rhardload(rsession, packages: Union[list, str]):
    if isinstance(packages, str):
        packages = [packages]
    for package in packages:
        if not rsession(f'"{package}" %in% (.packages())')[0]:
            if not rpackages.isinstalled(package):
                global utils
                if utils is None:
                    # so its done on demand just the first time
                    utils = rpackages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                utils.install_packages(package)
            rpackages.importr(package)
    return rsession