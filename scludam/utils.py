import numpy as np
import itertools
from attr import attrs
from typing import List
from ordered_set import OrderedSet
from typing import Optional, Tuple, List, Union, Callable, Type, Optional

def one_hot_encode(labels: np.ndarray):
    # labels must be np array.
    # Dinstinct labels must be able to be aranged into a list of consecutive int numbers
    # e.g. [-1, 0, 1, 2] is ok, [-1, 1, 3] is not ok
    # labels min could be 0 or -1 if noise is present
    labels = np.asarray(labels).astype(int)
    labels = labels + labels.min() * -1
    one_hot = np.zeros((labels.shape[0], labels.max()+1))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

@attrs(auto_attribs=True, init=False)
class Colnames:
    names: OrderedSet

    def __init__(self, names: List[str]):
        self.names = OrderedSet(names)

    def exclude(self, names: Union[list, str]):
        names = names.parse_to_list()
        return list(self.names - OrderedSet(names))

    def get_data_names(self, names: Union[list, str]=None):
        data = [name for name in list(self.names) if not name.endswith('_error') and not name.endswith('_corr')]
        if names is None:
            return data
        names = self.parse_to_list(names)
        return list(OrderedSet(names).intersection(data))

    def get_error_names(self, names: Union[list, str]=None):
        errors = [name for name in list(self.names) if name.endswith('_error')]
        if names is None:
            names = list(self.names)
        names = self.get_data_names(self.parse_to_list(names))
        sorted_errors = []
        for name in names:
            for err in errors:
                if err.startswith(name):
                    sorted_errors.append(err)
                    errors.remove(err)
                    break

        missing_errors = len(names) != len(sorted_errors)

        return sorted_errors, missing_errors

    def get_corr_names(self, names: Union[list, str]=None):
        correlations = [name for name in list(self.names) if name.endswith('_corr')]
        if names is None:
            names = list(self.names)
        names = self.get_data_names(self.parse_to_list(names))
        
        names_with_corr = []
        for name in names:
            for corr in correlations:
                if name in corr:
                    names_with_corr.append(name)
                    break
        names_with_corr = list(OrderedSet(names_with_corr))

        len_nwc = len(names_with_corr)
        if len_nwc == 0:
            return [], True

        corr_matrix = np.ndarray(shape=(len_nwc, len_nwc), dtype=f'|S{max([len(name) for name in names_with_corr + correlations])}')
        for i1, var1 in enumerate(names_with_corr):
            for i2, var2 in enumerate(names_with_corr):
                corr1 = f'{var1}_{var2}_corr'
                corr2 = f'{var2}_{var1}_corr'
                corr = corr1 if corr1 in correlations else corr2 if corr2 in correlations else ''
                corr_matrix[i1, i2] = corr
        
        sorted_correlations = list(corr_matrix[np.tril_indices(len(names_with_corr), k=-1)].astype(str))
        missing_correlations = len(names_with_corr) != len(names) or any(name == '' for name in sorted_correlations)
        sorted_correlations = [sc for sc in sorted_correlations if sc != '']
        return sorted_correlations, missing_correlations 

    def parse_to_list(self, names: Union[list, str]):
        if isinstance(names, str):
            names = [names]
        return names


def sorted_corr(variables: list, correlations: list):
    vc = variables
    vc_count = len(variables)
    corr_matrix = np.ndarray(shape=(vc_count, vc_count), dtype=f'|S{max([len(c) for c in variables + correlations])}')
    for i1, var1 in enumerate(vc):
        for i2, var2 in enumerate(vc):
            corr1 = f'{var1}_{var2}_corr'
            corr2 = f'{var2}_{var1}_corr'
            corr = corr1 if corr1 in correlations else corr2 if corr2 in correlations else ''
            corr_matrix[i1, i2] = corr
    return list(corr_matrix[np.tril_indices(vc_count, k=-1)].astype(str))

def sorted_err(variables: list, errors: list):
    ordered_errors = []
    for var in variables:
        for err in errors:
            if err.startswith(var):
                ordered_errors.append(err)
                errors.remove(err)
                break
    return ordered_errors

def subset(data: np.ndarray, limits: list):
    for i in range(len(limits)):
        data = data[(data[:,i] > limits[i][0]) & (data[:,i] < limits[i][1])]
    return data

def combinations(items: list):
    return list(itertools.product(*items))

def dict_combinations(items: list):
    "items is a list of dicts"
    return combinations([[{ k: v } for (k, v) in d.items()] for d in items])

def get_colnames(colnames: list):
    error_c = [c for c in colnames if c.endswith('_error')]

    return [f'{var_name}_error' for var_name in var_colnames]

def indices(arrays: Union[tuple, np.ndarray]):
    
    if isinstance(arrays, np.ndarray):
        if len(arrays.shape) == 1:
            return tuple(arrays)
        else:
            return tuple(map(tuple, tuple(arrays)))

    
    arrays = list(arrays)
    shape = None
    for i, arg in enumerate(arrays):
        if isinstance(arg, np.ndarray):
            if len(arg.shape) != 1:
                raise ValueError('arrays must be of 1 dimension')
            if shape is not None and shape != arg.shape:
                raise ValueError('arrays must have same shape')
            shape = arg.shape
            arrays[i] = arg.astype(int)
    
    for i, arg in enumerate(arrays):
        if isinstance(arg, int):
            arrays[i] = np.ones(shape, dtype=int) * arg
        
    return tuple(map(tuple, np.vstack(tuple(arrays))))

""" def args2r(*args, **kwargs):
    # only works for simple 
    return ", ".join(f"{key}={value}" for key, value in kwargs.items()) """
""" 
c = Colnames([
                'ra', 'dec', 'ra_error', 'dec_error', 'ra_dec_corr',
                'pmra', 'pmra_error', 'ra_pmra_corr', 'dec_pmra_corr',
                'pmdec', 'pmdec_error', 'ra_pmdec_corr', 'dec_pmdec_corr', 'pmra_pmdec_corr',
                'parallax', 'parallax_error', 'parallax_pmra_corr', 'parallax_pmdec_corr', 'ra_parallax_corr', 'dec_parallax_corr',
                'phot_g_mean_mag'
            ])

print('coso') """

""" thing = args2str("algo", 12, [3,4,5], False, coso="'algomas'", otrocoso=[1,2,3], tercercoso=np.array([1,2,3]))
print(thing) """