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

"""Module for helper functions."""
from typing import List, Union

import numpy as np
from attrs import define
from ordered_set import OrderedSet

from scludam.type_utils import Numeric1DArray


def one_hot_encode(labels: Numeric1DArray):
    """One-hot encode a list of labels.

    Distinct labels must form a contiguous range, e.g. [0, 1, 2, 3].

    Parameters
    ----------
    labels : Numeric1DArray
        Labels.

    Returns
    -------
    Numeric2DArray
        One-hot code.

    """
    labels = np.asarray(labels).astype(int)
    labels = labels + labels.min() * -1
    one_hot = np.zeros((labels.shape[0], labels.max() + 1))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


@define
class Colnames:
    """Class for column names.

    Stores column names as an ordered set and allows some operations on them.

    """

    names: OrderedSet

    def __init__(self, names: List[str]):
        self.names = OrderedSet(names)

    def exclude(self, names: Union[list, str]):
        """Exclude names from the set.

        Parameters
        ----------
        names : Union[list, str]
            Names to exclude from the original set.

        Returns
        -------
        List[str]
            Column names after exclusion.

        """
        names = names._parse_to_list()
        return list(self.names - OrderedSet(names))

    def data(self, names: Union[list, str] = None):
        """Get names of data columns.

        A column is considered data if it does not
        end with "_error" or "_corr".

        Parameters
        ----------
        names : Union[list, str], optional
            List of names to filter, by default None.
            If None, all names are used.

        Returns
        -------
        List[str]
            Column names.

        """
        data = [
            name
            for name in list(self.names)
            if not name.endswith("_error") and not name.endswith("_corr")
        ]
        if names is None:
            return data
        names = self._parse_to_list(names)
        return list(OrderedSet(names).intersection(data))

    def error(self, names: Union[list, str] = None):
        """Get names of error columns.

        A column is considered error if it ends with "_error".

        Parameters
        ----------
        names : Union[list, str], optional
            List of data column names to filter, by default None.
            If None, the function returns error columns
            found. If not, the function returns error columns
            of the names in the list.

        Returns
        -------
        List[str]
            Error column names.

        """
        errors = [name for name in list(self.names) if name.endswith("_error")]
        if names is None:
            names = list(self.names)
        names = self.data(self._parse_to_list(names))
        sorted_errors = []
        for name in names:
            for err in errors:
                if err.startswith(name):
                    sorted_errors.append(err)
                    errors.remove(err)
                    break

        return sorted_errors

    def missing_error(self, names: Union[list, str] = None):
        """Check if there are missing error columns.

        Parameters
        ----------
        names : Union[list, str], optional
            List of data column names to filter, by default None.
            If None, the function will check within all data columns
            . If not, the function returns error columns
            of the names in the list.

        Returns
        -------
        bool
            True if there are missing error columns, False otherwise.

        """
        if names is None:
            names = list(self.names)
        names = self.data(self._parse_to_list(names))
        errors = self.error(names)
        missing_errors = len(names) != len(errors)
        return missing_errors

    def _corr(self, names: Union[list, str] = None):
        correlations = [name for name in list(self.names) if name.endswith("_corr")]
        if names is None:
            names = list(self.names)
        names = self.data(self._parse_to_list(names))

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

        corr_matrix = np.ndarray(
            shape=(len_nwc, len_nwc),
            dtype=f"|S{max([len(name) for name in names_with_corr + correlations])}",
        )
        for i1, var1 in enumerate(names_with_corr):
            for i2, var2 in enumerate(names_with_corr):
                corr1 = f"{var1}_{var2}_corr"
                corr2 = f"{var2}_{var1}_corr"
                corr = (
                    corr1
                    if corr1 in correlations
                    else corr2
                    if corr2 in correlations
                    else ""
                )
                corr_matrix[i1, i2] = corr

        sorted_correlations = list(
            corr_matrix[np.tril_indices(len(names_with_corr), k=-1)].astype(str)
        )
        missing_correlations = len(names_with_corr) != len(names) or any(
            name == "" for name in sorted_correlations
        )
        sorted_correlations = [sc for sc in sorted_correlations if sc != ""]
        return sorted_correlations, missing_correlations

    def corr(self, names: Union[list, str] = None):
        """Get names of correlation columns.

        A column is considered correlation if it ends with "_corr".

        Parameters
        ----------
        names : Union[list, str], optional
            List of data column names to filter, by default None.
            If None, the function returns correlation columns
            found. If not, the function returns correlation columns
            related to the data column names in the list.

        Returns
        -------
        List[str]
            Correlation column names.

        """
        correlations, _ = self._corr(names)
        return correlations

    def missing_corr(self, names: Union[list, str] = None):
        """Check if there are missing correlation columns.

        Parameters
        ----------
        names : Union[list, str], optional
            List of data column names to filter, by default None.
            If None, the function will check within all data columns
            . If not, the function wil check within the columns
            of the names in the list.

        Returns
        -------
        bool
            True if there are missing correlation columns, False otherwise.

        """
        if names is None:
            names = list(self.names)
        names = self.data(self._parse_to_list(names))
        _, missing_correlations = self._corr(names)
        return missing_correlations

    def _parse_to_list(self, names: Union[list, str]):
        if isinstance(names, str):
            names = [names]
        return names
