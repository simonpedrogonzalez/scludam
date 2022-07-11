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

import numpy as np

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
