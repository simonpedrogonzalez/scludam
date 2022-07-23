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

"""Module for useful Type Annotations and runtime checking functions."""

from numbers import Number
from typing import List, Tuple, Union

import numpy as np
from beartype import beartype
from beartype.vale import Is, IsAttr, IsEqual
from numpy.typing import NDArray
from typing_extensions import Annotated


def _type(type_hint):
    @beartype
    def _validate(instance, attribute, value: type_hint):
        ...

    return _validate


Coord = Tuple[Number, Number]
Condition = Tuple[str, str, Union[str, Number]]
LogicalExpression = Tuple[str, str, str, Union[str, Number]]

Vector2Numpy = Annotated[NDArray[np.number], Is[lambda x: x.shape == (2,)]]
Vector3Numpy = Annotated[NDArray[np.number], Is[lambda x: x.shape == (3,)]]
Vector2 = Union[Tuple[Number, Number], Annotated[List[Number], 2], Vector2Numpy]
Vector3 = Union[Tuple[Number, Number, Number], Annotated[List[Number], 3], Vector3Numpy]

Vector2Array = Annotated[
    NDArray[np.number], IsAttr["ndim", IsEqual[2]], Is[lambda x: x.shape[1] == 2]
]
Vector3Array = Annotated[
    NDArray[np.number], IsAttr["ndim", IsEqual[2]], Is[lambda x: x.shape[1] == 3]
]

ArrayLike = Union[NDArray, List, Tuple]
OptionalArrayLike = Union[None, ArrayLike]
NumericArray = NDArray[np.number]
Numeric2DArray = Annotated[NDArray[np.number], IsAttr["ndim", IsEqual[2]]]
Numeric1DArray = Annotated[NDArray[np.number], IsAttr["ndim", IsEqual[1]]]

Numeric1DArrayLike = Union[Numeric1DArray, List[Number], Tuple[Number, ...]]
Numeric2DArrayLike = Union[
    Numeric2DArray, List[Numeric1DArrayLike], Tuple[Numeric1DArrayLike]
]

OptionalNumeric1DArrayLike = Union[Numeric1DArrayLike, None]
OptionalNumeric2DArrayLike = Union[Numeric2DArrayLike, None]
OptionalNumericArray = Union[NumericArray, None]
OptionalNumeric2DArray = Union[Numeric2DArray, None]
Int1DArray = Annotated[NDArray[int], IsAttr["ndim", IsEqual[1]]]
