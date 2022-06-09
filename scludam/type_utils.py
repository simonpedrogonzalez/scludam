from beartype.vale import IsAttr, IsEqual, Is
from beartype import beartype

from numpy.typing import ArrayLike, NDArray
from typing_extensions import Annotated
from typing import Tuple, Union, List
from numbers import Number
import numpy as np

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

Vector2Array = Annotated[NDArray[np.number], IsAttr["ndim", IsEqual[2]], Is[lambda x: x.shape[1] == 2]]
Vector3Array = Annotated[NDArray[np.number], IsAttr["ndim", IsEqual[2]], Is[lambda x: x.shape[1] == 3]]

Numeric2DArray = Annotated[NDArray[np.number], IsAttr["ndim", IsEqual[2]]]
Numeric1DArray = Annotated[NDArray[np.number], IsAttr["ndim", IsEqual[1]]]
