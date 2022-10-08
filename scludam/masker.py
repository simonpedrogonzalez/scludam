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

"""Module for helper masking functions."""

from abc import abstractmethod
from typing import Union

import numpy as np
from attrs import define

# from scipy.spatial import ConvexHull
# from sklearn.metrics import pairwise_distances

# from scludam.synthetic import is_inside_circle, is_inside_sphere


class DataMasker:
    """Abstract class for data masking."""

    @abstractmethod
    def mask(self, data) -> np.ndarray:
        """Mask data.

        Parameters
        ----------
        data : np.ndarray
            Data to mask.

        Returns
        -------
        np.ndarray
            Boolean array.

        """
        pass


@define
class RangeMasker(DataMasker):
    """Mask data outsude a hypercube according to limits.

    Attributes
    ----------
    limits : Union[np.ndarray, list]
        Limits of the hypercube as
        ``[[dim1_min, dim1_max], [dim2_min, dim2_max], ...]``.

    """

    limits: Union[list, np.ndarray]

    def mask(self, data: np.ndarray):
        """Mask the data.

        Parameters
        ----------
        data : np.ndarray
            data to be masked.

        Returns
        -------
        np.ndarray
            Mask as 1D boolean array

        Raises
        ------
        ValueError
            If limits do not have the correct shape.

        """
        # mask data outside a hypercube according to limits
        # data and limits must be in order
        obs, dims = data.shape
        limits = np.array(self.limits)
        ldims, lrange = limits.shape
        if lrange != 2:
            raise ValueError("limits must be of shape (d, 2)")

        mask = np.ones(obs, dtype=bool)

        for i in range(ldims):
            if i >= dims:
                break
            mask[(data[:, i] < limits[i][0]) | (data[:, i] > limits[i][1])] = False
        return mask


# @define
# class CenterMasker(DataMasker):
#     """Mask from data center and radius.

#     Only works for 2d or 3d.

#     Attributes
#     ----------
#     center: Union[np.ndarray, list]
#         Center of the mask.
#     radius: float
#         Radius of the mask.

#     """

#     center: Union[list, np.ndarray]
#     radius: Union[int, float]

#     def mask(self, data: np.ndarray):
#         """Mask the data.

#         Parameters
#         ----------
#         data : np.ndarray
#             data to be masked.

#         Returns
#         -------
#         np.ndarray
#             Mask as 1D boolean array

#         Raises
#         ------
#         ValueError
#             If center has incorrect shape.

#         """
#         # Crop data in a circle or sphere according to limits
#         # takes into account first 2 or 3 dims
#         obs, dims = data.shape
#         center = np.array(self.center)
#         radius = self.radius
#         cdims = center.shape[0]
#         if len(center.shape) > 1 or cdims not in [2, 3] or cdims > dims:
#             raise ValueError(
#               "Center must be shape (2,) or (3,) and <= data dimensions"
#             )

#         obs, dims = data.shape

#         if cdims == 2:
#             return is_inside_circle(center, radius, data[:, 0:2])
#         else:
#             return is_inside_sphere(center, radius, data[:, 0:3])


# @define
# class DistanceMasker(DataMasker):
#     """Mask data according to distance from center.

#     Get a percentage of the observations closest or
#     furthest from  and to the center.

#     Attributes
#     ----------
#     center : Union[np.ndarray, list, str]
#         Center, if str, it must be "geometric", by default "geometric".
#         Geometric center is the center of the data given its ranges.
#     percentage : float
#         Percentage of observations to take, by default 10.
#     metric : str
#         Metric to use for distance calculation, by default "euclidean".
#     mode : str
#         Mode of the mask, by default "closest". Can be one of
#         "closest" or "furthest".

#     """

#     center: Union[list, np.ndarray, str] = "geometric"
#     percentage: Union[int, float] = 10
#     metric: str = "euclidean"
#     mode: str = "closest"

#     def mask(self, data: np.ndarray):
#         """Mask the data.

#         Parameters
#         ----------
#         data : np.ndarray
#             Data to mask

#         Returns
#         -------
#         np.ndarray
#             Mask as 1D boolean array


#         Raises
#         ------
#         NotImplementedError
#             center kind not implemented.
#         ValueError
#             Mode is invalid

#         """
#         if isinstance(self.center, str):
#             if self.center == "geometric":
#                 center = data.min(axis=0) + (data.max(axis=0) - data.min(axis=0)) / 2
#             else:
#                 raise NotImplementedError()
#         else:
#             center = np.array(self.center)
#         distances = pairwise_distances(
#             data, center.reshape(1, -1), metric=self.metric
#         ).ravel()
#         n_obs = int(np.round(self.percentage / 100 * data.shape[0]))
#         idcs = np.argpartition(distances, -n_obs)[-n_obs:]
#         mask = np.zeros_like(distances).astype("bool")
#         mask[idcs] = True
#         if self.mode == "closest":
#             return ~mask
#         elif self.mode == "furthest":
#             return mask
#         else:
#             raise ValueError("Invalid mode")


# @define
# class CrustMasker(DataMasker):
#     """Mask data according to crust.

#     The crust is calculated as the convex hull of the data.

#     Attributes
#     ----------
#     percentage : float
#         Percentage of observations to take, by default 10.
#     mode : str
#         Mode of the calculation, by default "crust".
#         Can only be "crust".

#     """

#     percentage: Union[int, float] = 10
#     mode: str = "crust"

#     def mask(self, data: np.ndarray):
#         """Mask the data.

#         Parameters
#         ----------
#         data : np.ndarray
#             Data to mask
#         Returns
#         -------
#         np.ndarray
#             Mask as a 1D boolean array

#         """
#         n = data.shape[0]
#         n_obs = int(np.round(self.percentage / 100 * n))
#         ch = ConvexHull(data)
#         mask = np.zeros(n).astype(bool)
#         mask[ch.vertices] = True
#         idcs = np.where(~mask)[0]

#         while mask.sum() < n_obs:
#             data_iter = data[~mask]
#             ch = ConvexHull(data_iter)
#             submask = np.zeros(data_iter.shape[0]).astype(bool)
#             submask[ch.vertices] = True
#             mask[idcs[submask]] = True
#             idcs = np.where(~mask)[0]

#         return mask
