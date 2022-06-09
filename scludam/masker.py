import os
import sys
from abc import abstractmethod
from typing import Union

from attr import attrs

import numpy as np

from sklearn.metrics import pairwise_distances
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
import seaborn as sns


sys.path.append(os.path.join(os.path.dirname("scludam"), "."))
from scludam.synthetic import is_inside_circle, is_inside_sphere


class DataMasker:
    @abstractmethod
    def mask(self, data) -> np.ndarray:
        pass


@attrs(auto_attribs=True)
class RangeMasker(DataMasker):
    limits: Union[list, np.ndarray]

    def mask(self, data: np.ndarray):
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


@attrs(auto_attribs=True)
class CenterMasker(DataMasker):
    center: Union[list, np.ndarray]
    radius: Union[int, float]

    def mask(self, data: np.ndarray):
        # Crop data in a circle or sphere according to limits
        # takes into account first 2 or 3 dims
        obs, dims = data.shape
        center = np.array(self.center)
        radius = self.radius
        cdims = center.shape[0]
        if len(center.shape) > 1 or cdims not in [2, 3] or cdims > dims:
            raise ValueError("Center must be shape (2,) or (3,) and <= data dimensions")

        obs, dims = data.shape

        if cdims == 2:
            return is_inside_circle(center, radius, data[:, 0:2])
        else:
            return is_inside_sphere(center, radius, data[:, 0:3])


@attrs(auto_attribs=True)
class DistanceMasker(DataMasker):
    center: Union[list, np.ndarray, str] = "geometric"
    percentage: Union[int, float] = 10
    metric: str = "euclidean"
    mode: str = "furthest"

    def mask(self, data: np.ndarray):
        if isinstance(self.center, str):
            if self.center == "geometric":
                center = data.min(axis=0) + (data.max(axis=0) - data.min(axis=0)) / 2
            else:
                raise NotImplementedError()
        distances = pairwise_distances(
            data, center.reshape(1, -1), metric=self.metric
        ).ravel()
        n_obs = int(np.round(self.percentage / 100 * data.shape[0]))
        idcs = np.argpartition(distances, -n_obs)[-n_obs:]
        mask = np.zeros_like(distances).astype("bool")
        mask[idcs] = True
        if self.mode == "closest":
            return ~mask
        elif self.mode == "furthest":
            return mask
        else:
            raise ValueError("Invalid mode")


@attrs(auto_attribs=True)
class CrustMasker(DataMasker):
    percentage: Union[int, float] = 10
    mode: str = "crust"

    def mask(self, data: np.ndarray):
        n = data.shape[0]
        n_obs = int(np.round(self.percentage / 100 * n))
        ch = ConvexHull(data)
        mask = np.zeros(n).astype(bool)
        mask[ch.vertices] = True
        idcs = np.where(~mask)[0]

        while mask.sum() < n_obs:
            data_iter = data[~mask]
            ch = ConvexHull(data_iter)
            submask = np.zeros(data_iter.shape[0]).astype(bool)
            submask[ch.vertices] = True
            mask[idcs[submask]] = True
            idcs = np.where(~mask)[0]

        return mask
