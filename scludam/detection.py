import os
import sys
from abc import abstractmethod
from typing import Optional, Union, List
from warnings import warn

import numpy as np
from astropy.stats.sigma_clipping import sigma_clipped_stats
from attrs import define, field, Factory, validators
from scipy import ndimage
from skimage.feature import peak_local_max
import seaborn as sns
from matplotlib import pyplot as plt
import math

from scludam.masker import RangeMasker

# from scludam.synthetic import three_clusters_sample
from decimal import Decimal
from scludam.synthetic import (
    polar_to_cartesian,
    UniformSphere,
    StarCluster,
    StarField,
    Synthetic,
    UniformFrustum,
)
from scipy.stats import multivariate_normal

from numbers import Number


def get_default_mask(dim: int):
    indexes = np.array(np.meshgrid(*np.tile(np.arange(5), (dim, 1)))).T.reshape(
        (-1, dim)
    )
    mask = np.zeros([5] * dim)
    cond = np.sum((indexes - 2) ** 2, axis=1)
    mask[tuple(indexes[np.argwhere((cond > 0) & (cond < 5))].reshape((-1, dim)).T)] = 1
    return mask / np.count_nonzero(mask)


def convolve(data, mask: np.ndarray = None, c_filter: callable = None, *args, **kwargs):
    if c_filter:
        return c_filter(data, *args, **kwargs)
    if mask is not None:
        return ndimage.convolve(data, mask, *args, **kwargs)


# unused
# def var_filter(data, mask=None, *args, **kwargs):
#     if mask is not None:
#         kwargs["footprint"] = mask != 0
#     return convolve(
#         data,
#         c_filter=ndimage.generic_filter,
#         function=np.var,
#         *args,
#         **kwargs,
#     )


# unused
# def std_filter(data, mask=None, *args, **kwargs):
#     if mask is not None:
#         kwargs["footprint"] = mask != 0
#     return convolve(
#         data,
#         c_filter=ndimage.generic_filter,
#         function=np.std,
#         *args,
#         **kwargs,
#     )


def fast_std_filter(data, mask, **kwargs):
    u_x2 = convolve(data, mask=mask, **kwargs)
    ux_2 = convolve(data * data, mask=mask, **kwargs)
    return (ux_2 - u_x2 * u_x2) ** 0.5


def get_histogram_bins(data: np.ndarray, bin_shape: np.ndarray, offsets: list = None):
    _, dim = data.shape
    # calculate the margins which are added to the range in order
    # to fit a number of bins that is integer
    margins = [
        (
            bin_shape[i]
            - float(
                Decimal(float(data[:, i].max() - data[:, i].min()))
                % Decimal(float(bin_shape[i]))
            )
        )
        / 2.0
        for i in range(dim)
    ]
    # add base margins
    ranges = [
        [data[:, i].min() - margins[i], data[:, i].max() + margins[i]]
        for i in range(dim)
    ]
    if offsets is not None:
        ranges = [
            [ranges[i][0] + offsets[i], ranges[i][1] + offsets[i]] for i in range(dim)
        ]
    bins = [round((ranges[i][1] - ranges[i][0]) / bin_shape[i]) for i in range(dim)]
    return np.array(bins), np.array(ranges)


def histogram(
    data: np.ndarray, bin_shape: Union[list, np.ndarray], offsets: list = None
):
    _, dim = data.shape
    bins, ranges = get_histogram_bins(data, bin_shape, offsets)
    hist, edges = np.histogramdd(data, bins=bins, range=ranges, density=False)
    return hist, edges


def nyquist_offsets(bin_shape: list):
    dim = len(bin_shape)
    values = np.vstack((np.array(bin_shape) / 2, np.zeros(dim))).T
    combinations = np.array(np.meshgrid(*values)).T.reshape((-1, dim))
    return np.flip(combinations, axis=0)


@define
class Peak:
    index: np.ndarray
    significance: Union[float, int] = None
    count: Union[float, int] = None
    center: np.ndarray = None
    sigma: np.ndarray = None

    def is_in_neighbourhood(self, b):
        return not np.any(np.abs(self.index - b.index) > 1)


def get_most_significant_peaks(peaks: list):
    if not len(peaks):
        return []
    bests = [peaks[0]]
    peaks = peaks[1:]
    while len(peaks) > 0:
        peak = peaks.pop()
        i = 0
        while i < len(bests):
            if bests[i].is_in_neighbourhood(peak):
                if bests[i].significance < peak.significance:
                    bests[i] = peak
                break
            i += 1
        if i == len(bests):
            bests.append(peak)
    return bests


@define
class DetectionResult:
    peaks: List[Peak] = Factory(list)
    heatmaps = None


class PeakDetector:
    @abstractmethod
    def detect(data):
        pass


@define
class CountPeakDetector(PeakDetector):
    bin_shape: Union[list, np.ndarray] = field()
    mask: Union[list, np.ndarray] = field(default=None)
    nyquist_offset: bool = field(default=True)
    min_count: Number = field(default=5)
    min_dif: Number = field(default=10)
    min_sigma_dif: Number = field(default=None)
    min_significance: Number = field(default=1)
    max_num_peaks: Number = field(default=np.inf)
    min_interpeak_dist: Number = field(default=1)
    remove_low_density_regions: bool = field(default=True)
    norm_mode: str = field(default="std", validator=validators.in_(["std", "approx"]))
    _offsets = None

    def trim_low_density_regions(self, data: np.ndarray):
        obs, dim = data.shape

        # calculate data ranges in each dimension, taking into account
        # that bin number must be integer
        _, ranges = get_histogram_bins(data, self.bin_shape)

        for i in range(dim):
            shifts = [self.bin_shape[i], -self.bin_shape[i]]
            for j in range(2):
                while True:
                    if ranges[i][0] >= ranges[i][1]:
                        raise ValueError(
                            """No bin passed minimum density check.
                            Check min_count parameter."""
                        )
                    slice_ranges = np.copy(ranges)
                    # if j = 0 -> upper limit = lower limit + bin shape
                    # if j = 1 -> lower limit = upper limit - bin shape
                    slice_ranges[i][int(not (j))] = slice_ranges[i][j] + shifts[j]
                    data_slice = data[RangeMasker(limits=slice_ranges).mask(data)]
                    if data_slice.shape[0] != 0:
                        slice_histogram, _ = histogram(data_slice, self.bin_shape)
                        # if any bin has min required count, stop trimming
                        if np.any(slice_histogram >= self.min_count):
                            break
                        # else, move limit towards the center and continue
                    ranges[i][j] = slice_ranges[i][int(not (j))]

        # extend ranges half mask shape in each direction so data that belongs to
        # an invalid bin can contribute in a border valid bin when the mask is applied
        mask_shape = np.array(self.mask.shape)
        half_mask_shape = np.floor(mask_shape / 2)
        ranges[:, 0] = ranges[:, 0] - half_mask_shape * self.bin_shape
        ranges[:, 1] = ranges[:, 1] + half_mask_shape * self.bin_shape

        # trim data and return
        trimmed_data = data[RangeMasker(limits=ranges).mask(data)]
        return trimmed_data

    def set_nyquist_offsets(self):
        dim = len(self.bin_shape)
        if not self.nyquist_offset:
            self._offsets = np.atleast_2d(np.zeros(dim))
        else:
            values = np.vstack((np.array(self.bin_shape) / 2, np.zeros(dim))).T
            combinations = np.array(np.meshgrid(*values)).T.reshape((-1, dim))
            self._offsets = np.flip(combinations, axis=0)

    def detect(self, data: np.ndarray):
        if len(data.shape) != 2:
            raise ValueError("data array must have 2 dimensions")
        obs, dim = data.shape

        # mask setup
        if self.mask is None:
            self.mask = get_default_mask(dim)
        self.mask = np.array(self.mask)
        mask = self.mask

        # check mask and bin shape are compatible
        self.bin_shape = np.array(self.bin_shape)
        if len(mask.shape) != dim:
            raise ValueError("mask does not match data dimensions")
        if len(self.bin_shape) != dim:
            raise ValueError("bin_shape does not match data dimensions")

        # remove points in low density regions
        if self.remove_low_density_regions and self.min_count:
            data = self.trim_low_density_regions(data)

        # set nyquist offsets
        self.set_nyquist_offsets()

        # get histogram ranges and bin numbers
        bins, ranges = get_histogram_bins(data, self.bin_shape)

        # set peak detection parameters
        peak_detection_params = {}
        if np.any(np.array(bins) < np.array(mask.shape)):
            warn(
                f"Histogram has too few bins in some dimensions: bin numbers are {bins}"
            )
            peak_detection_params["exclude_border"] = False
        else:
            peak_detection_params["exclude_border"] = True
        if self.min_interpeak_dist:
            peak_detection_params["min_distance"] = self.min_interpeak_dist
        if self.min_significance:
            peak_detection_params["threshold_abs"] = self.min_significance
        if self.max_num_peaks:
            peak_detection_params["num_peaks"] = self.max_num_peaks

        # detection
        peaks = []
        for offset in self._offsets:
            hist, edges = histogram(data, self.bin_shape, offset)
            smoothed = convolve(hist, mask=mask)
            sharp = hist - smoothed
            
            # TODO: old block
            # std = fast_std_filter(hist, mask=mask)
            # normalized = sharp / np.sqrt(hist + std**2 + 1)
            
            if self.norm_mode == "approx":
                # approx explanation
                # err_hist = sqrt(hist)
                # because:
                # estimate of std: std[vhat] = sqrt(vhat) = sqrt(n)
                # error bars are given by the square root
                # of the number of entries in each bin of the histogram
                # err_smoothed = std(smoothed) = sqrt(smoothed)
                # sharp = hist - smoothed so
                # err_sharp = sqrt(err_hist^2 + err_smoothed^2 - ...)
                # because:
                # std[a - b] = sqrt(varianza(a) + varianza(b) - 2*covarianza(a,b))
                # according to: https://en.wikipedia.org/wiki/Propagation_of_uncertainty
                # covarianza(a,b) = varianza(a) * varianza(b) * corrcoef(a,b)
                # as normalized = sharp / err_sharp
                # +1 is added to avoid zero division errors
                normalized = sharp / np.sqrt(smoothed + hist + 1)
            elif self.norm_mode == "std":
                # directly gettig std
                normalized = sharp / (fast_std_filter(sharp, mask=mask) + 1)

            # TODO: remove, other way of getting the std approx
            # n4 = sharp / np.sqrt(std**2 + fast_std_filter(smoothed, mask=mask)**2 + 1)

            detection_img = np.copy(normalized)

            if self.min_dif is not None:
                # check for other way to implement
                # TODO: fix this beacuse it affects the value of the significance
                # TODO: is it solved?
                detection_img[sharp < self.min_dif] = 0
            if self.min_sigma_dif is not None:
                detection_img[sharp < self.min_sigma_dif * std] = 0

            clusters_idx = peak_local_max(detection_img, **peak_detection_params).T

            _, peak_count = clusters_idx.shape
            if peak_count != 0:

                counts = sharp[tuple(clusters_idx)]
                significance = normalized[tuple(clusters_idx)]
                limits = [
                    [
                        (
                            edges[i][clusters_idx[i][j]] - self.bin_shape[i],
                            edges[i][clusters_idx[i][j]] + self.bin_shape[i],
                        )
                        for i in range(dim)
                    ]
                    for j in range(peak_count)
                ]
                subsets = [
                    data[RangeMasker(limits=limits[j]).mask(data)]
                    for j in range(peak_count)
                ]

                # stats may be useless if other center and sigma are calculated
                # afterwards e.g. meanshift and profile analysis
                statitstics = np.array(
                    [
                        [
                            sigma_clipped_stats(
                                subsets[j][:, i],
                                cenfunc="median",
                                stdfunc="mad_std",
                                maxiters=None,
                                sigma=1,
                            )
                            for i in range(dim)
                        ]
                        for j in range(peak_count)
                    ]
                )

                current_peaks = [
                    Peak(
                        index=clusters_idx[:, i].T,
                        significance=significance[i],
                        count=counts[i],
                        center=statitstics[i, :, 1],
                        sigma=np.array(self.bin_shape),
                    )
                    for i in range(peak_count)
                ]
                peaks += current_peaks

        if len(peaks) == 0:
            return DetectionResult()

        # compare same peaks in different histogram offsets
        # and return most sifnificant peak for all offsets
        global_peaks = get_most_significant_peaks(peaks)
        global_peaks.sort(key=lambda x: x.significance, reverse=True)

        if self.max_num_peaks != np.inf:
            global_peaks = global_peaks[0 : self.max_num_peaks]

        res = DetectionResult(peaks=global_peaks)

        # TODO: remove, calculate heatmap in other function
        # if heatmaps:
            # it will return heatmap corresponding to last offset only
            # res.heatmaps = create_heatmaps(
            #     hist,
            #     edges,
            #     self.bin_shape,
            #     np.array([peak.index for peak in global_peaks]).T,
            # )
        return res

def extend_1dmask(mask, dim):
    m1 = np.asarray(mask)
    mi = m1
    for i in range(dim-1):
        mi = np.multiply.outer(mi, m1)
    return mi / mi.sum()
