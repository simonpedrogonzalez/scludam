import os
import sys
from abc import abstractmethod
from typing import Optional, Union, List
from warnings import warn

import numpy as np
from astropy.stats.sigma_clipping import sigma_clipped_stats
from attr import attrs, attrib
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
)
from scipy.stats import multivariate_normal


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
# TODO: should use mask, e.g.: not include center pixel.
# idea: use generic_filter footprint?
# TODO: remove
def var_filter(data, mask=None, *args, **kwargs):
    if mask is not None:
        kwargs["footprint"] = mask != 0
    return convolve(
        data,
        c_filter=ndimage.generic_filter,
        function=np.var,
        *args,
        **kwargs,
    )


# unused, only keep for testing
# TODO: use mask, should not include center pixel faster
def std_filter(data, mask=None, *args, **kwargs):
    if mask is not None:
        kwargs["footprint"] = mask != 0
    return convolve(
        data,
        c_filter=ndimage.generic_filter,
        function=np.std,
        *args,
        **kwargs,
    )


def fast_std_filter(data, mask, **kwargs):
    u_x2 = convolve(data, mask=mask, **kwargs)
    ux_2 = convolve(data * data, mask=mask, **kwargs)
    return (ux_2 - u_x2 * u_x2) ** 0.5


def window3D(w):
    # TODO: make n dimensional (ndkernel(1dkernel, ndim))
    # TODO: make n dimensional from function (ndkernel(1dfun, bin_shape))
    # Convert a 1D filtering kernel to 3D
    # eg, window3D(numpy.hanning(5))
    L = w.shape[0]
    m1 = np.outer(np.ravel(w), np.ravel(w))
    win1 = np.tile(m1, np.hstack([L, 1, 1]))
    m2 = np.outer(np.ravel(w), np.ones([1, L]))
    win2 = np.tile(m2, np.hstack([L, 1, 1]))
    win2 = np.transpose(win2, np.hstack([1, 2, 0]))
    win = np.multiply(win1, win2)
    return win


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


@attrs(auto_attribs=True)
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


@attrs(auto_attribs=True)
class DetectionResult:
    peaks: List[Peak] = attrib(factory=list)
    heatmaps = None


class PeakDetector:
    @abstractmethod
    def detect(data):
        pass


@attrs(auto_attribs=True)
class CountPeakDetector(PeakDetector):
    bin_shape: Union[list, np.ndarray]
    mask: Union[list, np.ndarray] = None
    nyquist_offset = True
    min_count: Union[int, float] = 5
    min_dif: Union[int, float] = 10
    min_sigma_dif: Union[int, float] = None
    min_significance: Union[int, float] = 1
    max_num_peaks: Union[int, float] = np.inf
    min_interpeak_dist: Union[int, float] = 1
    offsets = None

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
            self.offsets = np.atleast_2d(np.zeros(dim))
        else:
            values = np.vstack((np.array(self.bin_shape) / 2, np.zeros(dim))).T
            combinations = np.array(np.meshgrid(*values)).T.reshape((-1, dim))
            self.offsets = np.flip(combinations, axis=0)

    def detect(self, data: np.ndarray, heatmaps: bool = False):
        if len(data.shape) != 2:
            raise ValueError("data array must have 2 dimensions")
        obs, dim = data.shape

        if self.mask is None:
            self.mask = get_default_mask(dim)
        self.mask = np.array(self.mask)
        mask = self.mask

        # TODO: do in init
        self.bin_shape = np.array(self.bin_shape)
        if len(mask.shape) != dim:
            raise ValueError("mask does not match data dimensions")
        if len(self.bin_shape) != dim:
            raise ValueError("bin_shape does not match data dimensions")

        if self.min_count:
            data = self.trim_low_density_regions(data)

        # TODO: do it in init
        self.set_nyquist_offsets()

        bins, ranges = get_histogram_bins(data, self.bin_shape)

        # set detection parameters for all runs
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
        for offset in self.offsets:
            hist, edges = histogram(data, self.bin_shape, offset)
            smoothed = convolve(hist, mask=mask)
            sharp = hist - smoothed
            std = fast_std_filter(hist, mask=mask)
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
            normalized = sharp / np.sqrt(hist + std**2 + 1)

            if self.min_dif is not None:
                # check for other way to implement
                normalized[sharp < self.min_dif] = 0
            if self.min_sigma_dif is not None:
                normalized[sharp < self.min_sigma_dif * std] = 0

            clusters_idx = peak_local_max(normalized, **peak_detection_params).T

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

        if heatmaps:
            # it will return heatmap corresponding to last offset only
            res.heatmaps = create_heatmaps(
                hist,
                edges,
                self.bin_shape,
                np.array([peak.index for peak in global_peaks]).T,
            )
        return res


def create_heatmaps(hist, edges, bin_shape, clusters_idx):
    dim = len(hist.shape)
    labels = [(np.round(edges[i] + bin_shape[i] / 2, 2))[:-1] for i in range(dim)]
    if dim == 2:
        data = hist
        annot_idx = clusters_idx
        annot = np.ndarray(shape=data.shape, dtype=str).tolist()
        for i in range(annot_idx.shape[1]):
            annot[annot_idx[0, i]][annot_idx[1, i]] = str(
                round(data[annot_idx[0][i]][annot_idx[1][i]])
            )
        ax = sns.heatmap(
            data, annot=annot, fmt="s", yticklabels=labels[0], xticklabels=labels[1]
        )
        hlines = np.concatenate((annot_idx[0], annot_idx[0] + 1))
        vlines = np.concatenate((annot_idx[1], annot_idx[1] + 1))
        ax.hlines(hlines, *ax.get_xlim(), color="w")
        ax.vlines(vlines, *ax.get_ylim(), color="w")
    else:
        cuts = np.unique(clusters_idx[2])
        ncuts = cuts.size
        ncols = min(3, ncuts)
        nrows = math.ceil(ncuts / ncols)
        delete_last = nrows > ncuts / ncols
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5))
        for row in range(nrows):
            for col in range(ncols):
                idx = col * nrows + row
                if idx < cuts.size:
                    cut_idx = cuts[idx]
                    data = hist[:, :, cut_idx]
                    annot_idx = clusters_idx.T[(clusters_idx.T[:, 2] == cut_idx)].T[:2]
                    annot = np.ndarray(shape=data.shape, dtype=str).tolist()
                    for i in range(annot_idx.shape[1]):
                        annot[annot_idx[0, i]][annot_idx[1, i]] = str(
                            round(data[annot_idx[0][i]][annot_idx[1][i]])
                        )
                    if ncuts <= 1:
                        subplot = ax
                    else:
                        if nrows == 1:
                            subplot = ax[col]
                        else:
                            subplot = ax[row, col]
                    current_ax = sns.heatmap(
                        data,
                        annot=annot,
                        fmt="s",
                        yticklabels=labels[0],
                        xticklabels=labels[1],
                        ax=subplot,
                    )
                    current_ax.axes.set_xlabel("x")
                    current_ax.axes.set_ylabel("y")
                    current_ax.title.set_text(
                        f"z slice at value {round(edges[2][cut_idx]+bin_shape[2]/2, 4)}"
                    )
                    hlines = np.concatenate((annot_idx[0], annot_idx[0] + 1))
                    vlines = np.concatenate((annot_idx[1], annot_idx[1] + 1))
                    current_ax.hlines(hlines, *current_ax.get_xlim(), color="w")
                    current_ax.vlines(vlines, *current_ax.get_ylim(), color="w")
        if delete_last:
            ax.flat[-1].set_visible(False)
        fig.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.tight_layout()
    return ax


# data = np.vstack((np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))).T
# bin_shape = [.5, .5]
# offsets = [-0.2, 0.2]
# bins, ranges = get_histogram_bins(data, bin_shape, offsets)
# assert np.equal(ranges, np.array([[-.05, .05], [-.05, .05]]))

# test_detection()
# rang1 = np.arange(1, 10, 1)
# rang2 = np.array([4, 5, 6]*4)
# var1 = np.concatenate((rang1, rang2))
# var2 = np.concatenate((np.flip(rang2), np.flip(rang1)))
# data will result in a diagonal histogram
# with more density in 3 center elements
##data = np.vstack((var1, var2)).T
# bin_shape = [1, 1]
# bins, _ = get_histogram_bins(data, bin_shape)
# assert np.allclose(bins, np.array([9,9]))

# mask = np.ones((3,3))/9
# data2 has some points deleted, so the histogram
# generated will have just the center region
# and enough surrounding bins to use the mask given
# data2 = CountPeakDetector(bin_shape=[1,1], mask=mask, min_count=5).trim_low_density_regions(data)

# bins, _ = get_histogram_bins(data2, bin_shape)
# assert np.allclose(bins, np.array([5,5]))

# data3 = CountPeakDetector(bin_shape=[1,1], mask=mask, min_count=6).trim_low_density_regions(data)


# print(no_outliers)


def three_clusters_sample():
    field_size = int(1e4)
    cluster_size = int(1e2)
    field = StarField(
        pm=multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=100),
        n_stars=field_size,
    )
    clusters = [
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 5]), cov=0.5
            ),
            pm=multivariate_normal(mean=(0.5, 0), cov=1.0 / 10),
            n_stars=cluster_size,
        ),
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.8, -28.6, 5]), cov=0.5
            ),
            pm=multivariate_normal(mean=(4.5, 4), cov=1.0 / 10),
            n_stars=cluster_size,
        ),
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.9, -28.7, 5]), cov=0.5
            ),
            pm=multivariate_normal(mean=(7.5, 7), cov=1.0 / 10),
            n_stars=cluster_size,
        ),
    ]
    df = Synthetic(star_field=field, clusters=clusters).rvs()
    return df


""" df = three_clusters_sample()[['pmra', 'pmdec', 'log10_parallax']]
data = df.values
result = CountPeakDetector(bin_shape=[.5, .5, .05], min_count=5, min_dif=20).detect(data, heatmaps=True)
center1 = (.5, 0, np.log10(5))
center2 = (4.5, 4, np.log10(5))
center3 = (7.5, 7, np.log10(5))
# test that the peak detector can detect peaks
# in a simple case
print('coso') """
