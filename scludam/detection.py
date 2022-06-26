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

"""Module for density peak detection in numerical data.

This module provides a class for density peak detection in numerical data, with
some extra helper functions for tasks such as defining filtering masks.
"""

from decimal import Decimal
from numbers import Number
from typing import List
from warnings import warn

import numpy as np
from astropy.stats.sigma_clipping import sigma_clipped_stats
from attrs import define, field, validators
from beartype import beartype
from numpy.typing import NDArray
from scipy import ndimage
from skimage.feature import peak_local_max

from scludam.masker import RangeMasker
from scludam.type_utils import (
    ArrayLike,
    Numeric1DArrayLike,
    Numeric2DArray,
    OptionalArrayLike,
    OptionalNumeric1DArrayLike,
    _type,
)


@beartype
def default_mask(dim: int):
    """Create a default mean mask for a given dimension.

    It returns a mean weighted mask with 5 elements per dimension,
    to be used as a filter for convolution.

    Parameters
    ----------
    dim : int
        Dimension of the mask.

    Returns
    -------
    NDArray[np.number]
        Array with the mask.

    Notes
    -----
    The mask has the shaped defined by _[1]. The sum of the mask amounts to 1.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mask_(image_processing)

    Examples
    --------
    .. literalinclude:: ../../examples/detection/default_mask.py
    :language: python
    :linenos:

    """
    indexes = np.array(np.meshgrid(*np.tile(np.arange(5), (dim, 1)))).T.reshape(
        (-1, dim)
    )
    mask = np.zeros([5] * dim)
    cond = np.sum((indexes - 2) ** 2, axis=1)
    mask[tuple(indexes[np.argwhere((cond > 0) & (cond < 5))].reshape((-1, dim)).T)] = 1
    return mask / np.count_nonzero(mask)


def _convolve(
    data, mask: OptionalArrayLike = None, c_filter: callable = None, *args, **kwargs
):
    if c_filter:
        return c_filter(data, *args, **kwargs)
    if mask is not None:
        return ndimage.convolve(data, mask, *args, **kwargs)


# unused
# def var_filter(data, mask=None, *args, **kwargs):
#     if mask is not None:
#         kwargs["footprint"] = mask != 0
#     return _convolve(
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
#     return _convolve(
#         data,
#         c_filter=ndimage.generic_filter,
#         function=np.std,
#         *args,
#         **kwargs,
#     )


@beartype
def fast_std_filter(data: NDArray, mask: ArrayLike, **kwargs):
    """Fast standard deviation filter.

    To be applied over a image or histogram.

    Parameters
    ----------
    data : NDArray
        Image or n-dimensional histogram.
    mask : ArrayLike
        Mask to be used for the convolution.

    Returns
    -------
    NDArray
        Filtered image or histogram of the same dimensions.

    Notes
    -----
    Is possible to pass kwargs to the ndimage.convolve function.
    Its default border mode is 'reflect'.

    """
    u_x2 = _convolve(data, mask=mask, **kwargs)
    ux_2 = _convolve(data * data, mask=mask, **kwargs)
    return (ux_2 - u_x2 * u_x2) ** 0.5


@beartype
def get_histogram_bins(
    data: Numeric2DArray,
    bin_shape: Numeric1DArrayLike,
    offsets: OptionalNumeric1DArrayLike = None,
):
    """Get histogram bins and edges given a bin shape and data.

    The methods takes into account the data max and min values
    for each dimension and the bin shape to stablish the amount
    of bins and the edges to be used in a histogram. Half a bin
    is added to each extremum to avoid bins edges to be exactly
    on the data extremums.

    Parameters
    ----------
    data : Numeric2DArray
        Data to be used to get the histogram bins.
    bin_shape : Numeric1DArrayLike
        Bin shape (each dimension) to be used.
    offsets : OptionalNumeric1DArrayLike, optional
        Offsets to be added to the edges, by default None

    Returns
    -------
    (Numeric1DArray, Numeric2DArray)
        Number of bins and edges for each dimension.

    """
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


@beartype
def histogram(
    data: Numeric2DArray,
    bin_shape: Numeric1DArrayLike,
    offsets: OptionalNumeric1DArrayLike = None,
):
    """Get a histogram given a bin shape and data.

    Uses :func:`~scludam.detection.get_histogram_bins` results
    to create a n-dimensional histogram.

    Parameters
    ----------
    data : Numeric2DArray
        Data to be used.
    bin_shape : Numeric1DArrayLike
        Bin shape (in each dimension) to be used.
    offsets : OptionalNumeric1DArrayLike, optional
        Offsets to shift the edges of the histogram, by default None

    Returns
    -------
    (NDArray, NDArray)
        Histogram and edges for each dimension.

    """
    _, dim = data.shape
    bins, ranges = get_histogram_bins(data, bin_shape, offsets)
    hist, edges = np.histogramdd(data, bins=bins, range=ranges, density=False)
    return hist, edges


@beartype
def nyquist_offsets(bin_shape: Numeric1DArrayLike):
    """Get offsets for shifting a histogram.

    Get all possible offsets for a given bin shape, to be used
    to shift the histogram edges following the Nyquist frequency
    sampling rule. The offsets are calculated as the half of the
    bin shape.

    Parameters
    ----------
    bin_shape : Numeric1DArrayLike
        Bin shape (in each dimension).

    Returns
    -------
    Numeric2DArray
        Array of shape (n_combinations, n_dimensions) with the
        possible offsets.

    """
    dim = len(bin_shape)
    if not dim:
        return []
    values = np.vstack((np.array(bin_shape) / 2, np.zeros(dim))).T
    combinations = np.array(np.meshgrid(*values)).T.reshape((-1, dim))
    return np.flip(combinations, axis=0)


def _are_indices_adjacent(a: Numeric1DArrayLike, b: Numeric1DArrayLike):
    return np.all(np.abs(np.asarray(a) - np.asarray(b)) <= 1)


@beartype
def extend_1dmask(mask: ArrayLike, dim: int):
    """Extend a 1D filtering mask to a ND mask.

    From a numeric filtering 1D mask use the outer product to
    extend it to a n-dimensional mask. The resulting mask is
    the combination of n-dimensions 1D masks orthogonal to
    each other. The sum of the resulting mask is equal to 1.

    Parameters
    ----------
    mask : ArrayLike
        1D mask to be extended.
    dim : int
        Dimension of the mask to be extended to.

    Returns
    -------
    NDArray
        Extended mask.

    """
    m1 = np.asarray(mask)
    mi = m1
    for i in range(dim - 1):
        mi = np.multiply.outer(mi, m1)
    return mi / mi.sum()


def _get_higher_score_offset_per_peak(indices: List, scores: List):
    if not len(indices) or not len(scores):
        return []
    if len(scores) != len(indices):
        raise ValueError("indices and scores must have the same length")
    peaks = list(zip(list(range(len(indices))), indices, scores))
    best = [peaks[0]]
    peaks = peaks[1:]
    while len(peaks) > 0:
        peak = peaks.pop(0)
        ii, iindex, iscore = peak
        j = 0
        while j < len(best):
            ji, jindex, jscore = best[j]
            if ji != ii and _are_indices_adjacent(iindex, jindex):
                if iscore > jscore:
                    # keep same jindex but update peak
                    # so it does not follow a path of
                    # adjacent best peaks indefinitely
                    best[j] = [ii, jindex, iscore]
                break
            j += 1
        if j == len(best):
            best.append(peak)

    best = sorted(best, key=lambda x: x[2], reverse=True)
    return np.array([p[0] for p in best])


@define
class DetectionResult:
    """Result of a detection.

    Attributes
    ----------
    centers : Numeric2DArray
        Centers of the detected peaks. Are calculated using
        sigma clipped median over the data delimited by the
        ``edges``.
    edges : Numeric2DArray
        Edges that delimit the data used to calculate the
        centers. It is taken as a bin shape in each direction,
        in each dimension.
    scores : Numeric1DArray
        Scores of the detected peaks. The results are sorted
        by score in descending order.
    counts : Numeric1DArray
        Number of data points in the bin that represents the
        peak.
    sigmas : Numeric1DArray
        Sigma of the detected peaks. Currently arbitrarily set
        as the bin shape in each dimension.

    """

    centers: Numeric2DArray = np.array([])
    sigmas: NDArray[np.number] = np.array([])
    scores: NDArray[np.number] = np.array([])
    counts: NDArray[np.number] = np.array([])
    edges: Numeric2DArray = np.array([])


# TODO: add a plot for the result
@define
class CountPeakDetector:
    """Count peak detector class.

    Uses an n-dimensional histogram (or "image") to detect density
    peaks in the input data.

    Attributes
    ----------
    bin_shape : Numeric1DArrayLike
        Bin shape (in each dimension) to be used to create the histogram.
    mask: OptionalArrayLike, optional
        Mask to be used as in the filtering operations, by default uses
        :func:~scludam.detection.default_mask with the data dimensions.
    nyquist_offsets : bool, optional
        If True, the Nyquist frequency sampling rule is used to shift the
        histogram edges, by default True. It helps to avoid the peak detection
        to miss the peaks or underestimate the bin count due to an arbitrarily
        chosen bin edge shift.
    min_count: Number, optional
        Mimimum count to be elegible for a peak, by default 5. Also used to
        ``remove_low_density_regions`` if the option is enabled.
    remove_low_density_regions : bool, optional
        If ``True``, low density bins are removed from the histogram, by default
        True. It removes low density bins from the edges of the histogram,
        trimming down the region of interest and reducing the size of the
        histogram, which in turn reduces memory usage for sparse data. It uses
        the min_count value as the threshold. It also keeps bins that are in the
        neibourhood of a valid (dense) bin so the filtering operation can be applied
        to the remaining bins correctly. The neighborhood is defined by the size of
        the ``mask`` to be used for the filtering operations.
    min_dif: Number, optional
        Minimum difference between the background and the bin count to be considered
        elegible as peak, by default 10. The formula used is:
        ``elegible if histogram - background > min_dif`` where ``background`` is
        obtained by using filtering the histogram with the provided ``mask``.
    min_sigma_dif: Number, optional
        Sigma value to be used to calculate difference between the background and the
        bin count to define if a bin is elegible as peak, by default None (deactivated).
        The formula used is:
        ``elegible if histogram - background > min_sigma_dif*std`` where ``background``
        is obtained by using filtering the histogram with the provided
        ``mask`` and ``std``
        represents the standard deviation in a window surrounding the bin, calculated
        according to the ``norm_mode`` parameter.
    min_score: Number, optional
        Minimum score for a bin to be elegible as peak, by default 1. The score is
        calculated as the standardized difference between the bin count and the
        background:
        ``score = (histogram - background) / std`` where ``background`` is obtained by
        using filtering the histogram with the provided ``mask`` and ``std`` is
        calculated according to the ``norm_mode`` parameter.
    min_interpeak_distance: int, optional
        Minimum distance between peaks in number of bins, by default 1.
    norm_mode: str, optional
        Mode to be used to get the standard deviation used in the
        score calculation, by default "std". Can be one of the following:
        #.  "std": Standard deviation of the ``histogram - background`` calculated
                using the ``mask`` provided and
                :func:`~scludam.detection.fast_std_filter`

        #.  "approx": Approximation _[1] to the standard deviation taking into account
                how the ``sharp = histogram - background`` is obtained.

            *   An common estimate of the standard deviation of an histogram
                    is the root square of the bin count: ``std(h) = sqrt(h)``.
            *   According to the uncertainty propagation:
                    if ``s = h - b``, then
                    ``std(s) = sqrt(var(h) + var(b) - 2*cov(h,b))``.
            *   Considering ``2*cov(h,b)~0``, the approximation is:
                    ``std(s) = sqrt(h + b)``.

    Examples
    --------
    .. literalinclude:: ../../examples/detection/count_peak_detector.py
        :language: python
        :linenos:

    """

    bin_shape: Numeric1DArrayLike = field(validator=_type(Numeric1DArrayLike))
    mask: OptionalArrayLike = field(default=None, validator=_type(OptionalArrayLike))
    nyquist_offset: bool = field(default=True)
    min_count: Number = field(default=5)
    min_dif: Number = field(default=10)
    min_sigma_dif: Number = field(default=None)
    min_score: Number = field(default=1)
    max_num_peaks: int = field(default=np.inf)
    min_interpeak_dist: int = field(default=1, validator=_type(int))
    remove_low_density_regions: bool = field(default=True, validator=_type(bool))
    norm_mode: str = field(default="std", validator=validators.in_(["std", "approx"]))
    _offsets: OptionalArrayLike = None

    def _remove_low_density_regions(self, data: Numeric2DArray):
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

    def _set_nyquist_offsets(self):
        dim = len(self.bin_shape)
        if not self.nyquist_offset:
            self._offsets = np.atleast_2d(np.zeros(dim))
        else:
            values = np.vstack((np.array(self.bin_shape) / 2, np.zeros(dim))).T
            combinations = np.array(np.meshgrid(*values)).T.reshape((-1, dim))
            self._offsets = np.flip(combinations, axis=0)

    def detect(self, data: Numeric2DArray):
        """Detect peaks in the provided data.

        Uses the configuration provided in the class attributes.

        Parameters
        ----------
        data : Numeric2DArray
            Numerical data to be used to detect peaks.

        Returns
        -------
        DetectionResult
            Instance containing the detected peaks.


        Raises
        ------
        ValueError
            If ``remove_low_density_regions`` is used and no bin passes
            the density check, the min_count is probably too low.
        ValueError
            If ``data``, ``bin_shape`` and ``mask`` dimensionality does not match.
        Warns
        -----
        UserWarning
            If histogram has too few bins in some dimension, that causes that the
            filtering operations can still be applied but prone to border effects.

        """
        if len(data.shape) != 2:
            raise ValueError("data array must have 2 dimensions")
        obs, dim = data.shape

        # mask setup
        if self.mask is None:
            self.mask = default_mask(dim)
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
            data = self._remove_low_density_regions(data)

        # set nyquist offsets
        self._set_nyquist_offsets()

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
        if self.min_score:
            peak_detection_params["threshold_abs"] = self.min_score
        if self.max_num_peaks:
            peak_detection_params["num_peaks"] = self.max_num_peaks

        # detection
        g_centers = []
        g_scores = []
        g_edges = []
        g_sigmas = []
        g_counts = []
        g_indices = []
        for offset in self._offsets:
            hist, edges = histogram(data, self.bin_shape, offset)
            smoothed = _convolve(hist, mask=mask)
            sharp = hist - smoothed

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
                std = np.sqrt(smoothed + hist + 1)
            elif self.norm_mode == "std":
                # directly gettig std
                std = fast_std_filter(sharp, mask=mask) + 1

            normalized = sharp / std

            # TODO: remove, other way of getting the std approx
            # n4 = sharp / np.sqrt(std**2 + fast_std_filter(smoothed, mask=mask)**2 + 1)

            detection_img = np.copy(normalized)

            if self.min_dif is not None:
                detection_img[sharp < self.min_dif] = 0
            if self.min_sigma_dif is not None:
                detection_img[sharp < self.min_sigma_dif * std] = 0

            clusters_idx = peak_local_max(detection_img, **peak_detection_params).T

            _, peak_count = clusters_idx.shape
            if peak_count != 0:

                iter_indcs = clusters_idx.T
                iter_counts = sharp[tuple(clusters_idx)]
                iter_scores = normalized[tuple(clusters_idx)]

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

                iter_edges = limits

                subsets = [
                    data[RangeMasker(limits=limits[j]).mask(data)]
                    for j in range(peak_count)
                ]

                # stats may be useless if other center and sigma are calculated
                # afterwards e.g. meanshift and profile analysis
                statistics = np.array(
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

                iter_centers = statistics[:, :, 1]

                g_indices += iter_indcs.tolist()
                g_centers += iter_centers.tolist()
                g_scores += iter_scores.tolist()
                g_edges += iter_edges
                g_sigmas += [np.array(self.bin_shape) for _ in range(peak_count)]
                g_counts += iter_counts.tolist()

        if len(g_indices) == 0:
            return DetectionResult()

        # compare same peaks in different histogram offsets
        # and return most sifnificant peak for all offsets
        g_ind = _get_higher_score_offset_per_peak(g_indices, g_scores)
        if self.max_num_peaks != np.inf:
            g_ind = g_ind[0 : self.max_num_peaks]

        g_centers = np.array(g_centers)[g_ind]
        g_scores = np.array(g_scores)[g_ind]
        g_edges = np.array(g_edges)[g_ind]
        g_sigmas = np.array(g_sigmas)[g_ind]
        g_indices = np.array(g_indices)[g_ind]
        g_counts = np.array(g_counts)[g_ind]

        return DetectionResult(
            centers=g_centers,
            sigmas=g_sigmas,
            scores=g_scores,
            counts=g_counts,
            edges=g_edges,
        )
