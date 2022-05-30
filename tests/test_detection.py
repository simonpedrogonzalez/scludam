from scludam.synthetic import (
    polar_to_cartesian,
    UniformSphere,
    Cluster,
    Field,
    Synthetic,
)
from scipy.stats import multivariate_normal
from scludam.detection import (
    get_histogram_bins,
    histogram,
    get_default_mask,
    fast_std_filter,
    nyquist_offsets,
    Peak,
    get_most_significant_peaks,
    CountPeakDetector,
)
import pandas as pd
import math
import numpy as np
import pytest


@pytest.fixture
def three_clusters_sample():
    field_size = int(1e4)
    cluster_size = int(2e2)
    field = Field(
        pm=multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=10),
        star_count=field_size,
    )
    clusters = [
        Cluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 5]), cov=0.5
            ),
            pm=multivariate_normal(mean=(0.5, 0), cov=1.0 / 10),
            star_count=cluster_size,
        ),
        Cluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.8, -28.6, 5]), cov=0.5
            ),
            pm=multivariate_normal(mean=(4.5, 4), cov=1.0 / 10),
            star_count=cluster_size,
        ),
        Cluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.9, -28.7, 5]), cov=0.5
            ),
            pm=multivariate_normal(mean=(7.5, 7), cov=1.0 / 10),
            star_count=cluster_size,
        ),
    ]
    df = Synthetic(field=field, clusters=clusters).rvs()
    return df


def diagonal1_data():
    x1 = np.arange(1, 10, 1)
    x2 = np.array([4, 5, 6] * 4)
    x = np.concatenate((x1, x2))
    # data will result in a ppal diagonal histogram of 9x9
    # with more density in 3 center elements
    data = np.vstack((x, x)).T
    return data


def diagonal2_data():
    x1 = np.arange(1, 10, 1)
    x2 = np.array([4, 5, 6] * 4)
    y1 = np.flip(x1)
    y2 = np.array([6, 5, 4] * 4)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    # data will result in the 2nd diagonal histogram of 9x9
    # with more density in 3 center elements
    data = np.vstack((x, y)).T
    return data


def test_get_histogram_bins():
    # should create enough margins to fit a integer number of bins
    # data has range 0 to .9
    data = np.vstack((np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))).T
    bins, ranges = get_histogram_bins(data, [0.5, 0.5])
    assert np.allclose(ranges, np.array([[-0.05, 0.95], [-0.05, 0.95]]))
    assert np.allclose(bins, np.array([2, 2]))
    # data has range 0 to 1
    data2 = np.vstack((np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))).T
    bins, ranges = get_histogram_bins(data2, [0.5, 0.5])
    assert np.allclose(ranges, np.array([[-0.25, 1.25], [-0.25, 1.25]]))
    assert np.allclose(bins, np.array([3, 3]))


def test_get_histogram_bins_offset():
    # offset should be applied to every dimension separately
    # range should be kept the same, but limits should be shifted
    data = np.vstack((np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))).T
    bins, ranges = get_histogram_bins(data, [0.5, 0.5], [-0.31, 0.17])
    assert np.allclose(ranges, np.array([[-0.56, 0.94], [-0.08, 1.42]]))
    assert np.allclose(bins, np.array([3, 3]))


def test_histogram():
    correct_edges = np.array([[-0.25, 0.25, 0.75, 1.25], [-0.25, 0.25, 0.75, 1.25]])
    data = np.vstack((np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))).T
    # should be 3 by 3 histogram, showing the linear data distribution
    hist, edges = histogram(data, [0.5, 0.5])
    assert np.allclose(hist, np.array([[3, 0, 0], [0, 5, 0], [0, 0, 3]]))
    assert np.allclose(edges, correct_edges)
    # should be 3 by 3 histogram, shifted to the right and down by 1 bin
    hist, edges = histogram(data, [0.5, 0.5], [0.5, 0.5])
    assert np.allclose(hist, np.array([[5, 0, 0], [0, 3, 0], [0, 0, 0]]))
    assert np.allclose(edges, correct_edges + 0.5)


def test_default_mask():
    # create a 5^dim float array as a mean mask
    # with the shape of a radial shell with a hole
    # in the middle
    mask = get_default_mask(2)
    assert np.allclose(
        mask,
        np.array(
            [
                [0.0, 0.0, 0.08333333, 0.0, 0.0],
                [0.0, 0.08333333, 0.08333333, 0.08333333, 0.0],
                [0.08333333, 0.08333333, 0.0, 0.08333333, 0.08333333],
                [0.0, 0.08333333, 0.08333333, 0.08333333, 0.0],
                [0.0, 0.0, 0.08333333, 0.0, 0.0],
            ]
        ),
    )
    assert np.allclose(mask.sum(), 1)


def test_fast_std_filter():
    mask = np.ones((3, 3)) / 9
    data = np.zeros((3, 3))
    data[1, 1] = 1
    result = fast_std_filter(data, mask)
    assert data.shape == result.shape
    # in this case, it can be checked this way
    # because mask takes into account all values
    # and the operation user mirror mode by default
    assert np.allclose(result, np.ones((3, 3)) * np.std(data))


def test_fast_std_filter_with_mask():
    mask = get_default_mask(2)
    data = np.zeros((5, 5))
    data[2, 2] = 1
    result = fast_std_filter(data, mask, mode="constant", cval=0)
    assert data.shape == result.shape
    assert np.allclose(
        result,
        np.array(
            [
                [0.0, 0.0, 0.2763854, 0.0, 0.0],
                [0.0, 0.2763854, 0.2763854, 0.2763854, 0.0],
                [0.2763854, 0.2763854, 0.0, 0.2763854, 0.2763854],
                [0.0, 0.2763854, 0.2763854, 0.2763854, 0.0],
                [0.0, 0.0, 0.2763854, 0.0, 0.0],
            ]
        ),
    )


def test_nyquist_offsets():
    dim = 2
    offsets = nyquist_offsets([0.5] * dim)
    assert offsets.shape == (2**dim, dim)
    assert np.allclose(offsets, np.array([[0, 0], [0, 0.25], [0.25, 0], [0.25, 0.25]]))


@pytest.mark.parametrize(
    "index1, index2, correct",
    [
        ((0, 0, 0), (0, 0, 0), True),
        ((0, 0, 0), (0, 1, 0), True),
        ((0, 0, 0), (1, 1, 1), True),
        ((0, 0, 0), (0, 0, 2), False),
    ],
)
def test_peak_is_in_neighbourhood(index1, index2, correct):
    assert Peak(np.array(index1)).is_in_neighbourhood(Peak(np.array(index2))) == correct


@pytest.mark.parametrize(
    "peaks, expected",
    [
        (
            [
                Peak(np.array([0, 0]), significance=3),
                Peak(np.array([0, 1]), significance=2),
                Peak(np.array([1, 2]), significance=6),
                Peak(np.array([2, 2]), significance=7),
            ],
            [[0, 0], [2, 2]],
        ),
        (
            [
                Peak(np.array([0, 1]), significance=3),
                Peak(np.array([0, 2]), significance=3),
            ],
            [[0, 1]],
        ),
        ([], []),
    ],
)
def test_get_most_significant_peaks(peaks, expected):
    result = get_most_significant_peaks(peaks)
    assert isinstance(result, list)
    for peak in result:
        assert list(peak.index) in expected
        expected.remove(list(peak.index))


@pytest.mark.parametrize(
    "data",
    [
        (diagonal1_data()),
        (diagonal2_data()),
    ],
)
def test_count_based_outlier_removal(data):
    bin_shape = [1, 1]
    bins, _ = get_histogram_bins(data, bin_shape)
    assert np.allclose(bins, np.array([9, 9]))
    mask = np.ones((3, 3)) / 9
    # data2 has some points deleted, so the histogram
    # generated will have just the 3 bins center region
    # and enough surrounding bins to use the mask given
    # in the valid region.
    data2 = CountPeakDetector(
        bin_shape=[1, 1], mask=mask, min_count=5
    ).trim_low_density_regions(data)
    bins, _ = get_histogram_bins(data2, bin_shape)
    assert np.allclose(bins, np.array([5, 5]))


def test_count_based_outlier_removal_high_min_count():
    data = diagonal1_data()
    mask = np.ones((3, 3)) / 9
    with pytest.raises(ValueError):
        CountPeakDetector(
            bin_shape=[1, 1], mask=mask, min_count=6
        ).trim_low_density_regions(data)


@pytest.mark.parametrize(
    "data, mask, bin_shape",
    [
        (np.zeros(10), [1, 1], [1, 1]),
        (diagonal1_data(), [1, 1, 1], [1, 1]),
        (diagonal1_data(), [1, 1], [1, 1, 1]),
    ],
)
def test_detect_errors(data, mask, bin_shape):
    with pytest.raises(ValueError):
        CountPeakDetector(bin_shape=bin_shape, mask=mask).detect(data)


def test_detect_warnings():
    with pytest.warns(UserWarning):
        CountPeakDetector(bin_shape=[1, 5], mask=np.ones((3, 3)) / 9).detect(
            diagonal1_data()
        )


def is_same_center(center1, center2, bin_shape):
    return np.all(np.abs(center1 - center2) < bin_shape / 2)


# test if convolve with mask really works


def test_detect_happy_pass(three_clusters_sample):
    df = three_clusters_sample[["pmra", "pmdec", "log10_parallax"]]
    data = df.values
    result = CountPeakDetector(
        bin_shape=[0.5, 0.5, 0.05], min_count=5, min_dif=20
    ).detect(data)
    center1 = (7.5, 7, np.log10(5))
    center2 = (4.5, 4, np.log10(5))
    center3 = (0.5, 0, np.log10(5))

    # test that the peak detector can detect peaks
    # in a simple case
    return True
