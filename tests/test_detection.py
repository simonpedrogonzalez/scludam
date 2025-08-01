import numpy as np
import pytest
import seaborn as sns
from scipy.signal.windows import gaussian as gaus_win
from scipy.stats import multivariate_normal
from utils import assert_eq_warn_message, squarediff

from scludam import CountPeakDetector, default_mask, extend_1dmask
from scludam.detection import (
    _are_indices_adjacent,
    _get_higher_score_offset_per_peak,
    fast_std_filter,
    get_histogram_bins,
    histogram,
    nyquist_offsets,
)
from scludam.synthetic import (
    StarCluster,
    StarField,
    Synthetic,
    UniformFrustum,
    UniformSphere,
    polar_to_cartesian,
)


@pytest.fixture
def three_clusters_sample():
    field_size = int(1e4)
    cluster_size = int(1e2)
    field = StarField(
        pm=multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformFrustum(locs=(120.5, -27.5, 12), scales=(1, 1, -11.8)),
        n_stars=field_size,
    )
    clusters = [
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 1.15]), cov=0.5
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
                mean=polar_to_cartesian([120.9, -28.7, 8]), cov=0.5
            ),
            pm=multivariate_normal(mean=(7.5, 7), cov=1.0 / 10),
            n_stars=cluster_size,
        ),
    ]
    df = Synthetic(star_field=field, clusters=clusters).rvs()
    return df


@pytest.fixture
def no_clusters_sample():
    field_size = int(1e3)
    field = StarField(
        pm=multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformFrustum(locs=(120.5, -27.5, 12), scales=(1, 1, -11.8)),
        n_stars=field_size,
    )
    clusters = []
    df = Synthetic(star_field=field, clusters=clusters).rvs()
    return df


@pytest.fixture
def low_variance_in_plx_sample():
    field_size = int(1e4)
    cluster_size = int(2e2)
    field = StarField(
        pm=multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=10),
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


@pytest.fixture
def one_cluster_sample():
    np.random.seed(0)
    field_size = int(1e4)
    cluster_size = int(1e2)
    field = StarField(
        pm=multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformFrustum(locs=(120.5, -27.5, 12), scales=(1, 1, -11.8)),
        n_stars=field_size,
    )
    clusters = [
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 1.15]), cov=0.5
            ),
            pm=multivariate_normal(mean=(0.5, 0), cov=1.0 / 10),
            n_stars=cluster_size,
        ),
    ]
    df = Synthetic(star_field=field, clusters=clusters).rvs()
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


def window3D(w):
    L = w.shape[0]
    m1 = np.outer(np.ravel(w), np.ravel(w))
    win1 = np.tile(m1, np.hstack([L, 1, 1]))
    m2 = np.outer(np.ravel(w), np.ones([1, L]))
    win2 = np.tile(m2, np.hstack([L, 1, 1]))
    win2 = np.transpose(win2, np.hstack([1, 2, 0]))
    win = np.multiply(win1, win2)
    return win / win.sum()


def window2D(w):
    m1 = np.outer(np.ravel(w), np.ravel(w))
    return m1 / m1.sum()


def detection_gonzalez_alejo_method(H, edges):
    mask = [[[True for k in range(5)] for j in range(5)] for i in range(5)]
    for i in range(5):
        for j in range(5):
            for k in range(5):
                mask[i][j][k] = 0 < (i - 2) ** 2 + (j - 2) ** 2 + (k - 2) ** 2 < 5
    mask = np.array(mask)

    sH = H.copy()
    # rmsH=H.copy()
    # meH=H.copy()
    sHH = H.copy()
    HH = H * H
    varH = H.copy()
    print("Calculando distribuciones filtradas en el espacio pmra,pmde,plx ...")
    for i in range(2, H.shape[0] - 2):
        #    print("i=",i)
        print("{:5d}".format(int(100 * (i - 2) / (H.shape[0] - 4))) + " % ")
        for j in range(2, H.shape[1] - 2):
            for k in range(2, H.shape[2] - 2):
                B = H[i - 2 : i + 3, j - 2 : j + 3, k - 2 : k + 3]
                BB = HH[i - 2 : i + 3, j - 2 : j + 3, k - 2 : k + 3]
                # maskedB = np.ma.array(B, mask=mask)
                # maskedBB = np.ma.array(BB, mask=mask)
                #            meH[i,j,k] =maskedB.mean()
                #           rmsH[i,j,k]=maskedB.std()
                sH[i, j, k] = B[mask].mean()  # np.ma.mean(maskedB)  # 27 segundos
                sHH[i, j, k] = BB[mask].mean()  # np.ma.mean(maskedBB)
                if i == 30 and j == 30 and k == 11:
                    print("coso")

    print("Localizando sobredensidades")
    snH = (H - sH) / np.sqrt(sH + 1 + varH)
    varH = np.sqrt(sHH)

    # not in the original implementation but useful for testing
    def largest_indices(ary, n):
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    indices = largest_indices(snH, 1)

    score = snH[indices]

    lpclu = edges[2][indices[2]] + 0.025
    pmraclu = edges[0][indices[0]] + 0.25
    pmdeclu = edges[1][indices[1]] + 0.25

    centers = np.vstack((pmraclu, pmdeclu, lpclu)).T

    return score, centers


def is_same_center(center1, center2, bin_shape):
    return np.all(
        np.abs(np.asarray(center1) - np.asarray(center2)) < np.asarray(bin_shape) / 2
    )


def are_same_centers(centers1, centers2, bin_shape):
    res1 = [False] * len(centers1)
    for i, c in enumerate(centers1):
        for j, c2 in enumerate(centers2):
            if is_same_center(c, c2, bin_shape):
                centers2.pop(j)
                res1[i] = True
                break
    return not len(centers2) and np.all(np.array(res1))


def det_res_has_shape(result, n, d):
    if n == 0:
        return np.all(
            [
                result.centers.size == n,
                result.scores.size == n,
                result.indices.size == n,
                result.sigmas.size == n,
                result.counts.size == n,
                result.edges.size == n,
                result.offsets.size == n,
            ]
        )
    return np.all(
        [
            result.centers.shape == (n, d),
            result.scores.shape == (n,),
            result.indices.shape == (n, d),
            result.sigmas.shape == (n, d),
            result.counts.shape == (n,),
            result.edges.shape == (n, d, 2),
            result.offsets.shape == (n, d),
        ]
    )


def test_extend_1dmask():
    win = gaus_win(10, 1)
    assert np.allclose(extend_1dmask(win, 1), win / win.sum())
    assert np.allclose(extend_1dmask(win, 2), window2D(win))
    assert np.allclose(extend_1dmask(win, 3), window3D(win))


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
    correct_edges = correct_edges + 0.5
    assert np.allclose(edges, correct_edges)


def test_default_mask_2d():
    # create a 5^dim float array as a mean mask
    # with the shape of a radial shell with a hole
    # in the middle
    mask = default_mask(2)
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


def test_default_mask_3d():
    def old_3D_mask():
        mask = [[[True for k in range(5)] for j in range(5)] for i in range(5)]
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    mask[i][j][k] = 0 < (i - 2) ** 2 + (j - 2) ** 2 + (k - 2) ** 2 < 5
        mask = np.array(mask).astype(np.float64)
        return mask / mask.sum()

    assert np.allclose(default_mask(3), old_3D_mask())


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
    mask = default_mask(2)
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


@pytest.mark.parametrize(
    "bin_shape, correct",
    [
        ([], []),
        ([0.5, 0.5], np.array([[0, 0], [0, 0.25], [0.25, 0], [0.25, 0.25]])),
        (
            [0.2, 0.4, 0.6],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.1, 0.2, 0.0],
                    [0.0, 0.0, 0.3],
                    [0.0, 0.2, 0.3],
                    [0.1, 0.0, 0.3],
                    [0.1, 0.2, 0.3],
                ]
            ),
        ),
    ],
)
def test_nyquist_offsets(bin_shape, correct):
    assert np.allclose(nyquist_offsets(bin_shape), correct)


# TODO? remove
@pytest.mark.parametrize(
    "index1, index2, correct",
    [
        ((0, 0, 0), (0, 0, 0), True),
        ((0, 0, 0), (0, 1, 0), True),
        ((0, 0, 0), (1, 1, 1), True),
        ((0, 0, 0), (0, 0, 2), False),
    ],
)
def test_are_indices_adjacent(index1, index2, correct):
    assert _are_indices_adjacent(index1, index2) == correct


@pytest.mark.parametrize(
    "indices, scores, correct",
    [
        ([], [1], []),
        ([1], [], []),
        ([(0, 0, 0), (0, 0, 1)], [1, 2], [1]),
        ([(0, 0, 0), (-1, 0, 0)], [2, 1], [0]),
        ([(0, 0, 0), (1, 1, 1), (0, 0, 2)], [1, 2, 3], [2, 1]),
        (
            [(0, 0, 0), (1, 1, 1), (0, 0, 2), (0, 1, 2)],
            [1, 2, 3, 4],
            [3, 1],
        ),
    ],
)
def test_get_higher_score_offset_per_peak(indices, scores, correct):
    idcs = _get_higher_score_offset_per_peak(indices, scores)
    assert np.allclose(idcs, np.array(correct))


def test_get_higher_score_different_length():
    with pytest.raises(ValueError, match="must have the same length"):
        _get_higher_score_offset_per_peak([(0, 0, 0)], [1, 2])


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
    )._remove_low_density_regions(data)
    bins, _ = get_histogram_bins(data2, bin_shape)
    assert np.allclose(bins, np.array([5, 5]))


def test_count_based_outlier_removal_high_min_count():
    data = diagonal1_data()
    mask = np.ones((3, 3)) / 9
    with pytest.raises(
        ValueError,
        match="""No bin passed minimum density check.
                            Check min_count parameter.""",
    ):
        CountPeakDetector(
            bin_shape=[1, 1], mask=mask, min_count=6
        )._remove_low_density_regions(data)


@pytest.mark.parametrize(
    "data, mask, bin_shape",
    [
        # this is controlled by beartype now
        # (np.zeros(10), [1, 1], [1, 1]),
        (diagonal1_data(), [1, 1, 1], [1, 1]),
        (diagonal1_data(), [1, 1], [1, 1, 1]),
    ],
)
def test_detect_errors(data, mask, bin_shape):
    with pytest.raises(ValueError):
        CountPeakDetector(bin_shape=bin_shape, mask=mask).detect(data)


def test_low_variance_warns_too_few_bins(low_variance_in_plx_sample):
    df = low_variance_in_plx_sample[["pmra", "pmdec", "log10_parallax"]]
    data = df.values
    with pytest.warns(UserWarning) as record:
        result = CountPeakDetector(
            bin_shape=[0.5, 0.5, 0.05], min_count=5, min_dif=20
        ).detect(data)
        assert_eq_warn_message(
            record,
            "Histogram has too few bins in some dimensions: bin numbers are [41 42  1]",
        )
    center1 = (7.5, 7, np.log10(5))
    center2 = (4.5, 4, np.log10(5))
    center3 = (0.5, 0, np.log10(5))
    centers = [center1, center2, center3]
    assert are_same_centers(centers, result.centers.tolist(), [0.5, 0.5, 0.05])


def test_accordance_with_gonzalez_alejo_method(one_cluster_sample):
    data = one_cluster_sample[["pmra", "pmdec", "log10_parallax"]].values
    real_center = np.array([0.5, 0, 1.15])

    H, edges = histogram(data, [0.5, 0.5, 0.05])
    score, centers = detection_gonzalez_alejo_method(H, edges)
    ga_center = centers.flatten()
    # as method is using log10_parallax
    ga_center[-1] = 10 ** ga_center[-1]
    result = CountPeakDetector(
        bin_shape=[0.5, 0.5, 0.05],
        nyquist_offset=False,
        remove_low_density_regions=False,
        norm_mode="approx",
    ).detect(data)
    res_center = result.centers[0].copy()
    # as method is using log10_parallax
    res_center[-1] = 10 ** res_center[-1]
    # score is the same
    assert np.allclose(result.scores[0], score)
    # center calculation is a little bit more accurate
    assert squarediff(real_center, res_center) < squarediff(real_center, ga_center)


def test_nyquist_offset_yields_better_center_calculation(one_cluster_sample):
    data = one_cluster_sample[["pmra", "pmdec", "log10_parallax"]].values
    real_center = np.array([0.5, 0, 1.15])
    result1 = CountPeakDetector(
        bin_shape=[0.5, 0.5, 0.05], nyquist_offset=False
    ).detect(data)
    result2 = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05]).detect(data)
    assert squarediff(result1.centers[0], real_center) > squarediff(
        result2.centers[0], real_center
    )


def test_multiple_clusters(three_clusters_sample):
    data = three_clusters_sample[["pmra", "pmdec", "log10_parallax"]].values
    real_centers = [
        [7.5, 7, np.log10(8)],
        [4.5, 4, np.log10(5)],
        [0.5, 0.0, np.log10(1.15)],
    ]
    result = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05]).detect(data)
    centers = result.centers.tolist()
    assert are_same_centers(real_centers, centers, [0.5, 0.5, 0.05])
    assert det_res_has_shape(result, 3, 3)
    assert np.allclose(
        result.sigmas, np.repeat(np.array([0.5, 0.5, 0.05])[:, np.newaxis], 3, axis=1).T
    )


def test_no_cluster_sample_yields_no_found_peaks(no_clusters_sample):
    data = no_clusters_sample[["pmra", "pmdec", "log10_parallax"]].values
    result = CountPeakDetector(
        bin_shape=[0.5, 0.5, 0.05], remove_low_density_regions=False
    ).detect(data)
    assert det_res_has_shape(result, 0, 3)


def test_max_n_peaks(three_clusters_sample):
    data = three_clusters_sample[["pmra", "pmdec", "log10_parallax"]].values
    result = CountPeakDetector(
        bin_shape=[0.5, 0.5, 0.05],
        max_n_peaks=2,
    ).detect(data)
    assert det_res_has_shape(result, 2, 3)


def test_detector_plot_error_no_result_available():
    with pytest.raises(
        ValueError, match="No result available, run detect function first"
    ):
        CountPeakDetector(bin_shape=[0.5, 0.5, 0.05]).plot()


def test_detector_plot_peak_out_of_range(one_cluster_sample):
    data = one_cluster_sample[["pmra", "pmdec", "log10_parallax"]].values
    with pytest.raises(ValueError, match="No peak with index 1 detected in last run."):
        detector = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05])
        detector.detect(data)
        detector.plot(peak=1)


def test_detector_plot_mode_error(one_cluster_sample):
    data = one_cluster_sample[["pmra", "pmdec", "log10_parallax"]].values
    with pytest.raises(ValueError, match="Mode must be one of 'c', 'b', 'e' or 's'."):
        detector = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05])
        detector.detect(data)
        detector.plot(mode="x")


def test_detector_plot_error_no_peak_detected(no_clusters_sample):
    data = no_clusters_sample[["pmra", "pmdec", "log10_parallax"]].values
    with pytest.raises(ValueError, match="No peaks detected in last run."):
        detector = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05], min_count=0)
        detector.detect(data)
        detector.plot()


def test_detector_plot_error_invalid_dims(one_cluster_sample):
    data = one_cluster_sample[["pmra", "pmdec", "log10_parallax"]].values
    with pytest.raises(ValueError, match="x and y must be valid dimensions."):
        detector = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05])
        detector.detect(data)
        detector.plot(x=-1, y=-1)


# more cases can be generated, because the function has many parameters
# but those are the basic cases, and each comparison generates an
# image
@pytest.mark.parametrize(
    "peak, x, y, mode",
    [
        (0, 0, 1, "c"),
        (1, 2, 1, "b"),
        (2, 1, 2, "s"),
        (0, 2, 0, "e"),
    ],
)
@pytest.mark.mpl_image_compare
def test_detector_plot_images(peak, x, y, mode, three_clusters_sample):
    sns.set(font_scale=0.5)
    data = three_clusters_sample[["pmra", "pmdec", "log10_parallax"]].values
    detector = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05])
    detector.detect(data)
    fig = detector.plot(
        peak=peak, x=x, y=y, mode=mode, annot_kws={"fontsize": 4}
    ).get_figure()
    return fig


@pytest.mark.mpl_image_compare
def test_detector_lineplot(three_clusters_sample):
    sns.set(font_scale=0.5)
    data = three_clusters_sample[["pmra", "pmdec", "log10_parallax"]].values
    detector = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05])
    detector.detect(data)
    fig, ax = detector.lineplot()
    return fig
