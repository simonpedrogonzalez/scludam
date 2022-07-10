import numpy as np
import pytest
from hdbscan import HDBSCAN
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
from utils import squarediff

from scludam import SHDBSCAN
from scludam.synthetic import BivariateUniform


@pytest.fixture
def iris():
    return load_iris().data


@pytest.fixture
def uniform_sample():
    return BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(1000)


@pytest.fixture
def one_cluster_sample():
    sample = BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(500)
    return np.concatenate((sample, sample2))


@pytest.fixture
def only_one_cluster_sample():
    sample = multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(500)
    return sample


@pytest.fixture
def two_clusters_sample():
    sample = BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = multivariate_normal(mean=(0.75, 0.75), cov=1.0 / 200).rvs(250)
    sample3 = multivariate_normal(mean=(0.25, 0.25), cov=1.0 / 200).rvs(250)
    return np.concatenate((sample, sample2, sample3))


@pytest.fixture
def three_clusters_sample():
    sample = BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = multivariate_normal(mean=(0.75, 0.75), cov=1.0 / 200).rvs(160)
    sample3 = multivariate_normal(mean=(0.25, 0.25), cov=1.0 / 200).rvs(160)
    sample4 = multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(160)
    return np.concatenate((sample, sample2, sample3, sample4))


def assert_shdbscan_result_ok(
    shdbscan,
    n_classes,
    n,
    no_noise=False,
    noise_proba_mode=None,
    cluster_proba_mode=None,
):
    assert shdbscan.n_classes == n_classes
    assert shdbscan.proba.shape == (n, n_classes)
    assert np.allclose(shdbscan.proba.sum(axis=1), 1)
    assert shdbscan.labels.size == n
    if no_noise:
        assert np.allclose(np.unique(shdbscan.labels), np.arange(n_classes))
    else:
        assert np.allclose(np.unique(shdbscan.labels), np.arange(-1, n_classes - 1))
    if noise_proba_mode is not None and n_classes > 1 and not no_noise:
        if noise_proba_mode == "conservative":
            # there is no cluster classified as noise whose cluster
            # probability is grater than zero
            # for any cluster
            assert np.all(
                shdbscan.proba[:, 1:].sum(axis=1)[shdbscan.clusterer.labels_ == -1] == 0
            )
        if noise_proba_mode == "outlier":
            assert np.all(shdbscan.proba[:, 0] >= shdbscan.outlier_scores)
    if cluster_proba_mode is not None:
        if n_classes >= 2 and (
            cluster_proba_mode == "hard" or shdbscan.clusterer.allow_single_cluster
        ):
            # no point has positive probability for 2 clusters at the same time
            assert np.all((shdbscan.proba[:, 1:] > 0).astype(int).sum(axis=1) <= 1)


def test_min_sample_parameter_set_clusterer_min_samples_parameter(iris):
    shdbscan = SHDBSCAN(min_cluster_size=20, min_samples=10).fit(iris)
    assert shdbscan.clusterer.min_samples == 10


def test_clusterer_parameter_set_clusterer_parameter(iris):
    shdbscan = SHDBSCAN(
        clusterer=HDBSCAN(
            min_cluster_size=30,
            min_samples=10,
            allow_single_cluster=True,
        ),
        auto_allow_single_cluster=True,
    ).fit(iris)
    assert shdbscan.n_classes == 3
    assert shdbscan.proba.shape == (iris.shape[0], 3)
    assert shdbscan.clusterer.min_cluster_size == 30
    assert shdbscan.clusterer.min_samples == 10
    assert not shdbscan.clusterer.allow_single_cluster


def test_min_cluster_size_validator_errors(iris):
    with pytest.raises(
        ValueError, match="Either min_cluster_size or clusterer must be provided."
    ):
        SHDBSCAN().fit(iris)
    with pytest.raises(ValueError, match="min_cluster_size must be greater than 1."):
        SHDBSCAN(min_cluster_size=-1).fit(iris)
    with pytest.raises(
        ValueError, match="outlier_quantile selected is too low, the value for it is 0."
    ):
        SHDBSCAN(min_cluster_size=20, outlier_quantile=0.01).fit(iris)


@pytest.mark.parametrize("noise_proba_mode", ["score", "outlier", "conservative"])
@pytest.mark.parametrize("cluster_proba_mode", ["soft", "hard"])
class TestSHDBSCAN:
    # case there are no clusters
    #   a. not forced -> finds none
    def test_uniform_sample_finds_no_clusters(
        self, uniform_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=100,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(uniform_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            1,
            uniform_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #   b. forced
    #       I. by asc -> finds 1
    def test_uniform_sample_forced_asc_finds_one_cluster(
        self, uniform_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=100,
            allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(uniform_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            2,
            uniform_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #       II. by aasc -> finds 1
    def test_uniform_sample_forced_aasc_finds_one_cluster(
        self, uniform_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=100,
            auto_allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(uniform_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            2,
            uniform_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    # case there is only 1 cluster without noise (without field)
    #   a. not forced -> finds none
    def test_no_noise_sample_finds_no_clusters(
        self, only_one_cluster_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=500,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(only_one_cluster_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            1,
            only_one_cluster_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #  b. forced
    #      I. by asc -> finds 1
    def test_no_noise_sample_forced_asc_finds_one_cluster_no_noise(
        self, only_one_cluster_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=500,
            allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(only_one_cluster_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            1,
            only_one_cluster_sample.shape[0],
            no_noise=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #      II. by aasc -> finds 1
    def test_no_noise_sample_forced_aasc_finds_one_cluster_no_noise(
        self, only_one_cluster_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=500,
            auto_allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(only_one_cluster_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            1,
            only_one_cluster_sample.shape[0],
            no_noise=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    # case there is 1 cluster with noise
    #   a. not forced -> finds none
    def test_one_cluster_sample_finds_no_clusters(
        self, one_cluster_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=500,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(one_cluster_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            1,
            one_cluster_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #   b. forced
    #       I. by asc -> finds 1
    def test_one_cluster_sample_forced_asc_finds_one_cluster(
        self, one_cluster_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=500,
            allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(one_cluster_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            2,
            one_cluster_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #       II. by aasc -> finds 1
    def test_one_cluster_sample_forced_aasc_finds_one_cluster(
        self, one_cluster_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=500,
            auto_allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(one_cluster_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            2,
            one_cluster_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    # case there are 2 clusters
    #   a. not forced -> finds 2
    def test_two_clusters_sample_finds_two_clusters(
        self, two_clusters_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=250,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(two_clusters_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            3,
            two_clusters_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #   b. forced
    #       I. by asc -> finds 1
    def test_two_clusters_sample_forced_asc_finds_one_cluster(
        self, two_clusters_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=250,
            allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(two_clusters_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            2,
            two_clusters_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #       II. by aasc -> finds 2
    def test_two_clusters_sample_forced_aasc_finds_one_cluster(
        self, two_clusters_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=250,
            auto_allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(two_clusters_sample)
        assert_shdbscan_result_ok(
            shdbscan,
            3,
            two_clusters_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    #   d. not forced with 1 center -> finds 1
    def test_two_clusters_sample_center_specified_finds_one_cluster(
        self, two_clusters_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=250,
            allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(two_clusters_sample, centers=(0.25, 0.25))
        assert_shdbscan_result_ok(
            shdbscan,
            2,
            two_clusters_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )
        # found correct one
        center = two_clusters_sample[shdbscan.labels == 0].mean(axis=0)
        assert squarediff(center, (0.25, 0.25)) < squarediff(center, (0.75, 0.75))

    #   e. not forced with 2 centers -> finds 2
    def test_two_clusters_sample_two_centers_specified_finds_two_clusters(
        self, two_clusters_sample, noise_proba_mode, cluster_proba_mode
    ):
        # centers must be ignored if n_found_clusters <= centers count
        shdbscan = SHDBSCAN(
            min_cluster_size=250,
            allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(two_clusters_sample, centers=[(30, 30), (500, 500)])
        assert_shdbscan_result_ok(
            shdbscan,
            3,
            two_clusters_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )

    # case there are 3 clusters with 2 centers, and auto force -> finds 2
    def test_three_clusters_sample_two_centers_specified_finds_two_clusters(
        self, three_clusters_sample, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=90,
            auto_allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(three_clusters_sample, centers=[(0.75, 0.75), (0.25, 0.25)])
        assert_shdbscan_result_ok(
            shdbscan,
            3,
            three_clusters_sample.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )
        # found correct one and cluster order preserved
        center = three_clusters_sample[shdbscan.labels == 0].mean(axis=0)
        assert squarediff(center, (0.75, 0.75)) < squarediff(center, (0.25, 0.25))
        center2 = three_clusters_sample[shdbscan.labels == 1].mean(axis=0)
        assert squarediff(center2, (0.25, 0.25)) < squarediff(center2, (0.75, 0.75))

    # iris for good measure
    def test_iris_finds_three_clusters(
        self, iris, noise_proba_mode, cluster_proba_mode
    ):
        shdbscan = SHDBSCAN(
            min_cluster_size=20,
            auto_allow_single_cluster=True,
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        ).fit(iris)
        assert_shdbscan_result_ok(
            shdbscan,
            3,
            iris.shape[0],
            noise_proba_mode=noise_proba_mode,
            cluster_proba_mode=cluster_proba_mode,
        )


def test_outlier_quantile(three_clusters_sample):
    shdbscan = SHDBSCAN(min_cluster_size=90, outlier_quantile=0.8).fit(
        three_clusters_sample
    )
    maxos = np.quantile(shdbscan.clusterer.outlier_scores_, 0.8)
    os = shdbscan.clusterer.outlier_scores_
    os[os > maxos] = maxos
    os = os / maxos
    assert shdbscan.noise_proba_mode == "outlier"
    assert np.allclose(shdbscan.outlier_scores, os)
    assert np.all(shdbscan.proba[:, 0] >= os)


def test_validity_index(three_clusters_sample):
    from hdbscan.validity import validity_index

    shdbscan = SHDBSCAN(min_cluster_size=90).fit(three_clusters_sample)
    assert np.allclose(
        validity_index(three_clusters_sample, shdbscan.labels),
        shdbscan.validity_index(),
    )


class TestSHDBSCANplots:
    @pytest.mark.mpl_image_compare
    def test_shdbscan_outlierplot(self, iris):
        shdbscan = SHDBSCAN(min_cluster_size=20, outlier_quantile=0.8).fit(iris)
        fig = shdbscan.outlierplot(color="k", bins=20).get_figure()
        return fig

    @pytest.mark.mpl_image_compare
    def test_shdbscan_pairplot(self, iris):
        shdbscan = SHDBSCAN(min_cluster_size=20).fit(iris)
        fig = shdbscan.pairplot(
            diag_kind="hist",
            palette="copper",
            corner=True,
            cols=load_iris().feature_names,
            diag_kws={"bins": 20},
        ).fig
        return fig

    @pytest.mark.mpl_image_compare
    def test_shdbscan_tsneplot(self, iris):
        shdbscan = SHDBSCAN(min_cluster_size=20).fit(iris)
        np.random.seed(0)
        fig = shdbscan.tsneplot(palette="copper").get_figure()
        return fig

    @pytest.mark.mpl_image_compare
    def test_shdbscan_surfplot(self, iris):
        shdbscan = SHDBSCAN(min_cluster_size=20).fit(iris)
        fig = shdbscan.surfplot(
            x=1, y=2, palette="copper", cols=load_iris().feature_names
        )[0]
        return fig

    @pytest.mark.mpl_image_compare
    def test_shdbscan_scatter3dplot(self, iris):
        shdbscan = SHDBSCAN(min_cluster_size=20).fit(iris)
        fig = shdbscan.scatter3dplot(
            x=1, y=0, z=2, palette="copper", cols=load_iris().feature_names
        )[0]
        return fig
