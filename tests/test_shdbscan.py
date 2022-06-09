import pytest
from scludam.shdbscan import SHDBSCAN
from scludam.synthetic import BivariateUniform
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
from hdbscan import HDBSCAN


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


def squarediff(a, b):
    return np.sum((np.asarray(a) - np.asarray(b)) ** 2)


# case there are no clusters
#   a. not forced -> finds none
def test_no_clusters(uniform_sample):
    shdbscan = SHDBSCAN(min_cluster_size=100).fit(uniform_sample)
    assert shdbscan.n_classes == 1
    assert np.all(shdbscan.labels == -1)
    assert shdbscan.proba.shape == (uniform_sample.shape[0], 1)


#   b. forced
#       I. by asc -> finds 1
def test_no_clusters_asc(uniform_sample):
    shdbscan = SHDBSCAN(min_cluster_size=100, allow_single_cluster=True).fit(
        uniform_sample
    )
    assert shdbscan.n_classes == 2
    assert shdbscan.proba.shape == (uniform_sample.shape[0], 2)


#       II. by aasc -> finds 1
def test_no_clusters_aasc(uniform_sample):
    shdbscan = SHDBSCAN(min_cluster_size=100, auto_allow_single_cluster=True).fit(
        uniform_sample
    )
    assert shdbscan.n_classes == 2
    assert shdbscan.proba.shape == (uniform_sample.shape[0], 2)


# case there is only 1 cluster without noise
#   a. not forced -> finds none
def test_one_cluster_no_noise(only_one_cluster_sample):
    shdbscan = SHDBSCAN(min_cluster_size=500).fit(only_one_cluster_sample)
    assert shdbscan.n_classes == 1
    assert np.all(shdbscan.labels == -1)
    assert shdbscan.proba.shape == (only_one_cluster_sample.shape[0], 1)


#  b. forced
#      I. by asc -> finds 1
def test_one_cluster_no_noise_asc(only_one_cluster_sample):
    shdbscan = SHDBSCAN(min_cluster_size=500, allow_single_cluster=True).fit(
        only_one_cluster_sample
    )
    assert shdbscan.n_classes == 1
    assert np.all(shdbscan.labels == 0)
    assert shdbscan.proba.shape == (only_one_cluster_sample.shape[0], 1)


#      II. by aasc -> finds 1
def test_one_cluster_no_noise_aasc(only_one_cluster_sample):
    shdbscan = SHDBSCAN(min_cluster_size=500, auto_allow_single_cluster=True).fit(
        only_one_cluster_sample
    )
    assert shdbscan.n_classes == 1
    assert np.all(shdbscan.labels == 0)
    assert shdbscan.proba.shape == (only_one_cluster_sample.shape[0], 1)


# case there is 1 cluster with noise
#   a. not forced -> finds none
def test_one_cluster_with_noise(one_cluster_sample):
    shdbscan = SHDBSCAN(min_cluster_size=500).fit(one_cluster_sample)
    assert shdbscan.n_classes == 1
    assert np.all(shdbscan.labels == -1)
    assert shdbscan.proba.shape == (one_cluster_sample.shape[0], 1)


#   b. forced
#       I. by asc -> finds 1
def test_one_cluster_with_noise_asc(one_cluster_sample):
    shdbscan = SHDBSCAN(min_cluster_size=500, allow_single_cluster=True).fit(
        one_cluster_sample
    )
    assert shdbscan.n_classes == 2
    assert shdbscan.proba.shape == (one_cluster_sample.shape[0], 2)


#       II. by aasc -> finds 1
def test_one_cluster_with_noise_aasc(one_cluster_sample):
    shdbscan = SHDBSCAN(min_cluster_size=500, auto_allow_single_cluster=True).fit(
        one_cluster_sample
    )
    assert shdbscan.n_classes == 2
    assert shdbscan.proba.shape == (one_cluster_sample.shape[0], 2)


# case there are 2 clusters
#   a. not forced -> finds 2
def test_two_clusters(two_clusters_sample):
    shdbscan = SHDBSCAN(min_cluster_size=250).fit(two_clusters_sample)
    assert shdbscan.n_classes == 3
    assert shdbscan.proba.shape == (two_clusters_sample.shape[0], 3)


#   b. forced
#       I. by asc -> finds 1
def test_two_clusters_asc(two_clusters_sample):
    shdbscan = SHDBSCAN(min_cluster_size=250, allow_single_cluster=True).fit(
        two_clusters_sample
    )
    assert shdbscan.n_classes == 2
    assert shdbscan.proba.shape == (two_clusters_sample.shape[0], 2)


#       II. by aasc -> finds 2
def test_two_clusters_aasc(two_clusters_sample):
    shdbscan = SHDBSCAN(min_cluster_size=250, auto_allow_single_cluster=True).fit(
        two_clusters_sample
    )
    assert shdbscan.n_classes == 3
    assert shdbscan.proba.shape == (two_clusters_sample.shape[0], 3)


#   d. not forced with 1 center -> finds 1
def test_two_clusters_one_center(two_clusters_sample):
    shdbscan = SHDBSCAN(min_cluster_size=250, allow_single_cluster=True).fit(
        two_clusters_sample, centers=(0.25, 0.25)
    )
    assert shdbscan.n_classes == 2
    assert shdbscan.proba.shape == (two_clusters_sample.shape[0], 2)
    # found correct one
    center = two_clusters_sample[shdbscan.labels == 0].mean(axis=0)
    assert squarediff(center, (0.25, 0.25)) < squarediff(center, (0.75, 0.75))


#   e. not forced with 2 centers -> finds 2
def test_two_clusters_two_centers(two_clusters_sample):
    # centers must be ignored if n_found_clusters <= centers count
    shdbscan = SHDBSCAN(min_cluster_size=250, allow_single_cluster=True).fit(
        two_clusters_sample, centers=[(30, 30), (500, 500)]
    )
    assert shdbscan.n_classes == 3
    assert shdbscan.proba.shape == (two_clusters_sample.shape[0], 3)


# case there are 3 clusters with 2 centers, and auto force -> finds 2
def test_three_clusters_two_centers(three_clusters_sample):
    shdbscan = SHDBSCAN(min_cluster_size=90, auto_allow_single_cluster=True).fit(
        three_clusters_sample, centers=[(0.75, 0.75), (0.25, 0.25)]
    )
    assert shdbscan.n_classes == 3
    assert shdbscan.proba.shape == (three_clusters_sample.shape[0], 3)
    # found correct one and cluster order preserved
    center = three_clusters_sample[shdbscan.labels == 0].mean(axis=0)
    assert squarediff(center, (0.75, 0.75)) < squarediff(center, (0.25, 0.25))
    center2 = three_clusters_sample[shdbscan.labels == 1].mean(axis=0)
    assert squarediff(center2, (0.25, 0.25)) < squarediff(center2, (0.75, 0.75))


# iris for good measure
def test_iris(iris):
    shdbscan = SHDBSCAN(min_cluster_size=20, auto_allow_single_cluster=True).fit(iris)
    assert shdbscan.n_classes == 3
    assert shdbscan.proba.shape == (iris.shape[0], 3)


def test_min_sample_parameter(iris):
    shdbscan = SHDBSCAN(min_cluster_size=20, min_samples=10).fit(iris)
    assert shdbscan.clusterer.min_samples == 10


def test_clusterer_parameter(iris):
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
    assert shdbscan.clusterer.allow_single_cluster == False
